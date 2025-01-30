#!/usr/bin/env python3
"""
arxiv_agent_cli.py

Usage:
    python arxiv_agent_cli.py "Your query"

Example:
    python arxiv_agent_cli.py "Graph neural networks for large-scale optimization"
"""

import asyncio
import os
import sys
import logging
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import simplejson as json
import argparse

import arxiv
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from openai import AsyncOpenAI

import logfire

# -------------- Logging Configuration --------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels of logs

# Create handlers
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)  # INFO level for console

file_handler = logging.FileHandler("arxiv_agent.log")
file_handler.setLevel(logging.DEBUG)  # DEBUG level for file

# Create formatters and add to handlers
formatter = logging.Formatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# -------------- Load environment --------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
DEFAULT_YEAR_RANGE = 3  # Default to 3 years if not specified
DEFAULT_ARXIV_MAX_RESULTS = 10  # Default to 10 results if not specified
logger.info(f"Using model: {MODEL_NAME}")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire="if-token-present")


# -------------- Data Models --------------
class ArxivPaper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    url: str
    published: datetime = Field(default_factory=datetime.now)
    reference_id: Optional[str] = None
    include: bool = True  # whether we keep this paper
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)  # Relevance scoring
    rejection_reason: Optional[str] = None  # Rejection reason


class AcademicPaper(BaseModel):
    title: str
    abstract: str
    introduction: str
    literature_review: str
    methodology: str
    results: str
    discussion: str
    conclusion: str
    references: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)


# This will be our "state" that the agent can read/write
# while it calls tools. Think of it as "pipeline context."
@dataclass
class PipelineState:
    original_query: str
    refined_query: str | None
    papers: List[ArxivPaper]
    openai_client: AsyncOpenAI
    year_range: int
    max_results: int


# -------------- Model + Single Agent --------------
# Create an OpenAI model. Adjust temperature or other settings as desired.
model = OpenAIModel(model_name=MODEL_NAME, api_key=OPENAI_API_KEY)

agent = Agent[PipelineState, AcademicPaper](
    model=model,
    deps_type=PipelineState,
    result_type=AcademicPaper,
    system_prompt="""You are an academic research assistant tasked with producing a state-of-the-art literature review.

Workflow Steps:
1. Call refine_query() to optimize the original_query for arXiv search
2. Use arxiv_search() to retrieve relevant papers
3. For each paper, call evaluate_paper() to assess inclusion
4. Synthesize findings into an AcademicPaper with the following fields:

Required Fields:

title: str
- Technical title reflecting core research focus
- Use standard academic paper naming conventions

abstract: str
- Comprehensive summary of key findings
- Follow standard academic abstract structure

introduction: str
- Context and motivation from referenced papers
- Clear research objectives and scope

literature_review: str
- Critical analysis of approaches using [ref_X] citations
- Compare methodologies and findings across papers

methodology: str
- Technical analysis of methods from papers
- Focus on reproducible approaches and implementations

results: str
- Synthesize quantitative and qualitative findings
- Only include results explicitly stated in papers
- Use concrete numbers and metrics when available

discussion: str
- Analyze trends, patterns, and research gaps
- Base analysis strictly on included papers
- Compare strengths and limitations of approaches

conclusion: str
- Summarize key insights and contributions
- Present future directions supported by findings

references: List[str]
- Only include evaluated papers marked for inclusion
- Use provided reference_ids consistently for citations
- Follow the [ref_X] format

keywords: List[str]
- Extract directly from paper titles and abstracts
- No synthesized or inferred keywords
- Focus on technical and domain-specific terms

Guidelines:
- Maintain formal academic writing style
- Use [ref_X] format for all in-text citations
- Only include content from evaluated papers
- Return a valid AcademicPaper object
- Each section should reference multiple papers where appropriate
- Maintain consistent terminology throughout
- Ensure all claims are supported by citations""",
)

# -------------- Tools --------------


@agent.tool  # Tool #1
async def refine_query(ctx: RunContext[PipelineState], user_query: str) -> str:
    """
    Improve the search query specifically for arXiv's search engine with date filtering.
    Return only the refined query string in unencoded format for the Python arxiv library.
    """
    logger.debug("Starting refine_query tool")
    # Calculate date range for last `year_range` years
    current_year = datetime.now().year
    current_month = datetime.now().month
    from_date = (
        f"{current_year - ctx.deps.year_range}{str(current_month).zfill(2)}010000"
    )
    to_date = f"{current_year}{str(current_month).zfill(2)}312359"

    prompt = {
        "role": "system",
        "content": (
            "You are an expert in crafting arXiv search queries. Your task is to refine the given query "
            "following these specific guidelines:\n\n"
            "1. Use arXiv's field prefixes:\n"
            "   - ti: for title search\n"
            "   - au: for author search\n"
            "   - abs: for abstract search\n"
            "   - all: all fields\n"
            "2. For phrases, use double quotes (normal quotes, not encoded)\n"
            "3. Use proper Boolean operators:\n"
            "   - AND (default between terms)\n"
            "   - OR (for alternatives)\n"
            "   - ANDNOT (for exclusions)\n"
            "4. Group expressions using normal parentheses ( )\n"
            "5. Include relevant synonyms and alternative phrasings with OR\n"
            "6. Use regular spaces, not plus signs\n"
            "7. Focus on technical/scientific terminology\n"
            "8. Consider key methodologies or techniques\n\n"
            "Examples:\n"
            'Input: "bgp quic secure transport"\n'
            'Output: all:"BGP" AND (all:"TLS" OR all:"QUIC" OR all:"secure transport")\n\n'
            'Input: "deep learning for computer vision"\n'
            'Output: (ti:"deep learning" OR abs:"deep learning") AND ("computer vision" OR "image recognition" OR "visual recognition")\n\n'
            'Input: "quantum computing optimization"\n'
            'Output: (ti:"quantum computing" OR abs:"quantum computing") AND (optimization OR "quantum optimization" OR QAOA)\n\n'
            "Return ONLY the refined query string. The date range will be added automatically."
        ),
    }

    user_message = {
        "role": "user",
        "content": f"Craft an arXiv search query for: {user_query}",
    }

    try:
        response = await ctx.deps.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[prompt, user_message],
            temperature=0.3,
            max_tokens=200,
        )
        refined_query = response.choices[0].message.content.strip()
        if not refined_query:
            logger.warning("Received empty refined query. Using original query.")
            refined_query = user_query

        # Add date range to the query
        date_filter = f" AND submittedDate:[{from_date} TO {to_date}]"
        refined_query = refined_query + date_filter

        ctx.deps.refined_query = refined_query
        logger.info(f"Refined query: {refined_query}")
        return refined_query

    except Exception as e:
        logger.exception("Error in refine_query tool")
        raise ModelRetry(f"Failed to refine query: {str(e)}") from e


class AsyncIteratorWrapper:
    def __init__(self, iterator):
        self._iterator = iterator

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            value = next(self._iterator)
        except StopIteration:
            raise StopAsyncIteration
        return value


@agent.tool  # Tool #2
async def arxiv_search(ctx: RunContext[PipelineState]) -> List[ArxivPaper]:
    """
    Use the refined_query from the pipeline state to search on arXiv (CS category).
    Return a list of ArxivPaper objects with reference IDs.
    """
    logger.debug("Starting arxiv_search tool")
    if not ctx.deps.refined_query:
        logger.error("No refined query found. Call refine_query(...) first.")
        raise ModelRetry("No refined query found. Call refine_query(...) first.")

    refined_q = ctx.deps.refined_query
    logger.info(f"Refined query for arXiv search: {refined_q}")

    try:
        # Actually query arxiv
        client = arxiv.Client()
        search = arxiv.Search(
            query=refined_q,
            max_results=ctx.deps.max_results,
            sort_by=arxiv.SortCriterion.Relevance,
            sort_order=arxiv.SortOrder.Descending,
        )
        # We'll gather results asynchronously using the AsyncIteratorWrapper
        results = []
        ref_id_counter = 1
        async for result in AsyncIteratorWrapper(client.results(search)):
            ref_id = f"ref_{ref_id_counter:02d}"
            paper = ArxivPaper(
                title=result.title,
                authors=[str(a) for a in result.authors],
                abstract=result.summary,
                url=result.pdf_url,
                published=result.published,
                reference_id=ref_id,
            )
            results.append(paper)
            logger.debug(f"Found paper: {paper.title}")
            ref_id_counter += 1

        logger.info(f"Total papers found: {len(results)}")

        # Store them in pipeline state
        ctx.deps.papers = results
        return results

    except Exception as e:
        logger.exception("Error during arxiv_search tool execution")
        raise ModelRetry(f"Failed to search arXiv: {str(e)}") from e


@agent.tool  # Tool #3
async def evaluate_paper(
    ctx: RunContext[PipelineState], paper_index: int
) -> ArxivPaper:
    """
    Evaluate the paper at index 'paper_index' in ctx.deps.papers.
    Return the updated ArxivPaper with 'include=True' if relevant, else False.
    """
    logger.debug(f"Starting evaluate_paper tool for paper index: {paper_index}")
    if paper_index < 0 or paper_index >= len(ctx.deps.papers):
        logger.warning(
            f"paper_index {paper_index} out of range (0-{len(ctx.deps.papers)-1})"
        )
        return None

    paper = ctx.deps.papers[paper_index]
    logger.info(f"Evaluating paper: {paper.title}")

    prompt = {
        "role": "system",
        "content": """You are a paper evaluator that returns ONLY a JSON object.
Format: {"score": float between 0-1, "reason": "brief explanation"}

Evaluate paper relevance based on:
- Technical match with query
- Methodology quality
- Results significance
- Innovation level

Return ONLY the JSON object, nothing else.""",
    }

    user_message = {
        "role": "user",
        "content": f"""Query: {ctx.deps.original_query}
Title: {paper.title}
Abstract: {paper.abstract}

Return JSON only.""",
    }

    try:
        response = await ctx.deps.openai_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[prompt, user_message],
            temperature=0.2,
            max_tokens=100,
            response_format={"type": "json_object"},
        )

        response_text = response.choices[0].message.content.strip()
        if not response_text:
            logger.warning("Empty response from model. Using default evaluation.")
            evaluation = {"score": 0.0, "reason": "Empty evaluation response"}
        else:
            # Parse JSON response
            try:
                evaluation = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}")
                logger.debug(f"Response text: {response_text}")
                # Fallback scoring if JSON parsing fails
                evaluation = {"score": 0.0, "reason": "Failed to parse evaluation"}

        # Validate score range
        score = float(evaluation.get("score", 0.0))
        score = max(0.0, min(1.0, score))  # Clamp between 0 and 1

        # Update paper in state
        paper.relevance_score = score
        paper.include = score >= 0.7  # Threshold for inclusion

        # Store rejection reason if applicable
        if "reason" in evaluation:  # Check if reason is present
            paper.rejection_reason = evaluation["reason"]

        # Improved Evaluation Logging
        logger.info(
            f"Evaluated Paper:\n"
            f"  Title: {paper.title}\n"
            f"  Score: {paper.relevance_score:.2f}\n"
            f"  Reason: {paper.rejection_reason}"
        )

        ctx.deps.papers[paper_index] = paper
        return paper

    except Exception as e:
        logger.exception(f"Error evaluating paper '{paper.title}'")
        raise ModelRetry(f"Failed to evaluate paper: {str(e)}") from e


# -------------- Putting it All Together --------------
# Because result_type=AcademicPaper, the final output from the model must be
# a valid AcademicPaper. The agent can gather data from the pipeline state
# and produce it in structured form.


# -------------- Paper Generation Logic --------------
def format_reference(paper: ArxivPaper) -> str:
    """
    Format a paper reference with reference ID for cross-referencing.
    Format: [ref_id] Authors (Year). Title. [arXiv:ID](https://arxiv.org/abs/ID)
    """
    authors = ", ".join(paper.authors[:3]) + (
        " et al." if len(paper.authors) > 3 else ""
    )
    year = paper.published.year
    arxiv_id = paper.url.split("/")[-1]
    reference = (
        f"[{paper.reference_id}] {authors} ({year}). {paper.title}. "
        f"[arXiv:{arxiv_id}](https://arxiv.org/abs/{arxiv_id})"
    )
    logger.debug(f"Formatted reference: {reference}")
    return reference


async def run_pipeline(
    original_query: str,
    year_range: int = DEFAULT_YEAR_RANGE,
    max_results: int = DEFAULT_ARXIV_MAX_RESULTS,
) -> AcademicPaper:
    """
    Kick off the single-run pipeline.
    Args:
        original_query: The search query
        year_range: Number of years to look back (default: 5)

    The LLM is instructed to call the relevant function tools
    and eventually produce a AcademicPaper as final output.
    """
    logger.info("Starting pipeline")
    pipeline_state = PipelineState(
        original_query=original_query,
        openai_client=openai_client,
        papers=[],
        refined_query=None,
        year_range=year_range,
        max_results=max_results,
    )
    usage_limits = UsageLimits(request_limit=20, total_tokens_limit=25000)

    try:
        # We'll pass the user's prompt as the initial user message
        # i.e. "I want to search for X"
        # The system prompt already instructs the model to call the relevant tools.
        result = await agent.run(
            original_query,
            deps=pipeline_state,
            usage_limits=usage_limits,
        )
        included_papers = [p for p in pipeline_state.papers if p.include]
        result.data.references = [
            format_reference(p)
            for p in sorted(
                included_papers, key=lambda x: x.relevance_score, reverse=True
            )
        ]
        logger.info("Pipeline completed successfully")
        return result.data

    except Exception as e:
        logger.exception("Pipeline failed")
        raise e


# -------------- Enhanced Output Formatting --------------
def print_academic_paper(paper: AcademicPaper):
    logger.debug("Printing academic paper to console")
    print(f"# {paper.title}\n")
    print("## Abstract\n")
    print(paper.abstract + "\n")

    sections = [
        ("Introduction", paper.introduction),
        ("Literature Review", paper.literature_review),
        ("Methodology", paper.methodology),
        ("Results", paper.results),
        ("Discussion", paper.discussion),
        ("Conclusion", paper.conclusion),
    ]

    for header, content in sections:
        print(f"## {header}\n")
        print(content + "\n")

    print("## References\n")
    for ref in paper.references:
        print(f"- {ref}")


async def save_academic_paper_markdown(
    paper: AcademicPaper, output_dir: str = "generated_papers"
) -> str:
    """
    Save the academic paper as a markdown file in a specified directory.
    Args:
        paper: The academic paper to save
        output_dir: Directory where to save the markdown files (default: 'generated_papers')
    Returns:
        str: Path to the saved file
    """
    logger.debug(f"Saving academic paper '{paper.title}' as markdown")
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        logger.debug(f"Ensured output directory exists: {output_dir}")

        # Create a filename from the paper title
        safe_title = "".join(
            c for c in paper.title if c.isalnum() or c.isspace()
        ).rstrip()
        safe_title = safe_title.replace(" ", "_").lower()
        filename = f"{safe_title}_{datetime.now().strftime('%Y%m%d')}.md"
        output_path = os.path.join(output_dir, filename)

        markdown_content = [
            f"# {paper.title}\n",
            "## Abstract\n",
            f"{paper.abstract}\n",
            "## Keywords\n",
            ", ".join(paper.keywords) + "\n",
            "## Introduction\n",
            f"{paper.introduction}\n",
            "## Literature Review\n",
            f"{paper.literature_review}\n",
            "## Methodology\n",
            f"{paper.methodology}\n",
            "## Results\n",
            f"{paper.results}\n",
            "## Discussion\n",
            f"{paper.discussion}\n",
            "## Conclusion\n",
            f"{paper.conclusion}\n",
            "## References\n",
            "\n".join(f"- {ref}" for ref in paper.references),
        ]

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(markdown_content))

        logger.info(f"Markdown report saved to: {output_path}")
        return output_path

    except Exception as e:
        logger.exception("Error saving academic paper as markdown")
        raise e


async def main():
    parser = argparse.ArgumentParser(
        description="Search arXiv and generate academic literature review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s "Graph neural networks for optimization"
    %(prog)s --years 3 --output papers "Quantum computing advances"
    %(prog)s --max-results 20 "Machine learning in biology"
        """,
    )

    parser.add_argument("query", type=str, help="Search query for arXiv papers")

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="generated_papers",
        help="Output directory for generated markdown files (default: generated_papers)",
    )

    parser.add_argument(
        "-y",
        "--years",
        type=int,
        default=DEFAULT_YEAR_RANGE,
        help=f"Number of years to look back in search (default: {DEFAULT_YEAR_RANGE})",
    )

    parser.add_argument(
        "-m",
        "--max-results",
        type=int,
        default=DEFAULT_ARXIV_MAX_RESULTS,
        help="Maximum number of papers to retrieve (default: 10)",
    )

    args = parser.parse_args()

    logger.info("Starting main execution")
    logger.debug(f"Parsed arguments: {args}")

    try:
        # Run the pipeline with the specified parameters
        report = await run_pipeline(
            original_query=args.query,
            year_range=args.years,
            max_results=args.max_results,
        )

        if not report or report.references == []:
            logger.warning("No summary report generated.")
            return

        # Print to console
        print_academic_paper(report)

        # Save to markdown file
        try:
            output_file = await save_academic_paper_markdown(report, args.output)

        except Exception as e:
            logger.error(f"Error saving markdown file: {str(e)}")

    except Exception as e:
        logger.critical("An unexpected error occurred during execution", exc_info=True)
        print("An unexpected error occurred. Please check the logs for more details.")


if __name__ == "__main__":
    asyncio.run(main())

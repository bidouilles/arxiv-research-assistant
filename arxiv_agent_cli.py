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
from datetime import datetime
from typing import List, Optional
from dataclasses import dataclass
import simplejson as json

import arxiv
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.usage import UsageLimits
from openai import AsyncOpenAI

import logfire

# -------------- Load environment --------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL_NAME = os.getenv("LLM_MODEL", "gpt-4o-mini")
print(f"Using model: {MODEL_NAME}")

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
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)  # New relevance scoring


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
    # Calculate date range for last 5 years
    current_year = datetime.now().year
    current_month = datetime.now().month
    from_date = f"{current_year-5}{str(current_month).zfill(2)}010000"
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

    response = await ctx.deps.openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt, user_message],
        temperature=0.3,
        max_tokens=200,
    )

    refined_query = response.choices[0].message.content
    if refined_query is None:
        refined_query = user_query

    # Add date range to the query
    date_filter = f" AND submittedDate:[{from_date} TO {to_date}]"
    refined_query = refined_query + date_filter

    ctx.deps.refined_query = refined_query
    print(f"Refined query: {refined_query}")
    return refined_query


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
async def arxiv_search(
    ctx: RunContext[PipelineState], max_results: int = 10
) -> List[ArxivPaper]:
    """
    Use the refined_query from the pipeline state to search on arXiv (CS category).
    Return a list of ArxivPaper objects with reference IDs.
    """
    if not ctx.deps.refined_query:
        raise ModelRetry("No refined query found. Call refine_query(...) first.")
    refined_q = ctx.deps.refined_query
    print(f"Refined query: {refined_q}")

    # Actually query arxiv
    client = arxiv.Client()
    search = arxiv.Search(
        query=refined_q,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
        sort_order=arxiv.SortOrder.Descending,
    )
    # We'll gather results synchronously, but we want to be asynchronous
    # let's do a small async wrapper:
    results = []
    ref_id_counter = 1
    async for result in AsyncIteratorWrapper(client.results(search)):
        ref_id = f"ref_{ref_id_counter}"
        paper = ArxivPaper(
            title=result.title,
            authors=[str(a) for a in result.authors],
            abstract=result.summary,
            url=result.pdf_url,
            published=result.published,
            reference_id=ref_id,
        )
        results.append(paper)
        ref_id_counter += 1
        print(f"Found paper: {paper.title}")

    # store them in pipeline state
    ctx.deps.papers = results
    return results


@agent.tool  # Tool #3
async def evaluate_paper(
    ctx: RunContext[PipelineState], paper_index: int
) -> ArxivPaper:
    """
    Evaluate the paper at index 'paper_index' in ctx.deps.papers.
    Return the updated ArxivPaper with 'include=True' if relevant, else False.
    """
    if paper_index < 0 or paper_index >= len(ctx.deps.papers):
        raise ValueError("paper_index out of range")

    paper = ctx.deps.papers[paper_index]

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

    response = await ctx.deps.openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt, user_message],
        temperature=0.2,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    response_text = response.choices[0].message.content
    if not response_text:
        raise ValueError("Empty response from model")

    # Parse JSON response
    try:
        evaluation = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response text: {response_text}")
        # Fallback scoring if JSON parsing fails
        evaluation = {"score": 0.0, "reason": "Failed to parse evaluation"}

    # Validate score range
    score = float(evaluation.get("score", 0.0))
    score = max(0.0, min(1.0, score))  # Clamp between 0 and 1

    # Update paper in state
    paper.relevance_score = score
    paper.include = score >= 0.7  # Threshold for inclusion

    print(
        f"Evaluated {paper.title}: Score {paper.relevance_score:.2f} - Reason: {evaluation['reason']}"
    )

    ctx.deps.papers[paper_index] = paper
    return paper


# -------------- Putting it All Together --------------
# Because result_type=AcademicPaper, the final output from the model must be
# a valid AcademicPaper. The agent can gather data from the pipeline state
# and produce it in structured form.


# -------------- Paper Generation Logic --------------
def format_reference(paper: ArxivPaper) -> str:
    """
    Format a paper reference with reference ID for cross-referencing.
    Format: [ref_id] Authors (Year). Title. arXiv:ID
    """
    authors = ", ".join(paper.authors[:3]) + (
        " et al." if len(paper.authors) > 3 else ""
    )
    year = paper.published.year
    arxiv_id = paper.url.split("/")[-1]
    return f"[{paper.reference_id}] {authors} ({year}). {paper.title}. arXiv:{arxiv_id}"


async def run_pipeline(original_query: str) -> AcademicPaper:
    """
    Kick off the single-run pipeline.
    The LLM is instructed to call the relevant function tools
    and eventually produce a AcademicPaper as final output.
    """
    pipeline_state = PipelineState(
        original_query=original_query,
        openai_client=openai_client,
        papers=[],
        refined_query=None,
    )
    usage_limits = UsageLimits(request_limit=20, total_tokens_limit=20000)

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
        for p in sorted(included_papers, key=lambda x: x.relevance_score, reverse=True)
    ]
    return result.data


# -------------- Enhanced Output Formatting --------------
def print_academic_paper(paper: AcademicPaper):
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
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create a filename from the paper title
    safe_title = "".join(c for c in paper.title if c.isalnum() or c.isspace()).rstrip()
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

    return output_path


async def main():
    if len(sys.argv) < 2:
        print('Usage: python arxiv_agent_cli.py "Your query" [output_directory]')
        sys.exit(1)

    user_query = sys.argv[1]
    # Allow optional output directory specification
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "generated_papers"

    report = await run_pipeline(user_query)
    if not report:
        print("No summary report generated.")
        return

    # Print to console
    print_academic_paper(report)

    # Save to markdown file
    try:
        output_file = await save_academic_paper_markdown(report, output_dir)
        print(f"\nMarkdown report saved to: {output_file}")

        # Print the contents of the output directory
        files = os.listdir(output_dir)
        print(f"\nContents of {output_dir}/:")
        for file in files:
            print(f"- {file}")

    except Exception as e:
        print(f"\nError saving markdown file: {str(e)}")


if __name__ == "__main__":
    asyncio.run(main())

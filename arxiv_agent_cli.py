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


class SummaryReport(BaseModel):
    query: str
    title: str
    papers: List[ArxivPaper]
    summary: str
    state_of_the_art_analysis: str
    references: List[str] = Field(default_factory=list)


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

agent = Agent[PipelineState, SummaryReport](
    model=model,
    deps_type=PipelineState,
    result_type=SummaryReport,
    # system_prompt here will instruct the model on how to use the tools,
    # the result schema, and your "workflow" instructions.
    system_prompt=(
        "You are an advanced academic research assistant with access to function tools. "
        "Given the user's original query (stored in `original_query`), you must:\n\n"
        "1) Refine the query using the refine_query(...) tool.\n"
        "2) Search arXiv via arxiv_search(...).\n"
        "3) For each paper found, call evaluate_paper(...) to decide if we include it.\n"
        "4) Finally, return a SummaryReport with the fields:\n"
        "   - query: The refined query.\n"
        "   - title: A short academic-style title summarizing these papers.\n"
        "   - papers: The included ArxivPaper objects.\n"
        "   - summary: Summarize each included paper.\n"
        "   - state_of_the_art_analysis: Provide an advanced analysis of the SoTA.\n"
        "   - references: Provide a references list in the specified format.\n\n"
        "IMPORTANT:\n"
        " - Only use the provided tools to gather info or refine. Do not fabricate.\n"
        " - If no relevant papers are found, produce an empty list.\n"
        " - Use function calls for all structured steps.\n"
        " - End the run with a valid SummaryReport object."
    ),
)

# -------------- Tools --------------


@agent.tool  # Tool #1
async def refine_query(ctx: RunContext[PipelineState], user_query: str) -> str:
    """
    Improve the search query specifically for arXiv's search engine with date filtering.
    Return only the refined query string in unencoded format for the Python arxiv library.
    """
    # Calculate date range for last 3 years
    current_year = datetime.now().year
    current_month = datetime.now().month
    from_date = f"{current_year-3}{str(current_month).zfill(2)}010000"
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
            "   - co: for comments\n"
            "   - jr: for journal reference\n"
            "   - cat: for subject category\n"
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
        "content": (
            "Evaluate if this paper is highly relevant to the search query. "
            "Consider: relevance to query, methodology, and novelty. "
            "Respond with only 'true' or 'false'."
        ),
    }

    user_message = {
        "role": "user",
        "content": (
            f"Query: {ctx.deps.original_query}\n"
            f"Title: {paper.title}\n"
            f"Abstract: {paper.abstract}"
        ),
    }

    response = await ctx.deps.openai_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[prompt, user_message],
        temperature=0.2,
        max_tokens=50,
    )

    # Parse response to boolean
    include_paper = response.choices[0].message.content.strip().lower() == "true"

    print(f"Paper {paper.title} is {'INCLUDED' if include_paper else 'excluded'}")

    # Update paper in state
    paper.include = include_paper
    ctx.deps.papers[paper_index] = paper
    return paper


# -------------- Putting it All Together --------------
# Because result_type=SummaryReport, the final output from the model must be
# a valid SummaryReport. The agent can gather data from the pipeline state
# and produce it in structured form.


async def run_pipeline(original_query: str) -> SummaryReport:
    """
    Kick off the single-run pipeline.
    The LLM is instructed to call the relevant function tools
    and eventually produce a SummaryReport as final output.
    """
    pipeline_state = PipelineState(
        original_query=original_query,
        openai_client=openai_client,
        papers=[],
        refined_query=None,
    )
    usage_limits = UsageLimits(request_limit=20, total_tokens_limit=15000)

    # We'll pass the user's prompt as the initial user message
    # i.e. "I want to search for X"
    # The system prompt already instructs the model to call the relevant tools.
    result = await agent.run(
        original_query,
        deps=pipeline_state,
        usage_limits=usage_limits,
    )
    return result.data


async def main():
    if len(sys.argv) != 2:
        print('Usage: python arxiv_agent_cli.py "Your query"')
        sys.exit(1)

    user_query = sys.argv[1]
    report = await run_pipeline(user_query)
    if not report:
        print("No summary report generated.")
        return

    # Pretty-print final output
    print(f"# {report.title}\n")
    print("## Papers\n")
    for p in report.papers:
        if p.include:
            print(f"- **{p.title}**  (ID: {p.reference_id})")
            print(f"  - Authors: {', '.join(p.authors)}")
            print(f"  - URL: {p.url}")
            print(f"  - Published: {p.published}")
            print(f"  - Abstract:\n    {p.abstract}\n")

    print("## Summary\n")
    print(report.summary)

    print("\n## State of the Art Analysis\n")
    print(report.state_of_the_art_analysis)

    print("\n## References\n")
    for ref in report.references:
        print(f"- {ref}")


if __name__ == "__main__":
    asyncio.run(main())

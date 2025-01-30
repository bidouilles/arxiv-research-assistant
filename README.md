# ArXiv Agent CLI

A command-line tool for generating comprehensive academic research papers based on arXiv searches.

## Overview

The ArXiv Agent CLI is a powerful tool that uses AI to search arXiv papers, evaluate their relevance, and automatically generate structured academic papers. It combines the capabilities of OpenAI's language models with arXiv's research database to create detailed literature reviews and research summaries.

## Installation

### Prerequisites

- Python 3.7+
- OpenAI API key
- Required Python packages (install via pip):
  - arxiv
  - pydantic
  - pydantic-ai
  - openai
  - python-dotenv
  - simplejson
  - logfire

### Environment Setup

1. Create a `.env` file in your project directory
2. Add your OpenAI API key:
```
OPENAI_API_KEY=your_key_here
LLM_MODEL=gpt-4o-mini
```

## Usage

Basic command line usage:

```bash
python arxiv_agent_cli.py "Your research query"
```

Example:
```bash
python arxiv_agent_cli.py "Graph neural networks for large-scale optimization"
```

## Features

### 1. Query Refinement
- Automatically improves search queries for arXiv's search engine
- Uses field prefixes (ti:, au:, abs:, etc.)
- Incorporates Boolean operators and proper syntax
- Adds date filtering for recent papers

### 2. Paper Search
- Searches arXiv using refined queries
- Retrieves detailed paper information including:
  - Title
  - Authors
  - Abstract
  - URL
  - Publication date

### 3. Paper Evaluation
- Automatically evaluates paper relevance
- Scores papers based on:
  - Technical rigor
  - Innovation level
  - Citation potential
  - Methodology soundness
  - Results significance

### 4. Academic Paper Generation
Generates a structured academic paper with:
- Title
- Abstract
- Introduction
- Literature Review
- Methodology
- Results
- Discussion
- Conclusion
- References
- Keywords

## Architecture

### Data Models

#### ArxivPaper
```python
class ArxivPaper(BaseModel):
    title: str
    authors: List[str]
    abstract: str
    url: str
    published: datetime
    reference_id: Optional[str]
    include: bool
    relevance_score: float
```

#### AcademicPaper
```python
class AcademicPaper(BaseModel):
    title: str
    abstract: str
    introduction: str
    literature_review: str
    methodology: str
    results: str
    discussion: str
    conclusion: str
    references: List[str]
    keywords: List[str]
```

### Pipeline State
```python
@dataclass
class PipelineState:
    original_query: str
    refined_query: str | None
    papers: List[ArxivPaper]
    openai_client: AsyncOpenAI
```

## Tools

### 1. refine_query
Improves the search query for arXiv's search engine with date filtering.

### 2. arxiv_search
Searches arXiv using the refined query and returns a list of ArxivPaper objects.

### 3. evaluate_paper
Evaluates individual papers for relevance and updates their inclusion status.

## Output Format

The tool generates a well-structured academic paper with clear section headers:

```markdown
# Paper Title

## Abstract
[Generated abstract]

## Introduction
[Generated introduction]

## Literature Review
[Generated literature review]

## Methodology
[Generated methodology section]

## Results
[Generated results section]

## Discussion
[Generated discussion]

## Conclusion
[Generated conclusion]

## References
- [Reference 1]
- [Reference 2]
...
```

## Error Handling

- Validates input parameters
- Handles API errors gracefully
- Provides clear error messages for troubleshooting
- Implements retry logic for failed API calls

## Limitations

- Maximum of 10 papers per search by default
- Requires OpenAI API access
- Limited to papers available on arXiv
- Papers must be from the last 5 years
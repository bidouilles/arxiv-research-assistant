# arXiv Research Assistant

A powerful CLI tool that searches arXiv for academic papers, evaluates their relevance, and automatically generates comprehensive literature reviews. This tool uses GPT-4 to refine searches, evaluate papers, and synthesize findings into well-structured academic papers.

## Features

- 🔍 Smart query refinement optimized for arXiv's search engine
- 📊 Automatic paper relevance scoring and filtering
- 📝 Generates complete academic papers with proper citations
- 🤖 Powered by GPT-4 for intelligent paper analysis
- 📅 Configurable date range for recent research
- 💾 Exports results in clean Markdown format

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/arxiv-research-assistant.git
cd arxiv-research-assistant

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Configuration

1. Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_api_key_here
LLM_MODEL=gpt-4o-mini  # or your preferred model
```

2. Make the script executable:
```bash
chmod +x arxiv_agent_cli.py
```

## Usage

```bash
python arxiv_agent_cli.py [options] "Your research query"
```

### Options

- `-y, --years`: Number of years to look back (default: 5)
- `-m, --max-results`: Maximum number of papers to retrieve (default: 10)
- `-o, --output`: Output directory for generated markdown files (default: generated_papers)

### Examples

#### Computer Network Security with BGP

1. BGP Security Research:
```bash
python arxiv_agent_cli.py "BGP security mechanisms and attack prevention"
```

2. RPKI and ROV Analysis:
```bash
python arxiv_agent_cli.py "RPKI and ROV deployment in BGP infrastructure"
```

3. BGP with QUIC Integration:
```bash
python arxiv_agent_cli.py "BGP security with QUIC protocol integration"
```

4. BGP Route Leaks Prevention:
```bash
python arxiv_agent_cli.py "Detection and prevention of BGP route leaks"
```

#### Sample Output Structure

The tool generates a comprehensive academic paper with the following sections:

```markdown
# Title

## Abstract
[Comprehensive summary of findings]

## Keywords
[Relevant technical terms]

## Introduction
[Context and research objectives]

## Literature Review
[Analysis of current research]

## Methodology
[Technical approaches used]

## Results
[Key findings and metrics]

## Discussion
[Analysis and implications]

## Conclusion
[Summary and future directions]

## References
[Cited papers with arXiv links]
```

## Agent Workflow

The research assistant uses a sophisticated agent-based architecture to process your queries and generate comprehensive literature reviews. Here's how it works:

### Pipeline Flow

1. **Query Refinement (Tool 1)**
   - Takes your natural language query
   - Optimizes it for arXiv's search engine
   - Adds field-specific prefixes and Boolean operators
   - Includes date range filtering

2. **arXiv Search (Tool 2)**
   - Uses refined query to search arXiv
   - Retrieves papers based on relevance
   - Sorts results by importance
   - Assigns unique reference IDs

3. **Paper Evaluation (Tool 3)**
   - Evaluates each paper individually
   - Scores papers on 0-1 scale based on:
     * Technical relevance
     * Methodology quality
     * Results significance
     * Innovation level
   - Papers with scores ≥ 0.7 are included

4. **Paper Synthesis**
   - Combines findings from included papers
   - Generates structured academic paper
   - Maintains consistent citations
   - Creates coherent narrative

The entire process is orchestrated by a GPT-4 powered agent that:
- Maintains context throughout the pipeline
- Makes intelligent decisions about paper inclusion
- Ensures citation consistency
- Generates academically rigorous output

## Features in Detail

### Query Refinement
The tool automatically refines your search query using arXiv's field prefixes and Boolean operators:
- `ti:` for title search
- `abs:` for abstract search
- `au:` for author search
- Proper use of AND, OR, ANDNOT operators
- Date range filtering

### Paper Evaluation
Each paper is automatically evaluated based on:
- Technical relevance to the query
- Methodology quality
- Results significance
- Innovation level

### Output Generation
- Markdown files with proper academic structure
- In-text citations using [ref_X] format
- Sorted references by relevance
- Clean, readable formatting

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- arXiv API for providing access to research papers
- OpenAI for GPT-4 API
- All contributors and users of this tool

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{arxiv_research_assistant,
  author = {bidouilles},
  title = {arXiv Research Assistant},
  year = {2024},
  url = {https://github.com/yourusername/arxiv-research-assistant}
}
```
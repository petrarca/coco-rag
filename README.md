# CocoRAG

A semantic code search and indexing system that enables AI-powered code discovery and analysis. CocoRAG (Code Context Retrieval Augmented Generation) indexes your codebases using embeddings and provides powerful search capabilities through both interactive CLI and MCP server interfaces.

## Features

- **Semantic Code Search**: Find code snippets using natural language queries with advanced vector similarity
- **Advanced Reranking System**: Multiple reranking strategies including intelligent auto-selection, feature-based reranking, and NLP-based semantic analysis
- **Multi-Source Support**: Index multiple codebases with flexible pattern matching and topic-based organization
- **MCP Server Integration**: Expose search functionality to AI assistants via Model Context Protocol
- **Interactive CLI**: Rich command-line interface with syntax highlighting and dynamic control
- **Flexible Configuration**: Powerful pattern extension system for fine-grained control over indexed files
## Architecture

See [docs/design/architecture.md](docs/design/architecture.md) for the full architecture overview and [docs/design/reranker-design.md](docs/design/reranker-design.md) for the reranking system design.

## Configuration

CocoRAG is configured via a YAML file and environment variables. Settings priority:

1. **Command-line arguments** (highest priority)
2. **Environment variables** (`COCO_RAG_*` prefix)
3. **YAML configuration file** (`config.yml`)
4. **Default values**

Create `config.yml` (see `config.example.yml` for reference):

```yaml
sources:
  - name: "my_project"
    type: "local_file"
    path: "/path/to/your/project"
    included_patterns: ["*.ts", "*.js", "*.py"]
    excluded_patterns: ["*.test.js", "node_modules/**"]

settings:
  chunk_size: 1000
  min_chunk_size: 300
  chunk_overlap: 300
  embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
  reranking_enabled: true
  reranking_type: "auto"
```

## Getting Started

### Prerequisites

- Python 3.14+
- PostgreSQL 12+ with pgvector extension
- [uv](https://github.com/astral-sh/uv) package manager
- [Task](https://taskfile.dev/) task runner (optional but recommended)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/petrarca/coco-rag.git
cd coco-rag
```

2. **Start PostgreSQL with pgvector** (optional, if you don't have existing setup):
```bash
docker compose up -d
```

3. **Set up environment and install dependencies**:
```bash
task rebuild    # Clean all files, create virtual environment, install dependencies, and run tests
```

4. **Configure your environment**:
```bash
cp .env.example .env
cp config.example.yml config.yml
```

5. **Edit `config.yml`** to point at your source directories:
```yaml
sources:
  - name: "my_project"
    topic: "my_project"
    type: "local_file"
    path: "/path/to/your/source/code"
```

6. **Edit `.env`** with your database credentials (must match docker-compose):
```bash
COCOINDEX_DATABASE_URL=postgres://cocorag:cocorag@localhost:6432/cocorag
```

7. **Set up and index your code**:
```bash
task rag:setup     # Create DB schema and flow infrastructure
task rag:update    # Index your source files (incremental)
```

8. **Start searching**:
```bash
task run:cli       # Interactive CLI search
# or
task run:mcp:http  # MCP server for AI assistants
```

## Usage

### Interactive CLI

```bash
# Start interactive search mode
task run:cli

# With custom config file
task run:cli -- --config /path/to/custom-config.yml

# Interactive commands:
# /help           - Show available commands
# /topics         - List all available topics
# /topic <name>   - Set topic filter
# /reranker <type> - Set reranker (auto, pure_functional, spacy_nlp, disabled)
# /reset          - Clear topic filter
# /quit           - Exit
```

### Index Management

```bash
task rag:setup      # Initial index setup
task rag:update     # Incremental update with statistics
task rag:drop       # Drop all indexes

# With custom config file
task rag:setup -- --config /path/to/custom-config.yml
task rag:update -- --config /path/to/custom-config.yml
```

### MCP Server

```bash
# stdio transport (default for AI assistants)
task run:mcp

# HTTP transport (for web integrations)
task run:mcp:http

# With custom config file
task run:mcp -- --config /path/to/custom-config.yml
```

**Available MCP Tools**:
- `search(query, top_k=10, topic=None, reranker="auto")` → Ranked code chunks
- `list_topics()` → List all available topics
- `get_file(filename, topic=None, start_line=None, end_line=None)` → Full file content reassembled from chunks
- `list_files(topic=None, path_prefix=None, pattern=None, limit=100)` → Browse indexed files

### Direct Python Invocation

```bash
# Interactive search with custom config
python -m coco_rag.main main --config /path/to/custom-config.yml

# Index management with custom config
python -m coco_rag.main setup --config /path/to/custom-config.yml
python -m coco_rag.main update --config /path/to/custom-config.yml

# MCP server with custom config
python -m coco_rag.main mcp --config /path/to/custom-config.yml
```

## Development

### Code Quality

```bash
task format    # Format code using ruff
task check     # Check and fix code using ruff  
task test      # Run tests with pytest
task fct       # Run format, check, and test in sequence
task rebuild   # Clean all files, reinstall, and run tests
```

### Key Tasks

**Core Operations**:
- `task setup` - Create virtual environment
- `task install` - Install dependencies
- `task rebuild` - Complete clean rebuild
- `task clean:all` - Remove all generated files

**CocoRAG Operations**:
- `task rag:setup` - Initialize index
- `task rag:update` - Update index
- `task run:cli` - Interactive search
- `task run:mcp` - Start MCP server

**Optional Components**:
- `task spacy:install` - Install spaCy for advanced NLP reranking

## Environment Variables

Create a `.env` file based on `.env.example`:

```bash
COCO_RAG_CONFIG="./config.yml"
COCOINDEX_DATABASE_URL=postgres://cocorag:cocorag@localhost:6432/cocorag
```

See `.env.example` for the full list of optional overrides (reranking, chunking, MCP server, logging).

## Troubleshooting

### Common Issues

**Database Connection Issues**:
- Ensure PostgreSQL is running and accessible at the configured URL
- Verify pgvector extension is installed: `CREATE EXTENSION IF NOT EXISTS vector;`
- Check firewall settings if using remote database

**Index Setup Problems**:
- Verify source paths exist and are readable
- Check file patterns in configuration - too restrictive patterns may skip files
- Ensure sufficient disk space for embeddings storage

**MCP Server Issues**:
- For stdio transport: ensure no extra output/logging to stdout/stderr
- For HTTP transport: check port availability (default 5791)
- Verify MCP client configuration matches server transport type

**Performance Issues**:
- Large codebases: consider chunking strategy and embedding batch size
- Slow searches: check database indexes and query optimization
- Memory usage: monitor embedding model size and batch processing

## Contributing

Contributions are welcome! Here's how you can help:

### Reporting Issues

Found a bug or have a feature request? Please open an issue on GitHub with:
- Clear description of the problem or suggestion
- Steps to reproduce (for bugs)
- Expected vs. actual behavior
- Your environment (Python version, OS, PostgreSQL version)

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following the code quality guidelines
4. Run the quality checks: `task fct` (format, check, test)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

### Code Quality Guidelines

- Follow the existing code style
- Maintain max cyclomatic complexity of 10
- Add tests for new features
- Update documentation as needed
- Ensure all tests pass (`task test`)
- Run code formatting and checks (`task fct`)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Credits

CocoRAG is built on top of these projects:

- **[CocoIndex](https://cocoindex.io)** ([GitHub](https://github.com/cocoindex-io/cocoindex)) - Real-time data transformation and indexing framework providing the incremental processing pipeline, chunking, and embedding infrastructure.
- **[pgvector](https://github.com/pgvector/pgvector)** - Open-source vector similarity search extension for PostgreSQL.
- **[FastMCP](https://github.com/jlowin/fastmcp)** - Fast, Pythonic framework for building Model Context Protocol servers.
- **[spaCy](https://spacy.io)** - Industrial-strength NLP library used for the optional linguistic reranking engine.

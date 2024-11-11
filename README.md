
# Multi-Agent-RAG-Template

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-orange)](https://swarms.xyz)

A production-ready template for building Multi-Agent RAG (Retrieval-Augmented Generation) systems using the Swarms Framework. This template demonstrates how to create a collaborative team of AI agents that work together to process, analyze, and generate insights from documents.


## 🌟 Features

- **Plug-and-Play Agent Architecture**
  - Easily swap or modify agents without disrupting the system
  - Add custom agents with specialized capabilities
  - Define your own agent interaction patterns
  - Scale from 2 to 100+ agents seamlessly
  - Any LLM can be used, this template uses GROQ but you can use any other LLM such as OpenAI, Anthropic, Cohere, etc.

- **Adaptable Document Processing**
  - Support for any document format through custom extractors
  - Flexible document storage options (local, cloud, or hybrid)
  - Customizable chunking and embedding strategies
  - Dynamic index updates without system restart
  - Any RAG system can be used, this template uses LlamaIndexDB but you can use any other RAG system.

- **Configurable Workflows**
  - Design custom agent communication patterns
  - Implement parallel or sequential processing
  - Add conditional logic and branching workflows
  - Adjust system behavior through environment variables


## 🚀 Quick Start

1. **Clone the Repository**
```bash
git clone https://github.com/The-Swarm-Corporation/Multi-Agent-RAG-Template.git
cd Multi-Agent-RAG-Template
```

2. **Set Up Environment**
```bash
# Create and activate virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure Environment Variables**
```bash
# Create .env file

# Edit .env file with your credentials
GROQ_API_KEY="your-api-key-here"
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY="your-openai-api-key-here"
```

4. **Run the Example**
```bash
python main.py
```

## 🏗️ Project Structure

```
Multi-Agent-RAG-Template/
├── main.py                    # Main entry point
├── multi_agent_rag/
│   ├── agents.py             # Agent definitions
│   └── memory.py             # RAG implementation
├── docs/                      # Place your documents here
├── requirements.txt           # Project dependencies
└── .env                      # Environment variables
```

## 🔧 Customization

### Adding New Agents

1. Open `multi_agent_rag/agents.py`
2. Create a new agent using the Agent class:

```python
new_agent = Agent(
    agent_name="New-Agent",
    system_prompt="Your system prompt here",
    llm=model,
    max_loops=1,
    # ... additional configuration
)
```

### Modifying the Agent Flow

In `main.py`, update the `flow` parameter in the `AgentRearrange` initialization:

```python
flow=f"{agent1.agent_name} -> {agent2.agent_name} -> {new_agent.agent_name}"
```

## Integrating RAG

- The `memory_system` parameter in the `AgentRearrange` initialization is used to configure the RAG system.
- The `memory_system` parameter is an instance of `LlamaIndexDB`, which is a database class for storing and retrieving medical documents.
- The `memory_system` class must have a `query(query: str)` method that returns a string for the agent to use it.

```python

# Import the AgentRearrange class for coordinating multiple agents
from swarms import AgentRearrange

# Import specialized medical agents for different aspects of patient care
from multi_agent_rag.agents import (
    diagnostic_specialist,      # Agent for diagnostic analysis
    medical_data_extractor,    # Agent for extracting medical data
    patient_care_coordinator,  # Agent for coordinating patient care
    specialist_consultant,     # Agent for specialist consultation
    treatment_planner,        # Agent for treatment planning
)

# Import database class for storing and retrieving medical documents
from multi_agent_rag.memory import LlamaIndexDB

# Initialize the SwarmRouter to coordinate the medical agents
router = AgentRearrange(
    name="medical-diagnosis-treatment-swarm",
    description="Collaborative medical team for comprehensive patient diagnosis and treatment planning",
    max_loops=1,  # Limit to one iteration through the agent flow
    agents=[
        medical_data_extractor,    # First agent to extract medical data
        diagnostic_specialist,      # Second agent to analyze and diagnose
        treatment_planner,         # Third agent to plan treatment
        specialist_consultant,      # Fourth agent to provide specialist input
        patient_care_coordinator,  # Final agent to coordinate care plan
    ],
    # Configure the document storage and retrieval system
    memory_system=LlamaIndexDB(
        data_dir="docs",           # Directory containing medical documents
        filename_as_id=True,       # Use filenames as document identifiers
        recursive=True,            # Search subdirectories
        # required_exts=[".txt", ".pdf", ".docx"],  # Supported file types
        similarity_top_k=10,       # Return top 10 most relevant documents
    ),
    # Define the sequential flow of information between agents
    flow=f"{medical_data_extractor.agent_name} -> {diagnostic_specialist.agent_name} -> {treatment_planner.agent_name} -> {specialist_consultant.agent_name} -> {patient_care_coordinator.agent_name}",
)

# Example usage
if __name__ == "__main__":
    # Run a comprehensive medical analysis task for patient Lucas Brown
    router.run(
        "Analyze this Lucas Brown's medical data to provide a diagnosis and treatment plan"
    )

```



## 📚 Documentation

For detailed documentation on:
- [Swarms Framework](https://swarms.xyz)
- [LlamaIndex](https://docs.llamaindex.ai)
- [GROQ](https://groq.com)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
g

## 🛠 Built With

- [Swarms Framework](https://github.com/kyegomez/swarms)
- Python 3.10+
- GROQ API Key or you can change it to use any model from [Swarm Models](https://github.com/The-Swarm-Corporation/swarm-models)
- LlamaIndexDB for storing and retrieving medical documents

## 📬 Contact

Questions? Reach out:
- Twitter: [@kyegomez](https://twitter.com/kyegomez)
- Email: kye@swarms.world

---

## Want Real-Time Assistance?

[Book a call with here for real-time assistance:](https://cal.com/swarms/swarms-onboarding-session)

---

⭐ Star us on GitHub if this project helped you!

Built with ♥ using [Swarms Framework](https://github.com/kyegomez/swarms)








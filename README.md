
# Multi-Agent-RAG-Template

[![Join our Discord](https://img.shields.io/badge/Discord-Join%20our%20server-5865F2?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/agora-999382051935506503) [![Subscribe on YouTube](https://img.shields.io/badge/YouTube-Subscribe-red?style=for-the-badge&logo=youtube&logoColor=white)](https://www.youtube.com/@kyegomez3242) [![Connect on LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/kye-g-38759a207/) [![Follow on X.com](https://img.shields.io/badge/X.com-Follow-1DA1F2?style=for-the-badge&logo=x&logoColor=white)](https://x.com/kyegomezb)


[![GitHub stars](https://img.shields.io/github/stars/The-Swarm-Corporation/Legal-Swarm-Template?style=social)](https://github.com/The-Swarm-Corporation/Legal-Swarm-Template)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-blue)](https://github.com/kyegomez/swarms)


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Swarms Framework](https://img.shields.io/badge/Built%20with-Swarms-orange)](https://swarms.xyz)

A production-ready template for building Multi-Agent RAG (Retrieval-Augmented Generation) systems using the Swarms Framework. This template demonstrates how to create a collaborative team of AI agents that work together to process, analyze, and generate insights from documents.


## 🌟 Features

- **Pre-configured Agent System**: Includes specialized agents for different tasks
- **Document Processing**: Built-in RAG capabilities using LlamaIndex
- **Flexible Architecture**: Easy to customize and extend for different use cases
- **Production Ready**: Includes error handling, logging, and state management
- **GROQ Integration**: Optimized for high-performance LLM inference




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
# Copy example env file
cp .env.example .env

# Edit .env file with your credentials
GROQ_API_KEY="your-api-key-here"
WORKSPACE_DIR="agent_workspace"
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









import os
WORKSPACE_DIR = "C:/Users/Patrick Devaney/Desktop/Multi-Agent-RAG-Template/multi_agent_rag"
os.environ["WORKSPACE_DIR"] = WORKSPACE_DIR
# Import the AgentRearrange class for coordinating multiple agents
from swarms import AgentRearrange

# Import specialized medical agents for different aspects of patient care
from multi_agent_rag.agents import (
    diagnostic_specialist,  # Agent for diagnostic analysis
    medical_data_extractor,  # Agent for extracting medical data
    patient_care_coordinator,  # Agent for coordinating patient care
    specialist_consultant,  # Agent for specialist consultation
    treatment_planner,  # Agent for treatment planning
)

# Import database class for storing and retrieving medical documents
from multi_agent_rag.memory import LlamaIndexDB

# Initialize the SwarmRouter to coordinate the medical agents
router = AgentRearrange(
    name="medical-diagnosis-treatment-swarm",
    description="Collaborative medical team for comprehensive patient diagnosis and treatment planning",
    max_loops=1,  # Limit to one iteration through the agent flow
    agents=[
        medical_data_extractor,  # First agent to extract medical data
        diagnostic_specialist,  # Second agent to analyze and diagnose
        treatment_planner,  # Third agent to plan treatment
        specialist_consultant,  # Fourth agent to provide specialist input
        patient_care_coordinator,  # Final agent to coordinate care plan
    ],
    # Configure the document storage and retrieval system
    memory_system=LlamaIndexDB(
        data_dir="docs",  # Directory containing medical documents
        filename_as_id=True,  # Use filenames as document identifiers
        recursive=True,  # Search subdirectories
        similarity_top_k=10,  # Return top 10 most relevant documents
    ),
    # Define the sequential flow of information between agents
    flow=f"{medical_data_extractor.agent_name} -> {diagnostic_specialist.agent_name} -> {treatment_planner.agent_name} -> {specialist_consultant.agent_name} -> {patient_care_coordinator.agent_name}",
)

# Example usage
if __name__ == "__main__":
    # Run a comprehensive medical analysis task for patient Lucas Brown
    patient_data = "Patient Lucas Brown's medical data goes here."  # Example input
    router.run(patient_data)  # Pass the patient data through the agent flow

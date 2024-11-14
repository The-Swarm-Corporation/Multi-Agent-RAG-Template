import os
from swarms import AgentRearrange
from multi_agent_rag.agents import (
    diagnostic_specialist,
    medical_data_extractor,
    patient_care_coordinator,
    specialist_consultant,
    treatment_planner,
)
from multi_agent_rag.pinecone_wrapper import PineconeManager

router = AgentRearrange(
    name="medical-diagnosis-treatment-swarm",
    description="Collaborative medical team for comprehensive patient diagnosis and treatment planning",
    max_loops=1,
    agents=[
        medical_data_extractor,
        diagnostic_specialist,
        treatment_planner,
        specialist_consultant,
        patient_care_coordinator,
    ],
    memory_system=PineconeManager(
        api_key=os.getenv("PINECONE_API_KEY"),
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        environment=os.getenv("PINECONE_ENVIRONMENT"),
    ),
    flow=f"{medical_data_extractor.agent_name} -> {diagnostic_specialist.agent_name} -> {treatment_planner.agent_name} -> {specialist_consultant.agent_name} -> {patient_care_coordinator.agent_name}",
)

if __name__ == "__main__":
    router.run(
        "Analyze this Lucas Brown's medical data to provide a diagnosis and treatment plan"
    )

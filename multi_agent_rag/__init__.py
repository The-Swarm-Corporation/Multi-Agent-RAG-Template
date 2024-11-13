#initialized
from .agents import (
    medical_data_extractor,
    diagnostic_specialist,
    treatment_planner,
    specialist_consultant,
    patient_care_coordinator
)
from .swarm_models import OpenAIChat

# Initialize the model or other common components
from .common import initialize_model  # If you have a separate init function for the model

# Additional setup or logging if needed
import logging
logging.basicConfig(level=logging.INFO)

import os
import pinecone
from faker import Faker
import random
from datetime import datetime, timedelta
from llama_index.embeddings import get_embedding
from llama_index import SimpleVectorStore

fake = Faker()

# Medical-specific data
conditions = [
    "Hypertension", "Type 2 Diabetes", "Asthma", "Arthritis", "Depression",
    "Anxiety", "GERD", "Migraine", "Hypothyroidism", "Hyperlipidemia",
]

medications = [
    "Lisinopril 10mg", "Metformin 500mg", "Albuterol inhaler", "Sertraline 50mg",
    "Omeprazole 20mg", "Levothyroxine 75mcg", "Atorvastatin 40mg", "Amlodipine 5mg",
    "Metoprolol 25mg", "Gabapentin 300mg",
]

allergies = [
    "Penicillin", "Sulfa", "Latex", "Peanuts", "Shellfish", "Iodine", "Aspirin", 
    "Morphine", "None", "Dairy",
]

# Initialize Pinecone
pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))
pinecone_index = pinecone.Index("medical")  # for our medical example, update with your actual index name

def generate_vitals():
    return {
        "Blood Pressure": f"{random.randint(110,140)}/{random.randint(60,90)}",
        "Heart Rate": f"{random.randint(60,100)} bpm",
        "Temperature": f"{round(random.uniform(97.0, 99.5), 1)}°F",
        "Respiratory Rate": f"{random.randint(12,20)} breaths/min",
        "Weight": f"{random.randint(120,220)} lbs",
        "Height": f"{random.randint(60,75)} inches",
    }

def generate_patient_history():
    num_visits = random.randint(3, 8)
    history = []
    current_date = datetime.now()

    for _ in range(num_visits):
        visit = {
            "Date": current_date.strftime("%Y-%m-%d"),
            "Chief Complaint": fake.sentence(nb_words=4),
            "Vitals": generate_vitals(),
            "Assessment": random.sample(conditions, random.randint(1, 3)),
            "Medications": random.sample(medications, random.randint(1, 4)),
            "Plan": fake.paragraph(nb_sentences=2),
        }
        history.append(visit)
        current_date -= timedelta(days=random.randint(30, 180))

    return history

def send_to_pinecone(patient_id, document_text):
    """Embed and index patient document to Pinecone."""
    embedding = get_embedding(document_text)  # Generate embedding for the document text
    # Upsert the embedding to the Pinecone index
    pinecone_index.upsert([(patient_id, embedding)])  # Ensure this format matches Pinecone's expected input

def create_patient_file(patient_id):
    """Create a patient record and send it to Pinecone."""
    patient = {
        "Patient ID": patient_id,
        "Name": fake.name(),
        "DOB": fake.date_of_birth(minimum_age=18, maximum_age=90).strftime("%Y-%m-%d"),
        "Gender": random.choice(["Male", "Female"]),
        "Contact": {
            "Phone": fake.phone_number(),
            "Email": fake.email(),
            "Address": fake.address(),
        },
        "Insurance": fake.company(),
        "Allergies": random.sample(allergies, random.randint(0, 3)),
        "Medical History": generate_patient_history(),
    }

    # Create directory if it doesn't exist
    if not os.path.exists("docs"):
        os.makedirs("docs")

    # Write patient data to file
    filename = f"docs/patient_{patient_id}.txt"
    with open(filename, "w") as f:
        f.write(f"PATIENT MEDICAL RECORD\n{'='*50}\n\n")
        f.write(f"Patient ID: {patient['Patient ID']}\n")
        f.write(f"Name: {patient['Name']}\n")
        f.write(f"DOB: {patient['DOB']}\n")
        f.write(f"Gender: {patient['Gender']}\n\n")

        f.write("Contact Information:\n")
        for key, value in patient["Contact"].items():
            f.write(f"{key}: {value}\n")

        f.write(f"\nInsurance: {patient['Insurance']}\n")
        f.write(
            f"Allergies: {', '.join(patient['Allergies']) if patient['Allergies'] else 'None'}\n\n"
        )

        f.write("MEDICAL HISTORY\n")
        f.write("-" * 50 + "\n")

        for visit in patient["Medical History"]:
            f.write(f"\nVisit Date: {visit['Date']}\n")
            f.write(f"Chief Complaint: {visit['Chief Complaint']}\n")

            f.write("\nVitals:\n")
            for k, v in visit["Vitals"].items():
                f.write(f"  {k}: {v}\n")

            f.write(f"\nAssessment: {', '.join(visit['Assessment'])}\n")
            f.write(f"Medications: {', '.join(visit['Medications'])}\n")
            f.write(f"Plan: {visit['Plan']}\n")
            f.write("-" * 50 + "\n")

    # Send patient file to Pinecone after creation
    with open(filename, "r") as f:
        document_text = f.read()
    send_to_pinecone(patient_id, document_text)

def main():
    num_patients = 10  # Change this number to generate more or fewer patient records
    for i in range(num_patients):
        create_patient_file(f"P{str(i+1).zfill(6)}")
    print(f"Generated {num_patients} patient records in the 'docs' directory and indexed them in Pinecone.")

if __name__ == "__main__":
    main()

    # Cleanup Pinecone at the end
    pinecone.deinit()

# import re
# import os
# import json
# from openai import OpenAI
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field
# from typing import List
# import io
# import requests
# from pdf2image import convert_from_bytes
# import pytesseract
# import docx2txt
# import boto3
# from botocore.exceptions import NoCredentialsError, PartialCredentialsError
# from concurrent.futures import ThreadPoolExecutor, as_completed
# from typing import Optional

# load_dotenv()

# client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
# AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

# class CandidateSummary(BaseModel):
#     candidate_id: str = Field(..., description="The unique identifier for the candidate.")
#     score: float = Field(..., description="The similarity score of the candidate.")
#     summary: str = Field(..., description="A brief summary of the candidate's profile.")

# class GenerativeOutput(BaseModel):
#     query: str = Field(..., description="The user's query.")
#     candidates: List[CandidateSummary] = Field(..., description="List of relevant candidates.")

# class JobDescription(BaseModel):
#     role: str = Field(None, description="The job role/title.")
#     experience: str = Field(None, description="Required experience level.")
#     location: str = Field(None, description="Job location.")
#     job_description: str = Field(None, description="Detailed job description.")
#     key_responsibilities: List[str] = Field([], description="List of key responsibilities.")
#     qualifications: List[str] = Field([], description="List of required qualifications.")
#     skills: List[str] = Field([], description="List of required skills.")

# class PineconeLoader:
#     def __init__(self, aggregated_json_path, index_name="cv-automation", embedding_model='text-embedding-ada-002'):
#         self.aggregated_json_path = aggregated_json_path
#         self.index_name = index_name
#         self.embedding_model = embedding_model
#         self.s3_client = self.authenticate_to_s3()
        
#         index_names = [index['name'] for index in pc.list_indexes()]
#         if self.index_name not in index_names:
#             pc.create_index(
#                 name=self.index_name,
#                 dimension=1536,  
#                 metric="cosine",
#                 spec=ServerlessSpec(cloud="aws", region="us-east-1")
#             )
#             print(f"Successfully created the new index with name {self.index_name}")

#     def authenticate_to_s3(self):
#         """Authenticate to AWS S3."""
#         try:
#             s3_client = boto3.client(
#                 's3',
#                 aws_access_key_id=AWS_ACCESS_KEY_ID,
#                 aws_secret_access_key=AWS_SECRET_ACCESS_KEY
#             )
#             print("✅ Authenticated to S3.")
#             return s3_client
#         except (NoCredentialsError, PartialCredentialsError) as e:
#             print(f"❌ Authentication failed: {e}")
#             raise

#     def list_files_in_bucket(self, bucket_name, prefix=""):
#         """List files in an S3 bucket."""
#         try:
#             response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
#             return response.get('Contents', []) if 'Contents' in response else []
#         except Exception as e:
#             print(f"❌ Error listing files: {e}")
#             return []

#     def download_file_as_bytes(self, bucket_name, file_key):
#         """Download a file from S3 as bytes."""
#         try:
#             file_stream = io.BytesIO()
#             self.s3_client.download_fileobj(bucket_name, file_key, file_stream)
#             file_stream.seek(0)
#             return file_stream
#         except Exception as e:
#             print(f"❌ Error downloading file: {e}")
#             return None

#     def extract_text_from_pdf(self, pdf_bytes):
#         """Extract text from a PDF file."""
#         try:
#             images = convert_from_bytes(pdf_bytes.read())
#             extracted_text = " ".join([pytesseract.image_to_string(image) for image in images if image])
#             return extracted_text.strip()
#         except Exception as e:
#             print(f"❌ Error extracting text from PDF: {e}")
#             return ""

#     def extract_text_from_word(self, doc_bytes):
#         """Extract text from a Word document."""
#         try:
#             text = docx2txt.process(io.BytesIO(doc_bytes.read()))
#             return text.strip()
#         except Exception as e:
#             print(f"❌ Error extracting text from Word: {e}")
#             return ""

#     def clean_json_response(self, response_text):
#         """Clean OpenAI JSON response."""
#         match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
#         return match.group(1) if match else response_text.strip()

#     def process_jd_text_with_openai(self, extracted_text):
#         """Process extracted JD text with OpenAI."""
#         headers = {
#             "Content-Type": "application/json",
#             "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
#         }

#         schema = JobDescription.schema()
#         payload = {
#             "model": "gpt-4o-mini",
#             "messages": [
#                 {
#                     "role": "system",
#                     "content": (
#                         f"Extract job details from the given text based on this JSON schema: {json.dumps(schema, indent=2)}.\n"
#                         "Ensure the response avoids Unicode characters (e.g., use '-' instead of '—').\n"
#                         "Additionally, infer relevant skills based on the qualifications provided."
#                     )
#                 },
#                 {"role": "user", "content": extracted_text}
#             ],
#             "max_tokens": 2000
#         }

#         try:
#             response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
#             if response.status_code != 200:
#                 print(f"❌ Error: {response.status_code} - {response.text}")
#                 return None
            
#             response_json = response.json()
#             if "choices" not in response_json or not response_json["choices"]:
#                 print("❌ OpenAI API returned an empty response.")
#                 return None
            
#             content = response_json["choices"][0]["message"]["content"].strip()
#             cleaned_json = self.clean_json_response(content)
            
#             return json.loads(cleaned_json)

#         except json.JSONDecodeError:
#             print("❌ Error: Unable to parse JSON response.")
#             print("Raw Response:", content)
#             return None
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Network error: {e}")
#             return None

#     def process_jd_file(self, bucket_name, file):
#         """Process a single JD file."""
#         file_key = file['Key']
#         print(f"🔍 Processing JD: {file_key}")
#         file_stream = self.download_file_as_bytes(bucket_name, file_key)
#         if not file_stream:
#             return None

#         if file_key.endswith('.pdf'):
#             extracted_text = self.extract_text_from_pdf(file_stream)
#         elif file_key.endswith('.docx') or file_key.endswith('.doc'):
#             extracted_text = self.extract_text_from_word(file_stream)
#         else:
#             return None

#         if not extracted_text:
#             print(f"⏩ Skipping {file_key} due to empty extracted text.")
#             return None

#         jd_data = self.process_jd_text_with_openai(extracted_text)
#         return jd_data

#     def get_jd_data(self, bucket_name="datacrux-dev", prefix="hem/"):
#         """Get all JD data from S3 bucket."""
#         files = self.list_files_in_bucket(bucket_name, prefix)
#         jd_data = []

#         with ThreadPoolExecutor(max_workers=5) as executor:
#             future_to_file = {executor.submit(self.process_jd_file, bucket_name, file): file for file in files}
            
#             for future in as_completed(future_to_file):
#                 jd = future.result()
#                 if jd:
#                     jd_data.append(jd)

#         return jd_data

#     def format_jd_for_query(self, jd_data):
#         """Format JD data for querying."""
#         if not jd_data:
#             return ""
        
#         formatted_jd = ""
#         for jd in jd_data:
#             formatted_jd += f"Job Role: {jd.get('role', 'N/A')}\n"
#             formatted_jd += f"Experience Required: {jd.get('experience', 'N/A')}\n"
#             formatted_jd += f"Location: {jd.get('location', 'N/A')}\n"
#             formatted_jd += f"Description: {jd.get('job_description', 'N/A')}\n"
#             formatted_jd += "Key Responsibilities:\n- " + "\n- ".join(jd.get('key_responsibilities', [])) + "\n"
#             formatted_jd += "Qualifications:\n- " + "\n- ".join(jd.get('qualifications', [])) + "\n"
#             formatted_jd += "Skills:\n- " + "\n- ".join(jd.get('skills', [])) + "\n\n"
        
#         return formatted_jd.strip()

#     def load_and_index(self):
#         """Load JSON data and index it into Pinecone."""
#         aggregated_data = self.load_json(self.aggregated_json_path)
#         if aggregated_data:
#             self.process_candidates(aggregated_data)

#     def load_json(self, file_path):
#         """Load JSON data from a file."""
#         try:
#             with open(file_path, 'r') as file:
#                 return json.load(file)
#         except Exception as e:
#             print(f"Error loading JSON file {file_path}: {e}")
#             return None

#     def process_candidates(self, json_data):
#         """Process all candidates in the JSON data."""
#         try:
#             candidates = json_data.get("candidates", [])
#             if not candidates:
#                 print("No candidates found in the JSON data.")
#                 return
            
#             for candidate in candidates:
#                 if not isinstance(candidate, dict):
#                     print(f"Skipping invalid candidate: {candidate}")
#                     continue
#                 self.process_candidate_data(candidate)
#         except Exception as e:
#             print(f"Error processing candidates: {e}")

#     # def process_candidate_data(self, candidate):
#     #     """Process a single candidate's data."""
#     #     try:
#     #         candidate_id = candidate.get("name", "null")
#     #         if candidate_id == "null":
#     #             candidate_id = candidate.get("email", f"candidate_{hash(json.dumps(candidate))}")
            
#     #         print(f"Processing candidate with ID: {candidate_id}")  
#     #         combined_text = self.combine_all_sections(candidate)
#     #         self.upsert_candidate(candidate_id, combined_text)
#     #     except Exception as e:
#     #         print(f"Error processing candidate {candidate_id}: {e}")

#     def combine_all_sections(self, candidate):
#         """Combine and weight relevant sections of a candidate's data for embeddings."""
#         sections = {
#             "skills": candidate.get("skills", []),
#             "projects": candidate.get("projects", []),
#             "experience": candidate.get("experience", []),
#             "certifications": candidate.get("certifications", []),
#             "personal_info": {  
#                 "name": candidate.get("name", ""),
#                 "email": candidate.get("email", ""),
#                 "phone": candidate.get("phone", "")
#             }
#         }
        
#         weights = {"skills": 4, "projects": 3, "experience": 2, "certifications": 1, "personal_info": 1}
#         combined_text = ""
        
#         for section_name, section_content in sections.items():
#             combined_text += (f"{section_name}: {self.json_to_text(section_content)}\n" * weights[section_name])
        
#         return combined_text.strip()

#     def json_to_text(self, json_data):
#         """Convert JSON data to a plain text string."""
#         if isinstance(json_data, dict):
#             return ' '.join([f"{key}: {value}" for key, value in json_data.items() if value])
#         elif isinstance(json_data, list):
#             return ' '.join([self.json_to_text(item) for item in json_data if item])
#         else:
#             return str(json_data)

#     def generate_embedding(self, text):
#         """Generate embeddings for the given text using OpenAI."""
#         response = client.embeddings.create(input=text, model=self.embedding_model)
#         return response.data[0].embedding

#     def upsert_candidate(self, candidate_id, combined_text):
#         """Upsert a candidate's combined data into Pinecone."""
#         try:
#             index = pc.Index(self.index_name)
#             embedding = self.generate_embedding(combined_text)
#             metadata = {'candidate_id': candidate_id, 'content': combined_text}
#             index.upsert(vectors=[(candidate_id, embedding, metadata)])
#             print(f"Upserted candidate {candidate_id} into Pinecone.")
#         except Exception as e:
#             print(f"Error upserting candidate {candidate_id}: {e}")

#     def generate_candidate_summary(self, candidate_content):
#         """Generate a natural language summary of the candidate using OpenAI."""
#         try:
#             prompt = f"""
#             Generate a concise and professional summary of the following candidate profile. 
#             Include the candidate's contact information (email, phone, linkedin and github) at the end of the summary.
#             Also, highlight the candidate's key skills and how they align with the job requirements.

#             Candidate Profile:
#             {candidate_content}

#             The summary should highlight key skills, experience, and qualifications in a natural and readable format.
#             At the end, include the following contact details:
#             - Email: [Candidate's Email]
#             - Phone: [Candidate's Phone]
#             - LinkedIn: [Candidate's LinkedIn] 
#             - GitHub: [Candidate's GitHub] 
#             """
            
#             response = client.chat.completions.create(
#                 model="gpt-4o-mini", 
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant that generates professional candidate summaries."},
#                     {"role": "user", "content": prompt}
#                 ]
#             )
            
#             summary = response.choices[0].message.content.strip()
#             return summary
#         except Exception as e:
#             print(f"Error generating candidate summary: {e}")
#             return "No summary available."

#     def generate_markdown_report(self, candidates):
#         """Generate a markdown report for the selected candidates."""
#         markdown = "# Selected Candidates Report\n\n"
#         for candidate in candidates:
#             markdown += f"## Candidate: {candidate['candidate_id']}\n"
#             markdown += f"{self.generate_candidate_summary(candidate['content'])}\n\n"
#         return markdown

#     def save_markdown_report(self, markdown_content, file_path="selected_candidates_report.md"):
#         """Save the markdown report to a local file."""
#         try:
#             with open(file_path, 'w') as file:
#                 file.write(markdown_content)
#             print(f"Markdown report saved to {file_path}")
#         except Exception as e:
#             print(f"Error saving markdown report: {e}")

#     def query(self, prompt=None, jd_data=None, top_k=5, confidence_threshold=0.7):
#         """Query Pinecone for relevant candidates based on user query and/or JD data."""
#         try:
#             index = pc.Index(self.index_name)
            
#             jd_text = self.format_jd_for_query([jd_data]) if jd_data else ""
            
#             combined_query = ""
#             if prompt:
#                 combined_query += f"User Query: {prompt}\n\n"
#             if jd_text:
#                 combined_query += f"Job Description Details:\n{jd_text}"
            
#             if not combined_query.strip():
#                 print("❌ No query or JD provided for search.")
#                 return []
            
#             embedding = self.generate_embedding(combined_query)
#             query_response = index.query(
#                 vector=embedding,
#                 top_k=top_k,
#                 include_metadata=True
#             )
            
#             candidates = []
#             for match in query_response['matches']:
#                 score = match['score']
#                 if score >= confidence_threshold:
#                     candidate_data = {
#                         'candidate_id': match['id'],
#                         'score': round(score, 2),
#                         'content': match['metadata'].get('content', 'No details available')
#                     }
#                     candidates.append(candidate_data)

#             markdown_report = self.generate_markdown_report(candidates)
#             self.save_markdown_report(markdown_report)
#             print(markdown_report)

#             return candidates

#         except Exception as e:
#             print(f"Error querying Pinecone: {e}")
#             return []

# def process_input(input_str: str, loader: PineconeLoader) -> Optional[dict]:
#     """
#     Process input string - could be:
#     1. Local file path (upload to S3 then process)
#     2. Text query (use directly)
#     """
#     try:
#         input_str = input_str.strip().strip('"\'')  
        
#         if os.path.exists(input_str):
#             print(f"🔍 Processing local file: {input_str}")
            
#             with open(input_str, 'rb') as f:
#                 file_content = f.read()
            
#             file_name = os.path.basename(input_str)
#             s3_key = f"hem/{file_name}"
#             loader.s3_client.put_object(
#                 Bucket="datacrux-dev",
#                 Key=s3_key,
#                 Body=file_content
#             )
#             print(f"✅ Uploaded to s3://datacrux-dev/{s3_key}")
            
#             jd_data = loader.process_jd_file("datacrux-dev", {"Key": s3_key})
#             if jd_data:
#                 return {"jd_data": jd_data, "source": f"s3://datacrux-dev/{s3_key}"}
#             return None

#         else:
#             print("🔍 Treating input as direct query")
#             return {"query": input_str}

#     except Exception as e:
#         print(f"❌ Error processing input: {e}")
#         return None

# if __name__ == "__main__":
#     print("Script is running...")
    
#     loader = PineconeLoader(aggregated_json_path="aggregated_data.json")
#     loader.load_and_index()
    
#     user_input = input("\n🔍 Enter file path or search query: ").strip()
    
#     if not user_input:
#         print("👋 No input provided - exiting.")
#     else:
#         result = process_input(user_input, loader)
        
#         if not result:
#             print("⚠️ No valid input processed")
#         elif "jd_data" in result:
#             print(f"\n🔎 Searching with JD from {result['source']}")
#             loader.query(jd_data=result["jd_data"])
#         elif "query" in result:
#             print(f"\n🔎 Searching for: {result['query']}")
#             loader.query(prompt=result['query'])
    
#     print("👋 Script completed.")


# ###########################################################################################

import re
import os
import json
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import List
import io
import requests
from pdf2image import convert_from_bytes
import pytesseract
import docx2txt
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

class CandidateSummary(BaseModel):
    candidate_id: str = Field(..., description="The unique identifier for the candidate.")
    score: float = Field(..., description="The similarity score of the candidate.")
    summary: str = Field(..., description="A brief summary of the candidate's profile.")

class GenerativeOutput(BaseModel):
    query: str = Field(..., description="The user's query.")
    candidates: List[CandidateSummary] = Field(..., description="List of relevant candidates.")

class JobDescription(BaseModel):
    role: str = Field(None, description="The job role/title.")
    experience: str = Field(None, description="Required experience level.")
    location: str = Field(None, description="Job location.")
    job_description: str = Field(None, description="Detailed job description.")
    key_responsibilities: List[str] = Field([], description="List of key responsibilities.")
    qualifications: List[str] = Field([], description="List of required qualifications.")
    skills: List[str] = Field([], description="List of required skills.")

class PineconeLoader:
    def __init__(self, aggregated_json_path, index_name="cv-automation", embedding_model='text-embedding-ada-002'):
        self.aggregated_json_path = aggregated_json_path
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.s3_client = self.authenticate_to_s3()
        
        index_names = [index['name'] for index in pc.list_indexes()]
        if self.index_name not in index_names:
            pc.create_index(
                name=self.index_name,
                dimension=1536,  
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Successfully created the new index with name {self.index_name}")

    def authenticate_to_s3(self):
        """Authenticate to AWS S3."""
        try:
            s3_client = boto3.client(
                's3',
                aws_access_key_id=AWS_ACCESS_KEY_ID,
                aws_secret_access_key=AWS_SECRET_ACCESS_KEY
            )
            print("✅ Authenticated to S3.")
            return s3_client
        except (NoCredentialsError, PartialCredentialsError) as e:
            print(f"❌ Authentication failed: {e}")
            raise

    def list_files_in_bucket(self, bucket_name, prefix=""):
        """List files in an S3 bucket."""
        try:
            response = self.s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
            return response.get('Contents', []) if 'Contents' in response else []
        except Exception as e:
            print(f"❌ Error listing files: {e}")
            return []

    def download_file_as_bytes(self, bucket_name, file_key):
        """Download a file from S3 as bytes."""
        try:
            file_stream = io.BytesIO()
            self.s3_client.download_fileobj(bucket_name, file_key, file_stream)
            file_stream.seek(0)
            return file_stream
        except Exception as e:
            print(f"❌ Error downloading file: {e}")
            return None

    def extract_text_from_pdf(self, pdf_bytes):
        """Extract text from a PDF file."""
        try:
            images = convert_from_bytes(pdf_bytes.read())
            extracted_text = " ".join([pytesseract.image_to_string(image) for image in images if image])
            return extracted_text.strip()
        except Exception as e:
            print(f"❌ Error extracting text from PDF: {e}")
            return ""

    def extract_text_from_word(self, doc_bytes):
        """Extract text from a Word document."""
        try:
            text = docx2txt.process(io.BytesIO(doc_bytes.read()))
            return text.strip()
        except Exception as e:
            print(f"❌ Error extracting text from Word: {e}")
            return ""

    def clean_json_response(self, response_text):
        """Clean OpenAI JSON response."""
        match = re.search(r"```json\n(.*?)\n```", response_text, re.DOTALL)
        return match.group(1) if match else response_text.strip()

    def process_jd_text_with_openai(self, extracted_text):
        """Process extracted JD text with OpenAI."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
        }

        schema = JobDescription.schema()
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {
                    "role": "system",
                    "content": (
                        f"Extract job details from the given text based on this JSON schema: {json.dumps(schema, indent=2)}.\n"
                        "Ensure the response avoids Unicode characters (e.g., use '-' instead of '—').\n"
                        "Additionally, infer relevant skills based on the qualifications provided."
                    )
                },
                {"role": "user", "content": extracted_text}
            ],
            "max_tokens": 2000
        }

        try:
            response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
            
            if response.status_code != 200:
                print(f"❌ Error: {response.status_code} - {response.text}")
                return None
            
            response_json = response.json()
            if "choices" not in response_json or not response_json["choices"]:
                print("❌ OpenAI API returned an empty response.")
                return None
            
            content = response_json["choices"][0]["message"]["content"].strip()
            cleaned_json = self.clean_json_response(content)
            
            return json.loads(cleaned_json)

        except json.JSONDecodeError:
            print("❌ Error: Unable to parse JSON response.")
            print("Raw Response:", content)
            return None
        except requests.exceptions.RequestException as e:
            print(f"❌ Network error: {e}")
            return None

    def process_jd_file(self, bucket_name, file):
        """Process a single JD file."""
        file_key = file['Key']
        print(f"🔍 Processing JD: {file_key}")
        file_stream = self.download_file_as_bytes(bucket_name, file_key)
        if not file_stream:
            return None

        if file_key.endswith('.pdf'):
            extracted_text = self.extract_text_from_pdf(file_stream)
        elif file_key.endswith('.docx') or file_key.endswith('.doc'):
            extracted_text = self.extract_text_from_word(file_stream)
        else:
            return None

        if not extracted_text:
            print(f"⏩ Skipping {file_key} due to empty extracted text.")
            return None

        jd_data = self.process_jd_text_with_openai(extracted_text)
        return jd_data

    def get_jd_data(self, bucket_name="datacrux-dev", prefix="hem/"):
        """Get all JD data from S3 bucket."""
        files = self.list_files_in_bucket(bucket_name, prefix)
        jd_data = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_file = {executor.submit(self.process_jd_file, bucket_name, file): file for file in files}
            
            for future in as_completed(future_to_file):
                jd = future.result()
                if jd:
                    jd_data.append(jd)

        return jd_data

    def format_jd_for_query(self, jd_data):
        """Format JD data for querying."""
        if not jd_data:
            return ""
        
        formatted_jd = ""
        for jd in jd_data:
            formatted_jd += f"Job Role: {jd.get('role', 'N/A')}\n"
            formatted_jd += f"Experience Required: {jd.get('experience', 'N/A')}\n"
            formatted_jd += f"Location: {jd.get('location', 'N/A')}\n"
            formatted_jd += f"Description: {jd.get('job_description', 'N/A')}\n"
            formatted_jd += "Key Responsibilities:\n- " + "\n- ".join(jd.get('key_responsibilities', [])) + "\n"
            formatted_jd += "Qualifications:\n- " + "\n- ".join(jd.get('qualifications', [])) + "\n"
            formatted_jd += "Skills:\n- " + "\n- ".join(jd.get('skills', [])) + "\n\n"
        
        return formatted_jd.strip()

    def load_and_index(self):
        """Load JSON data and index it into Pinecone."""
        aggregated_data = self.load_json(self.aggregated_json_path)
        if aggregated_data:
            self.process_candidates(aggregated_data)

    def load_json(self, file_path):
        """Load JSON data from a file."""
        try:
            with open(file_path, 'r') as file:
                return json.load(file)
        except Exception as e:
            print(f"Error loading JSON file {file_path}: {e}")
            return None

    def process_candidates(self, json_data):
        """Process all candidates in the JSON data."""
        try:
            candidates = json_data.get("candidates", [])
            if not candidates:
                print("No candidates found in the JSON data.")
                return
            
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    print(f"Skipping invalid candidate: {candidate}")
                    continue
                self.process_candidate(candidate)  # Changed from process_candidate_data to process_candidate
        except Exception as e:
            print(f"Error processing candidates: {e}")

    def process_candidate(self, candidate):
        """Process a single candidate's data."""
        try:
            # Get candidate name from personal_info section
            candidate_id = candidate.get("personal_info", {}).get("name", "null")
            if candidate_id == "null":
                print("Skipping candidate with no name")
                return
            
            print(f"Processing candidate: {candidate_id}")  
            combined_text = self.combine_all_sections(candidate)
            self.upsert_candidate(candidate_id, combined_text)
        except Exception as e:
            print(f"Error processing candidate {candidate_id}: {e}")

    def combine_all_sections(self, candidate):
        """Combine and weight relevant sections with special handling for contact info."""
        combined_text = ""
        
        # First add personal info with contact details
        personal_info = candidate.get("personal_info", {})
        if personal_info:
            combined_text += "personal_info:\n"
            for key, value in personal_info.items():
                combined_text += f"{key}: {value}\n"
            combined_text += "\n"
        
        # Then add other sections
        sections = {
            "skills": candidate.get("skills", []),
            "projects": candidate.get("projects", []),
            "experience": candidate.get("experience", []),
            "certifications": candidate.get("certifications", []),
            "education": candidate.get("education", [])
        }
        
        weights = {
            "skills": 4, 
            "projects": 3, 
            "experience": 4, 
            "certifications": 2, 
            "education": 2
        }
        
        for section_name, section_content in sections.items():
            combined_text += (f"{section_name}: {self.json_to_text(section_content)}\n" * weights[section_name])
        
        return combined_text.strip()

    def json_to_text(self, json_data):
        """Convert JSON data to a plain text string."""
        if isinstance(json_data, dict):
            return ' '.join([f"{key}: {value}" for key, value in json_data.items() if value])
        elif isinstance(json_data, list):
            return ' '.join([self.json_to_text(item) for item in json_data if item])
        else:
            return str(json_data)

    def generate_embedding(self, text):
        """Generate embeddings for the given text using OpenAI."""
        response = client.embeddings.create(input=text, model=self.embedding_model)
        return response.data[0].embedding

    # def upsert_candidate(self, candidate_name, combined_text):
    #     """Upsert a candidate's combined data into Pinecone using their name as ID."""
    #     try:
    #         index = pc.Index(self.index_name)
    #         embedding = self.generate_embedding(combined_text)
    #         metadata = {
    #             'candidate_name': candidate_name,
    #             'content': combined_text,
    #             'raw_text': combined_text  # Store raw text for summary generation
    #         }
    #         index.upsert(vectors=[(candidate_name, embedding, metadata)])
    #         print(f"Upserted candidate {candidate_name} into Pinecone.")
    #     except Exception as e:
    #         print(f"Error upserting candidate {candidate_name}: {e}")

    def extract_contact_info(self, candidate_content):
        """Extract contact information from candidate content with more robust patterns."""
        contact_info = {
            "email": "",
            "phone": "",
            "linkedin": "",
            "github": ""
        }
        
        # More robust email pattern
        email_match = re.search(r"(?:email|e-mail|mail)[:\s]*([a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)", 
                            candidate_content, re.IGNORECASE)
        if email_match:
            contact_info["email"] = email_match.group(1).strip()
        
        # More comprehensive phone pattern
        phone_match = re.search(r"(?:phone|mobile|tel)[:\s]*([+\d\s\-()]{7,})", 
                            candidate_content, re.IGNORECASE)
        if phone_match:
            contact_info["phone"] = phone_match.group(1).strip()
        
        # Improved LinkedIn pattern (handles various formats)
        linkedin_pattern = r"(?:linkedin|linked-in|li)[:\s]*(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|company)\/[a-zA-Z0-9-]+"
        linkedin_match = re.search(linkedin_pattern, candidate_content, re.IGNORECASE)
        if linkedin_match:
            contact_info["linkedin"] = linkedin_match.group(0).strip()
            if not contact_info["linkedin"].startswith("http"):
                contact_info["linkedin"] = "https://" + contact_info["linkedin"]
        
        # Improved GitHub pattern
        github_pattern = r"(?:github|gh)[:\s]*(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9-]+"
        github_match = re.search(github_pattern, candidate_content, re.IGNORECASE)
        if github_match:
            contact_info["github"] = github_match.group(0).strip()
            if not contact_info["github"].startswith("http"):
                contact_info["github"] = "https://" + contact_info["github"]
        
        return contact_info

    def generate_candidate_summary(self, candidate_content, candidate_name):
        """Generate a detailed natural language summary of the candidate."""
        try:
            contact_info = self.extract_contact_info(candidate_content)
            
            prompt = f"""
            Generate a professional summary for the candidate named {candidate_name} based on the following profile information.
            The summary should be in markdown format with the following sections:
            
            **Candidate Summary:**
            [Provide a 3-4 sentence professional overview highlighting their experience, expertise, and career focus]
            
            **Key Skills:**
            - [List 5-8 most relevant technical skills]
            - [List 3-5 most relevant soft skills]
            
            **Contact Information:**
            - Email: {contact_info['email'] or 'Not provided'}
            - Phone: {contact_info['phone'] or 'Not provided'}
            - LinkedIn: {contact_info['linkedin'] or 'Not provided'}
            - GitHub: {contact_info['github'] or 'Not provided'}
            
            Candidate Profile Information:
            {candidate_content}
            
            Important Notes:
            1. Only include contact information that is actually provided in the profile
            2. Be factual and only include information that can be clearly derived from the profile
            3. Keep the summary professional and concise
            4. Format the output in proper markdown with clear section headings
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert HR professional that creates accurate, professional candidate summaries."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                temperature=0.3  
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            print(f"Error generating candidate summary: {e}")
            contact_info = self.extract_contact_info(candidate_content)
            return f"""
**Candidate Summary:**
Basic information available for {candidate_name}. Please review full profile for details.

**Key Skills:**
[Skills could not be automatically extracted]

**Contact Information:**
- Email: {contact_info['email'] or 'Not provided'}
- Phone: {contact_info['phone'] or 'Not provided'}
- LinkedIn: {contact_info['linkedin'] or 'Not provided'}
- GitHub: {contact_info['github'] or 'Not provided'}
            """

    def generate_markdown_report(self, candidates):
        """Generate a detailed markdown report for the selected candidates."""
        markdown = "# Selected Candidates Report\n\n"
        
        for candidate in candidates:
            candidate_name = candidate.get('candidate_name', candidate.get('candidate_id', 'Unknown Candidate'))
            markdown += f"## {candidate_name}\n"
            
            summary = self.generate_candidate_summary(
                candidate.get('content', 'No information available'),
                candidate_name
            )
            markdown += summary + "\n\n"
            
            markdown += "-------------------------------------------------------------------------\n\n"  
        
        return markdown

    def save_markdown_report(self, markdown_content, file_path="selected_candidates_report.md"):
        """Save the markdown report to a local file."""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(markdown_content)
            print(f"📄 Markdown report saved to {file_path}")
            return file_path
        except Exception as e:
            print(f"Error saving markdown report: {e}")
            return None

    def query(self, prompt=None, jd_data=None, top_k=5, confidence_threshold=0.7):
        """Query Pinecone for relevant candidates based on user query and/or JD data."""
        try:
            index = pc.Index(self.index_name)
            
            jd_text = self.format_jd_for_query([jd_data]) if jd_data else ""
            
            combined_query = ""
            if prompt:
                combined_query += f"User Query: {prompt}\n\n"
            if jd_text:
                combined_query += f"Job Description Details:\n{jd_text}"
            
            if not combined_query.strip():
                print("❌ No query or JD provided for search.")
                return []
            
            embedding = self.generate_embedding(combined_query)
            query_response = index.query(
                vector=embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            candidates = []
            for match in query_response['matches']:
                score = match['score']
                if score >= confidence_threshold:
                    candidate_data = {
                        'candidate_id': match['id'],
                        'candidate_name': match['metadata'].get('candidate_name', match['id']),
                        'score': round(score, 2),
                        'content': match['metadata'].get('content', 'No details available')
                    }
                    candidates.append(candidate_data)

            candidates.sort(key=lambda x: x['score'], reverse=True)
            
            markdown_report = self.generate_markdown_report(candidates)
            report_path = self.save_markdown_report(markdown_report)
            
            if report_path:
                print(f"\n✅ Report generated with {len(candidates)} candidates")
                print(f"📄 View the complete report at: {os.path.abspath(report_path)}")
            
            return candidates

        except Exception as e:
            print(f"Error querying Pinecone: {e}")
            return []

def process_input(input_str: str, loader: PineconeLoader) -> Optional[dict]:
    """
    Process input string - could be:
    1. S3 URI (e.g., s3://bucket-name/path/to/file)
    2. Text query (use directly)
    """
    try:
        input_str = input_str.strip().strip('"\'')  
        
        if input_str.startswith('s3://'):
            print(f"🔍 Processing S3 URI: {input_str}")
            # Parse S3 URI
            parts = input_str[5:].split('/', 1)
            if len(parts) != 2:
                print("❌ Invalid S3 URI format. Expected format: s3://bucket-name/path/to/file")
                return None
            
            bucket_name = parts[0]
            file_key = parts[1]
            
            jd_data = loader.process_jd_file(bucket_name, {"Key": file_key})
            if jd_data:
                return {"jd_data": jd_data, "source": input_str}
            return None
        else:
            print("🔍 Treating input as direct query")
            return {"query": input_str}

    except Exception as e:
        print(f"❌ Error processing input: {e}")
        return None

if __name__ == "__main__":
    print("Starting CV Automation System...")
    
    loader = PineconeLoader(aggregated_json_path="aggregated_data.json")
    
    print("\n🔨 Loading and indexing candidate data...")
    #loader.load_and_index()
    
    user_input = input("\n🔍 Enter S3 URI (s3://bucket/path) or search query: ").strip()
    
    if not user_input:
        print("👋 No input provided - exiting.")
    else:
        result = process_input(user_input, loader)
        
        if not result:
            print("⚠️ No valid input processed")
        elif "jd_data" in result:
            print(f"\n🔎 Searching with JD from {result['source']}")
            loader.query(jd_data=result["jd_data"])
        elif "query" in result:
            print(f"\n🔎 Searching for: {result['query']}")
            loader.query(prompt=result['query'])
    
    print("\n👋 Script completed.")
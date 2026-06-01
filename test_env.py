from dotenv import load_dotenv
import os

load_dotenv()
print(repr(os.getenv("SS_API_KEY")))
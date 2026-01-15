import sys
import os

# Add the backend directory to the path so we can import our modules
# This allows Vercel to find 'app.main' correctly even when running from root
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app.main import app

# Vercel looks for 'app' as the entry point
# Alternatively, we could rename the FastAPI instance in main.py to 'app'
# which we have already done.

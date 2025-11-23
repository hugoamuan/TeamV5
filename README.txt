HelpMeLandAJob - FastAPI Server
- Host for the local llama model and job-scraping services

# app.py
- main entry point of the application
- creates the FastAPI() instance
- includes all route modules 
- adding CORS middleware
- basic endpoints: /health and /model-info

# routes/
- folder containing all the api endpoints
- defines route paths
- receieves requests and calls the appropriate service layer 
- returns json responses

# services/
- folder containing core logic 
- llm_service.py loads the local AI model and handles chat_completion() and summarize_job()
   - handles chunking long job descriptions
   - separates AI functionalities from the API
- scrape_service.py scrapes LinkedIn seeMoreJobPostings pages
   - extracts job title, company, location, short summary, full description
   - uses caching so repeated calls don't overload LinkedIn
- match_service.py
   - performs token-based exact skill matching
   - ensure "java" does not match "javascript"


# --------------
# Linux Shell commands
# --------------
ssh root@167.172.11.6.168    // SSH into the DigitalOcean Droplet

# Dependencies
apt update && apt upgrade -y
apt install python3 python3-venv python3-pip git -y

# Create venv for isolated environment
mkdir -p /root/ai-llm
cd /root/ai-llm
python3 -m venv .venv
source .venv/bin/activate

# Install FastAPI, Uvicorn and llama-cpp
# llama allows for python to call C++ routines
pip install fastapi uvicorn[standard] llama-cpp-python

# Launch API Server
cd /root/ai-llm
uvicorn app:app --host 0.0.0.0 --port 8000

# Server functions
systemctl daemon-reload
systemctl enable ai-llm
systemctl start ai-llm
systemctl status ai-llm 

# systemd is main service manager (automatic restarts, booting, daemons, dependencies)
nano /etc/systemd/system/ai-llm.service

# NGINX
sudo apt update
sudo apt install nginx

sudo nano /etc/nginx/sites-available/teamv5 // nginx config

sudo ln -s /etc/nginx/sites-available/teamv5 /etc/nginx/sites-enabled/    //enable config

# Certbot
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d teamv5.duckdns.org  // generate https certificate and auto-edit nginx config

sudo certbot renew --dry-run // test renewal

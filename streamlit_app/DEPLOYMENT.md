# Deployment Guide for Cline Web IDE

This guide provides instructions for deploying Cline Web IDE in different environments.

## Table of Contents
- [Local Development](#local-development)
- [Production Deployment](#production-deployment)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)

## Local Development

For local development and testing:

1. Set up the environment:
```bash
python run.py --setup
```

2. Launch the application in development mode:
```bash
python run.py --no-auth --debug
```

3. Access the application at http://localhost:8501

## Production Deployment

For deploying on a server:

1. Set up the server environment:
```bash
python server.py --setup
```

2. Configure environment variables:
```
ANTHROPIC_API_KEY=your_api_key_here
FIREBASE_PRIVATE_KEY_ID=your_key_id_here
FIREBASE_PRIVATE_KEY=your_private_key_here
FIREBASE_CLIENT_EMAIL=your_client_email_here
FIREBASE_CLIENT_ID=your_client_id_here
FIREBASE_CERT_URL=your_cert_url_here
```

3. Launch the server:
```bash
python server.py --workers 4
```

4. For high-availability, consider using a process manager like supervisor:
```bash
# Install supervisor
apt-get install supervisor -y

# Create a config file
cat > /etc/supervisor/conf.d/cline-web-ide.conf << EOF
[program:cline-web-ide]
command=python /path/to/cline-web-ide/server.py
directory=/path/to/cline-web-ide
autostart=true
autorestart=true
stderr_logfile=/var/log/cline-web-ide.err.log
stdout_logfile=/var/log/cline-web-ide.out.log
user=your_username
environment=ANTHROPIC_API_KEY="your_api_key_here"
EOF

# Reload supervisor
supervisorctl reread
supervisorctl update
```

## Docker Deployment

For deploying with Docker:

1. Create a `.env` file with your configuration:
```
ANTHROPIC_API_KEY=your_api_key_here
DISABLE_AUTH=false
```

2. Build and start the container:
```bash
docker-compose up -d
```

3. Access the application at http://localhost:8501

4. To stop the container:
```bash
docker-compose down
```

### Docker Compose with HTTPS

For a more secure setup with HTTPS using Traefik:

```yaml
version: '3.8'

services:
  cline-web-ide:
    build: .
    container_name: cline-web-ide
    restart: unless-stopped
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - ./data:/app/data
      - ./temp:/app/temp
    labels:
      - "traefik.enable=true"
      - "traefik.http.routers.cline.rule=Host(`cline.yourdomain.com`)"
      - "traefik.http.routers.cline.entrypoints=websecure"
      - "traefik.http.routers.cline.tls.certresolver=myresolver"

  traefik:
    image: traefik:v2.9
    container_name: traefik
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
      - ./traefik/traefik.yml:/etc/traefik/traefik.yml
      - ./traefik/acme.json:/acme.json
```

## Cloud Deployment

### Heroku Deployment

1. Create a `Procfile`:
```
web: streamlit run app.py --server.port=$PORT
```

2. Push to Heroku:
```bash
git init
git add .
git commit -m "Initial commit"
heroku create cline-web-ide
git push heroku master
heroku config:set ANTHROPIC_API_KEY=your_api_key_here
```

### AWS Deployment

1. Create an EC2 instance (Ubuntu 20.04 LTS recommended)
2. Install dependencies:
```bash
sudo apt update
sudo apt install -y python3 python3-pip
git clone https://github.com/yourusername/cline-web-ide.git
cd cline-web-ide
pip3 install -r requirements.txt
```

3. Set up Nginx as a reverse proxy:
```bash
sudo apt install -y nginx
sudo cat > /etc/nginx/sites-available/cline-web-ide << EOF
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
    }
}
EOF

sudo ln -s /etc/nginx/sites-available/cline-web-ide /etc/nginx/sites-enabled/
sudo systemctl restart nginx
```

4. Set up SSL with Let's Encrypt:
```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

5. Launch the application with supervisor (as shown in the Production Deployment section)

### Google Cloud Run Deployment

1. Build and push Docker image:
```bash
gcloud builds submit --tag gcr.io/your-project/cline-web-ide
```

2. Deploy to Cloud Run:
```bash
gcloud run deploy cline-web-ide \
  --image gcr.io/your-project/cline-web-ide \
  --platform managed \
  --allow-unauthenticated \
  --set-env-vars="ANTHROPIC_API_KEY=your_api_key_here,DISABLE_AUTH=false"
```

## Security Considerations

1. Always use HTTPS in production environments
2. Keep API keys and secrets secure
3. Enable authentication in production
4. Implement rate limiting for API requests
5. Regularly update dependencies
6. Consider using a web application firewall (WAF)

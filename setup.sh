#!/bin/bash

# Update and install Docker
sudo apt update -y
sudo apt install -y docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER
sudo systemctl enable docker

# Install Caddy
# Fetch the latest release version from GitHub
LATEST_RELEASE=$(curl -s https://api.github.com/repos/caddyserver/caddy/releases/latest | grep 'tag_name' | cut -d\" -f4)
CADDY_URL="https://github.com/caddyserver/caddy/releases/download/${LATEST_RELEASE}/caddy_${LATEST_RELEASE#v}_linux_amd64.tar.gz"

# Download and install Caddy
curl -fsSL $CADDY_URL | sudo tar xzf - -C /usr/local/bin
sudo chmod +x /usr/local/bin/caddy

# Verify the installation
if [ ! -x /usr/local/bin/caddy ]; then
  echo "Caddy installation failed!"
  exit 1
fi

# Create Caddyfile
sudo bash -c 'cat > /etc/caddy/Caddyfile' <<EOF
streamlit-server.duckdns.org {
    reverse_proxy localhost:8501
}
EOF

sudo systemctl daemon-reload
sudo systemctl start caddy
sudo systemctl enable caddy

# Pull the latest Docker image and deploy
sudo docker pull mohamed06/wf_streamlit:latest
sudo docker stop streamlit-app || true
sudo docker rm streamlit-app || true
sudo docker run -d -p 8501:8501 --name streamlit-app mohamed06/wf_streamlit:latest

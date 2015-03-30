# Install and configure lmktfyctl

## Download the lmktfyctl CLI tool

### Darwin

```
wget https://storage.googleapis.com/lmktfy-release/release/v0.10.1/bin/darwin/amd64/lmktfyctl
```

### Linux

```
wget https://storage.googleapis.com/lmktfy-release/release/v0.10.1/bin/linux/amd64/lmktfyctl
```

### Copy lmktfyctl to your path

```
chmod +x lmktfyctl
mv lmktfyctl /usr/local/bin/
```

### Create a secure tunnel for API communication

```
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

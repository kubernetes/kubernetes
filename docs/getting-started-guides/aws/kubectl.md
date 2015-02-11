# Install and configure kubectl

## Download the kubectl CLI tool

### Darwin

```
wget https://storage.googleapis.com/kubernetes-release/release/v0.9.2/bin/darwin/amd64/kubectl
```

### Linux

```
wget https://storage.googleapis.com/kubernetes-release/release/v0.9.2/bin/linux/amd64/kubectl
```

### Copy kubectl to your path

```
chmod +x kubectl
mv kubectl /usr/local/bin/
```

### Create a secure tunnel for API communication

```
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

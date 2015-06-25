# Install and configure kubectl

## Download the kubectl CLI tool
```bash
### Darwin
wget https://storage.googleapis.com/kubernetes-release/release/v0.17.0/bin/darwin/amd64/kubectl

### Linux
wget https://storage.googleapis.com/kubernetes-release/release/v0.17.0/bin/linux/amd64/kubectl
```

### Copy kubectl to your path
```bash
chmod +x kubectl
mv kubectl /usr/local/bin/
```

### Create a secure tunnel for API communication
```bash
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws/kubectl.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/getting-started-guides/aws/kubectl.md?pixel)]()

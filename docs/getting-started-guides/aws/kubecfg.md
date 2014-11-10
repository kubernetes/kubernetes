# Install and configure kubecfg

## Downlaod the kubecfg cli tool

### Darwin

```
wget http://storage.googleapis.com/k8s/darwin/kubecfg
```

### Linux

```
wget http://storage.googleapis.com/k8s/darwin/kubecfg
```

### Copy kubecfg to your path

```
chmod +x kubecfg
mv kubecfg /usr/local/bin/
```

### Create a secure tunnel for API communication

```
ssh -f -nNT -L 8080:127.0.0.1:8080 core@<master-public-ip>
```

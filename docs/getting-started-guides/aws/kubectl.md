<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Install and configure kubectl

## Download the kubectl CLI tool

```bash
### Darwin
wget https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/darwin/amd64/kubectl

### Linux
wget https://storage.googleapis.com/kubernetes-release/release/v1.0.1/bin/linux/amd64/kubectl
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

<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws/kubectl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

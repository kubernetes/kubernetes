<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
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


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws/kubectl.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

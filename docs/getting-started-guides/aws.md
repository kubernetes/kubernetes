## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

### Cluster turnup

#### Download Kubernetes
##### a) Preferred Option: Install from [0.5 release](https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.5)
1. ```wget https://github.com/GoogleCloudPlatform/kubernetes/releases/download/v0.5/kubernetes.tar.gz```
2. ```tar -xzf kubernetes.tar.gz; cd kubernetes```
3. ```export PATH=$PATH:$PWD/platforms/<os>/<platform>```
4. __Temporary for v0.5__ : Edit the ```cluster/aws/config-default.sh``` so that ```IMAGE=ami-39501209``` 

##### b) Alternate Option: Install from source at head
1. ```git clone https://github.com/GoogleCloudPlatform/kubernetes.git```
2. ```cd kubernetes; make release```
3. ```export PATH=$PATH:$PWD/_output/local/bin/<os>/<platform>```

#### Turn up the cluster
```
export KUBERNETES_PROVIDER=aws
cluster/kube-up.sh
```

The script above relies on AWS S3 to deploy the software to instances running in EC2.

Once the cluster is up, it will print the ip address of your cluster.

```
export KUBERNETES_MASTER=https://<ip-address>
```

Also setup your path to point to the released binaries:
```

```

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes#where-to-go-next)

### Tearing down the cluster
```
cd kubernetes
cluster/kube-down.sh
```

### Cloud Formation [optional]
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.

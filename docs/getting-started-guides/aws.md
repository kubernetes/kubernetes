## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)

### Cluster turnup

#### Preferred Option: Install from [0.5 release](https://github.com/GoogleCloudPlatform/kubernetes/releases/tag/v0.5)
1. ```wget https://github.com/GoogleCloudPlatform/kubernetes/releases/download/v0.5/kubernetes.tar.gz```
2. ```tar -xzf kubernetes.tar.gz```
3. ```cd kubernetes```

#### Alternate Option: Install from source at head
1. ```git clone https://github.com/GoogleCloudPlatform/kubernetes.git```
2. ```cd kubernetes; make release```

#### Turn up the cluster
```
export KUBERNETES_PROVIDER=aws
cluster/kube-up.sh
```

The script above relies on AWS S3 to deploy the software to instances running in EC2.

Once the cluster is up, it will print the ip address of your cluster.

```
export PATH=$PATH:$PWD/platforms/<os>/<platform>
export KUBERNETES_MASTER=https://<ip-address>
```

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes#where-to-go-next)

### Tearing down the cluster
```
cd kubernetes
cluster/kube-down.sh
```

### Cloud Formation
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.

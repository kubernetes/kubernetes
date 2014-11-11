## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)

### Cluster turnup

#### Install from source
1. ```git clone https://github.com/GoogleCloudPlatform/kubernetes.git```
2. ```cd kubernetes; make release```

#### Turn up the cluster
```
export KUBERNETES_PROVIDER=aws
cluster/kube-up.sh
```

The script above relies on AWS S3 to deploy the software to instances running in EC2.

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes#where-to-go-next)

### Tearing down the cluster
```
cd kubernetes
cluster/kube-down.sh
```

### Cloud Formation
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.

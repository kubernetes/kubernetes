## Getting started on AWS

### Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

### Cluster turnup

Using ```wget```
```sh
export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash
```

or if you prefer ```curl```

```sh
export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash
```

NOTE: This script calls [cluster/kube-up.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/kube-up.sh)
which in turn calls [cluster/aws/util.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/aws/util.sh)
using [cluster/aws/config-default.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/aws/config-default.sh).
By default, the script will provision a new VPC and a 4 node k8s cluster in us-west-2 (Oregon). It'll also try to create or reuse
a keypair called "kubernetes", and IAM profiles called "kubernetes-master" and "kubernetes-minion".  If these already exist, make
sure you want them to be used here. You can override the variables defined in config-default.sh to change this behavior.

Once the cluster is up, it will print the ip address of your cluster, this process takes about 5 to 10 minutes.

```
export KUBERNETES_MASTER=https://<ip-address>
```

Copy the appropriate ```kubectl``` binary to somewhere in your ```PATH```, for example:

```bash
# OS X
sudo cp kubernetes/platforms/darwin/amd64/kubectl /usr/local/bin/kubectl

# Linux
sudo cp kubernetes/platforms/linux/amd64/kubectl /usr/local/bin/kubectl
```


### Getting started with your cluster
See [a simple nginx example](../../examples/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples)

### Tearing down the cluster
```bash
cd kubernetes
cluster/kube-down.sh
```

### Running examples

Take a look at [next steps](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook)

### Cloud Formation [optional]
There is a contributed [example](aws-coreos.md) from [CoreOS](http://www.coreos.com) using Cloud Formation.

### Further reading
Please see the [Kubernetes docs](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs) for more details on administering and using a Kubernetes cluster.

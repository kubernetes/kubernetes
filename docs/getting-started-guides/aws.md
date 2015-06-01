# Getting started on AWS EC2

## Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

## Cluster turnup
### Supported procedure: `get-kube`
```bash
#Using wget
export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash

#Using cURL
export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash
```

NOTE: This script calls [cluster/kube-up.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/kube-up.sh)
which in turn calls [cluster/aws/util.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/aws/util.sh)
using [cluster/aws/config-default.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/aws/config-default.sh).

This process takes about 5 to 10 minutes. Once the cluster is up, the IP addresses of your master and node(s) will be printed,
as well as information about the default services running in the cluster (monitoring, logging, dns). User credentials and security
tokens are written in `~/.kube/kubeconfig`, they will be necessary to use the CLI or the HTTP Basic Auth.

By default, the script will provision a new VPC and a 4 node k8s cluster in us-west-2a (Oregon) with `t2.micro` instances running on Ubuntu.
You can override the variables defined in [config-default.sh](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/cluster/aws/config-default.sh) to change this behavior as follows:

```bash
export KUBE_AWS_ZONE=eu-west-1c
export NUM_MINIONS=2
export MINION_SIZE=m3.medium
export AWS_S3_REGION=eu-west-1
export AWS_S3_BUCKET=mycompany-kubernetes-artifacts
export INSTANCE_PREFIX=k8s
...
```

It will also try to create or reuse a keypair called "kubernetes", and IAM profiles called "kubernetes-master" and "kubernetes-minion".
If these already exist, make sure you want them to be used here.

NOTE: If using an existing keypair named "kubernetes" then you must set the `AWS_SSH_KEY` key to point to your private key.

### Alternatives
A contributed [example](aws-coreos.md) allows you to setup a Kubernetes cluster based on [CoreOS](http://www.coreos.com), either using
AWS CloudFormation or EC2 with user data (cloud-config).

## Getting started with your cluster
### Command line administration tool: `kubectl`
Copy the appropriate `kubectl` binary to any location defined in your `PATH` environment variable, for example:

```bash
# OS X
sudo cp kubernetes/platforms/darwin/amd64/kubectl /usr/local/bin/kubectl

# Linux
sudo cp kubernetes/platforms/linux/amd64/kubectl /usr/local/bin/kubectl
```

An up-to-date documentation page for this tool is available here: [kubectl manual](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubectl.md)

By default, `kubectl` will use the `kubeconfig` file generated during the cluster startup for authenticating against the API.
For more information, please read [kubeconfig files](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/docs/kubeconfig-file.md)

### Examples
See [a simple nginx example](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples/simple-nginx.md) to try out your new cluster.

The "Guestbook" application is another popular example to get started with Kubernetes: [guestbook example](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/examples/guestbook)

For more complete applications, please look in the [examples directory](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/examples)

## Tearing down the cluster
Make sure the environment variables you used to provision your cluster are still exported, then call the following script inside the
`kubernetes` directory:

```bash
cluster/kube-down.sh
```

## Further reading
Please see the [Kubernetes docs](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs) for more details on administering
and using a Kubernetes cluster.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws.md?pixel)]()

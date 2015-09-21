<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

<strong>
The latest 1.0.x release of this document can be found
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/aws.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started on AWS EC2
--------------------------

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Cluster turnup](#cluster-turnup)
    - [Supported procedure: `get-kube`](#supported-procedure-get-kube)
    - [Alternatives](#alternatives)
- [Getting started with your cluster](#getting-started-with-your-cluster)
    - [Command line administration tool: `kubectl`](#command-line-administration-tool-kubectl)
    - [Examples](#examples)
- [Tearing down the cluster](#tearing-down-the-cluster)
- [Further reading](#further-reading)

## Prerequisites

1. You need an AWS account. Visit [http://aws.amazon.com](http://aws.amazon.com) to get started
2. Install and configure [AWS Command Line Interface](http://aws.amazon.com/cli)
3. You need an AWS [instance profile and role](http://docs.aws.amazon.com/IAM/latest/UserGuide/instance-profiles.html) with EC2 full access.

NOTE: This script use the 'default' AWS profile by default.
You may explicitly set AWS profile to use using the `AWS_DEFAULT_PROFILE` environment variable:

```bash
export AWS_DEFAULT_PROFILE=myawsprofile
```

## Cluster turnup

### Supported procedure: `get-kube`

```bash
#Using wget
export KUBERNETES_PROVIDER=aws; wget -q -O - https://get.k8s.io | bash

#Using cURL
export KUBERNETES_PROVIDER=aws; curl -sS https://get.k8s.io | bash
```

NOTE: This script calls [cluster/kube-up.sh](http://releases.k8s.io/HEAD/cluster/kube-up.sh)
which in turn calls [cluster/aws/util.sh](http://releases.k8s.io/HEAD/cluster/aws/util.sh)
using [cluster/aws/config-default.sh](http://releases.k8s.io/HEAD/cluster/aws/config-default.sh).

This process takes about 5 to 10 minutes. Once the cluster is up, the IP addresses of your master and node(s) will be printed,
as well as information about the default services running in the cluster (monitoring, logging, dns). User credentials and security
tokens are written in `~/.kube/config`, they will be necessary to use the CLI or the HTTP Basic Auth.

By default, the script will provision a new VPC and a 4 node k8s cluster in us-west-2a (Oregon) with `t2.micro` instances running on Ubuntu.
You can override the variables defined in [config-default.sh](http://releases.k8s.io/HEAD/cluster/aws/config-default.sh) to change this behavior as follows:

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

A contributed [example](coreos/coreos_multinode_cluster.md) allows you to setup a Kubernetes cluster based on [CoreOS](http://www.coreos.com), using
EC2 with user data (cloud-config).

## Getting started with your cluster

### Command line administration tool: `kubectl`

The cluster startup script will leave you with a `kubernetes` directory on your workstation.
Alternately, you can download the latest Kubernetes release from [this page](https://github.com/kubernetes/kubernetes/releases).

Next, add the appropriate binary folder to your `PATH` to access kubectl:

```bash
# OS X
export PATH=<path/to/kubernetes-directory>/platforms/darwin/amd64:$PATH

# Linux
export PATH=<path/to/kubernetes-directory>/platforms/linux/amd64:$PATH
```

An up-to-date documentation page for this tool is available here: [kubectl manual](../../docs/user-guide/kubectl/kubectl.md)

By default, `kubectl` will use the `kubeconfig` file generated during the cluster startup for authenticating against the API.
For more information, please read [kubeconfig files](../../docs/user-guide/kubeconfig-file.md)

### Examples

See [a simple nginx example](../../docs/user-guide/simple-nginx.md) to try out your new cluster.

The "Guestbook" application is another popular example to get started with Kubernetes: [guestbook example](../../examples/guestbook/)

For more complete applications, please look in the [examples directory](../../examples/)

## Tearing down the cluster

Make sure the environment variables you used to provision your cluster are still exported, then call the following script inside the
`kubernetes` directory:

```bash
cluster/kube-down.sh
```

## Further reading

Please see the [Kubernetes docs](../../docs/) for more details on administering
and using a Kubernetes cluster.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/aws.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

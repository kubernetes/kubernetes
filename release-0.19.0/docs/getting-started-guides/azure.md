## Getting started on Microsoft Azure

### Azure Prerequisites

1. You need an Azure account. Visit http://azure.microsoft.com/ to get started.
2. Install and configure the Azure cross-platform command-line interface. http://azure.microsoft.com/en-us/documentation/articles/xplat-cli/
3. Make sure you have a default account set in the Azure cli, using `azure account set`

### Prerequisites for your workstation

1. Be running a Linux or Mac OS X.
2. Get or build a [binary release](binary_release.md)
3. If you want to build your own release, you need to have [Docker
installed](https://docs.docker.com/installation/).  On Mac OS X you can use
[boot2docker](http://boot2docker.io/).

### Setup
The cluster setup scripts can setup Kubernetes for multiple targets. First modify `cluster/kube-env.sh` to specify azure:

    KUBERNETES_PROVIDER="azure"

Next, specify an existing virtual network and subnet in `cluster/azure/config-default.sh`:

    AZ_VNET=<vnet name>
    AZ_SUBNET=<subnet name>

You can create a virtual network:

    azure network vnet create <vnet name> --subnet=<subnet name> --location "West US" -v

Now you're ready.

You can then use the `cluster/kube-*.sh` scripts to manage your azure cluster, start with:

    cluster/kube-up.sh

The script above will start (by default) a single master VM along with 4 worker VMs.  You
can tweak some of these parameters by editing `cluster/azure/config-default.sh`.

### Getting started with your cluster
See [a simple nginx example](../../examples/simple-nginx.md) to try out your new cluster.

For more complete applications, please look in the [examples directory](../../examples).

### Tearing down the cluster
```
cluster/kube-down.sh
```


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/azure.md?pixel)]()


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/release-0.19.0/docs/getting-started-guides/azure.md?pixel)]()

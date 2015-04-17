# Getting Started

## Environment Considerations

Kubernetes has specific cloud provider integration, and as of the current writing of this document that supported list includes the official Juju supported providers:

- [Amazon AWS](https://jujucharms.com/docs/config-aws)
- [Azure](https://jujucharms.com/docs/config-azure)
- [Vagrant](https://jujucharms.com/docs/config-vagrant)

Other providers available for use as a *juju manual environment* can be listed in the [Kubernetes Documentation](https://github.com/GoogleCloudPlatform/kubernetes/tree/master/docs/getting-started-guides)

## Deployment

The Kubernetes Charms are currently under heavy development. We encourage you to fork these charms and contribute back to the development effort! See our [contributing](contributing.md) doc for more information on this.

#### Deploying the Preview Release charms

    juju deploy cs:~hazmat/trusty/etcd
    juju deploy cs:~hazmat/trusty/flannel
    juju deploy local:trusty/kubernetes-master
    juju deploy local:trusty/kubernetes

    juju add-relation etcd flannel
    juju add-relation etcd kubernetes
    juju add-relation etcd kubernetes-master
    juju add-relation kubernetes kubernetes-master

#### Deploying the Development Release Charms

> These charms are known to be unstable as they are tracking the current efforts of the community at enabling different features against Kubernetes. This includes the specifics for integration per cloud environment, and upgrading to the latest development version.

    mkdir -p ~/charms/trusty
    git clone https://github.com/whitmo/kubernetes-master.git ~/charms/trusty/kubernetes-master
    git clone https://github.com/whitmo/kubernetes.git ~/charms/trusty/kubernetes

##### Skipping the manual deployment after git clone

> **Note:** This path requires the pre-requisite of juju-deployer. You can obtain juju-deployer via `apt-get install juju-deployer`

    wget https://github.com/whitmo/bundle-kubernetes/blob/master/develop.yaml kubernetes-devel.yaml
    juju-deployer kubernetes-devel.yaml


## Verifying Deployment with the Kubernetes Agent

You'll need the kubernetes command line client to utlize the created cluster. And this can be fetched from the [Releases](https://github.com/GoogleCloudPlatform/kubernetes/releases) page on the Kubernetes project. Make sure you're fetching a client library that matches what the charm is deploying.

Grab the tarball and from the extracted release you can just directly use the cli binary at ./kubernetes/platforms/linux/amd64/kubecfg

You'll need the address of the kubernetes master as environment variable :

    juju status kubernetes-master/0

Grab the public-address there and export it as KUBERNETES_MASTER environment variable :

    export KUBERNETES_MASTER=$(juju status --format=oneline kubernetes-master | cut -d' ' -f3):8080

And now you can run through the kubernetes examples per normal. :

    kubecfg list minions


## Scale Up

If the default capacity of the bundle doesn't provide enough capacity for your workload(s) you can scale horizontially by adding a unit to the flannel and kubernetes services respectively.

    juju add-unit flannel
    juju add-unit kubernetes --to # (machine id of new flannel unit)

## Known Issues / Limitations

Kubernetes currently has platform specific functionality. For example load balancers and persistence volumes only work with the google compute provider atm.

The Juju integration uses the kubernetes null provider. This means external load balancers and storage can't be directly driven through kubernetes config files.

## Where to get help

If you run into any issues, file a bug at our [issue tracker](http://github.com/whitmo/kubernetes-charm/issues), email the Juju Mailing List at &lt;juju@lists.ubuntu.com&gt;, or feel free to join us in #juju on irc.freenode.net.



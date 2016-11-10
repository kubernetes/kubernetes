# Cluster Federation

Kubernetes Cluster Federation enables users to federate multiple
Kubernetes clusters. Please see the [user guide](http://kubernetes.io/docs/admin/federation/)
and the [admin guide](http://kubernetes.io/docs/user-guide/federation/federated-services/)
for more details about setting up and using the Cluster Federation.

# Building Kubernetes Cluster Federation

Please see the [Kubernetes Development Guide](https://github.com/kubernetes/kubernetes/blob/master/docs/devel/development.md)
for initial setup. Once you have the development environment setup
as explained in that guide, you also need to install [`jq`](https://stedolan.github.io/jq/download/)
<!-- TODO(madhusudancs): Re-evaluate using jq even in the development
     environment. There is a concern that adding more tools as dependencies
     might lead to proliferation of tools one need to install to develop
     Kubernetes. jq is already a dependency for kubernetes-anywhere on
     which this workflow depends, so we are giving an exception to jq
     for now. -->

Building cluster federation artifacts should be as simple as running:

```shell
make build
```

You can specify the docker registry to tag the image using the
KUBE_REGISTRY environment variable. Please make sure that you use
the same value in all the subsequent commands.

To push the built docker images to the registry, run:

```shell
make push
```

To initialize the deployment run:

(This pulls the installer images)

```shell
make init
```

To deploy the clusters and install the federation components, edit the
`${KUBE_ROOT}/_output/federation/config.json` file to describe your
clusters and run:

```shell
make deploy
```

To turn down the federation components and tear down the clusters run:

```shell
make destroy
```

# Ideas for improvement

1. Continue with `destroy` phase even in the face of errors.

   The bash script sets `set -e errexit` which causes the script to exit
   at the very first error. This should be the default mode for deploying
   components but not for destroying/cleanup.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/federation/README.md?pixel)]()

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

Building cluster federation should be as simple as running:

```shell
make build do=gen
```

To deploy clusters and install federation components, edit the
`config.default.json` file to describe your clusters and run

```shell
make build do=deploy
```

To turn down the federation components and tear down the clusters run:

```shell
make build do=destroy
```

# Ideas for improvement

1. Split the `build` phase (make recipe) into multiple phases:
    1. `init`: pull installer images
    2. `build-binaries`
    3. `build-docker`
    4. `build`: build-binary + build-docker
    5. `push`: to push the built images
    6. `genconfig`
    7. `deploy-clusters`
    8. `deploy-federation`
    9. `deploy`: deploy-clusters + deploy-federation
    10. `destroy-federation`
    11. `destroy-clusters`
    12. `destroy`: destroy-federation + destroy-clusters
    13. `redeploy-federation`: just redeploys the federation components.

2. Add a `release` phase to run as part of Kubernetes release process
   that copies only a part of the `build.sh` script that's relevant to
   the users into the release.

3. Continue with `destroy` phase even in the face of errors.

   The bash script sets `set -e errexit` which causes the script to exit
   at the very first error. This should be the default mode for deploying
   components but not for destroying/cleanup.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/federation/README.md?pixel)]()

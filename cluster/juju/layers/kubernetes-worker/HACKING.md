 # Kubernetes Worker

### Building from the layer

You can clone the kubernetes-worker layer with git and build locally if you
have the charm package/snap installed.

```shell
# Instal the snap
sudo snap install charm --channel=edge

# Set the build environment
export JUJU_REPOSITORY=$HOME

# Clone the layer and build it to our JUJU_REPOSITORY
git clone https://github.com/juju-solutions/kubernetes
cd kubernetes/cluster/juju/layers/kubernetes-worker
charm build -r
```

### Contributing

TBD



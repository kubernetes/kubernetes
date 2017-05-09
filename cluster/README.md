# Cluster Configuration

##### Deprecation Notice: This directory has entered maintenance mode and will not be accepting new providers. Please submit new automation deployments to [kube-deploy](https://github.com/kubernetes/kube-deploy). Deployments in this directory will continue to be maintained and supported at their current level of support.

The scripts and data in this directory automate creation and configuration of a Kubernetes cluster, including networking, DNS, nodes, and master components.

Tools in this folder are `bash` scripts which make managing single cluster easier, but they are currently not designed to allow easy management of multiple clusters from a single machine. If it's necessary it is possible to replicate existing e2e test cluster setup that consists mainly of `hack/ginkgo-e2e.sh` script and special test configs for specific cloud providers e.g. `cluster/gce/config-test.sh`. This means that in general scripts from `cluster` directory find a running cluster by themselves, and you don't need to explicitly specify the cluster you want to modify, as those tools assume that there's the cluster running.

Note the `cluster/kube-env.sh` script that defines the `KUBERNETES_PROVIDER` variable which is used by all other scripts. Default value is `gce` so if you want to manage non-GCE cluster you need to assign an identifier of your provider (e.g. `gke`, `aws`, `azure`, `vagrant`, etc.) to `KUBERNETES_PROVIDER` variable e.g. by putting the line
```bash
export KUBERNETES_PROVIDER=vagrant
```
to your `.bashrc` or `.profile` file.

See the [getting-started guides](../docs/getting-started-guides) for examples of how to use the scripts.

*cloudprovider*/`config-default.sh` contains a set of tweakable definitions/parameters for the cluster.

The heavy lifting of configuring the VMs is done by [SaltStack](http://www.saltstack.com/).


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/README.md?pixel)]()

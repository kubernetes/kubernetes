# Rackspace
In general, the dev-build-and-up.sh workflow for Rackspace is the similar to GCE. The specific implementation is different due to the use of CoreOS and network design.


NOTE: The rackspace scripts do NOT rely on saltstack.

For older versions please either:
* Sync back to `v0.3` with `git checkout v0.3`
* Download a [snapshot of `v0.3`](https://github.com/GoogleCloudPlatform/kubernetes/archive/v0.3.tar.gz)

The current cluster design is inspired by:
- [corekube](https://github.com/metral/corekube/)
- [Angus Lees](https://github.com/anguslees/kube-openstack/)

## Prerequisites
1. You need to have both `nova` and `swiftly` installed. It's recommended to use a python virtualenv to install these packages into.
2. Make sure you have the appropriate environment variables set to interact with the OpenStack APIs. See [Rackspace Documentation](http://docs.rackspace.com/servers/api/v2/cs-gettingstarted/content/section_gs_install_nova.html) for more details.
3. You can test this by running `nova list` to make sure you're authenticated successfully.

## Provider: Rackspace
- To use Rackspace as the provider, set the KUBERNETES_PROVIDER ENV variable:
  `export KUBERNETES_PROVIDER=rackspace` and run the `bash cluster/kube-up.sh` script.

## Build
1. The kubernetes binaries will be built via the common build scripts in `release/`. There is a specific `release/rackspace` directory with scripts for the following steps:
2. A cloud files container will be created via the `swiftly` CLI and a temp URL will be enabled on the object.
3. The built `master-release.tar.gz` will be uploaded to this container and the URL will be passed to master/minions nodes when booted.
- NOTE: RELEASE tagging and launch scripts are not used currently.

## Cluster
1. There is a specific `cluster/rackspace` directory with the scripts for the following steps:
2. A cloud network will be created and all instances will be attached to this network. We will connect the master API and minion kubelet service via this network.
3. A SSH key will be created and uploaded if needed. This key must be used to ssh into the machines since we won't capture the password.
4. A master and minions will be created via the `nova` CLI. A `cloud-config.yaml` is generated and provided as user-data with the entire configuration for the systems.
5. We then boot as many minions as defined via `$RAX_NUM_MINIONS`.

## Some notes:
- The scripts expect `eth2` to be the cloud network that the containers will communicate across.
- A number of the items in `config-default.sh` are overridable via environment variables.

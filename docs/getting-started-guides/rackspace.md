# Rackspace
In general, the dev-build-and-up.sh workflow for Rackspace is the similar to GCE. The specific implementation is different mainly due to network differences between the providers:

## Prerequisites
1. You need to have both `nova` and `swiftly` installed. It's recommended to use a python virtualenv to install these packages into.
2. Make sure you have the appropriate environment variables set to interact with the OpenStack APIs. See [Rackspace Documentation](http://docs.rackspace.com/servers/api/v2/cs-gettingstarted/content/section_gs_install_nova.html) for more details.
3. You can test this by running `nova list` to make sure you're authenticated successfully.

## Provider: Rackspace
- To use Rackspace as the provider, set the KUBERNETES_PROVIDER ENV variable:
  `export KUBERNETES_PROVIDER=rackspace` and run the `hack/rackspace/dev-build-and-up.sh` script.

## Release
1. The kubernetes binaries will be built via the common build scripts in `release/`. There is a specific `release/rackspace` directory with scripts for the following steps:
2. A cloud files contianer will be created via the `swiftly` CLI and a temp URL will be enabled on the object.
3. The built `master-release.tar.gz` will be uploaded to this container and the URL will be passed to master/minions nodes when booted.
- NOTE: RELEASE tagging and launch scripts are not used currently.

## Cluster
1. There is a specific `cluster/rackspace` directory with the scripts for the following steps:
2. A cloud network will be created and all instances will be attached to this network. We will connect the master API and minion kubelet service via this network.
3. A SSH key will be created and uploaded if needed. This key must be used to ssh into the machines since we won't capture the password.
4. A master will be created via the `nova` CLI. A `cloud-config.yaml` is generated and provided as user-data. A basic `masterStart.sh` will be injected as a file and cloud-init will run it.
5. We sleep for 25 seconds since we need to make sure we can get the IP address of the master on the cloud network we've created to provide the minions as their salt master.
6. We then boot as many minions as defined via `$RAX_NUM_MINIONS`. We pass both a `cloud-config.yaml` as well as a `minionStart.sh`. The latter is executed via cloud-init just like on the master.

## Some notes:
- The scripts expect `eth2` to be the cloud network that the containers will communicate across.
- `vxlan` is required on the cloud network interface since cloud networks will filter based on MAC address. This is the workaround for the time being.
- A linux image with a recent kernel `> 13.07` is required for `vxlan`. Ubuntu 14.04 works.
- A number of the items in `config-default.sh` are overridable via environment variables.
- routes must be configured on each minion so that containers and kube-proxy are able to locate containers on another system. This is due to the network design in kubernetes and the MAC address limits on Cloud Networks. Static Routes are currently leveraged until we implement a more advanced solution.
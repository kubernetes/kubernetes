# Kubernetes on CoreOS on GCE

The goal is to develop standard set of fully automated scripts and configs that can launch a kubernetes cluster on GCE using CoreOS on the master and the minions.  This is work in progress.  Some of the features in the current version are:

* **cloud-config**: uses cloud config YAML files instead of Salt
* **etcd clustering**: uses CoreOS' public discovery service to create an etcd cluster
* **security**: secure etcd peering and kubernetes API server to client communication using self-signed certificates

### Warning
Please note that the certificates used in this setup are self-signed, which is far from the ideal solution. Please use with caution.

A major feature missing in this version is the "kube-push" functionality (that is, updating kubernetes binaries on an existing cluster).  Other features on the TODO list include logging, node and cluster monitoring, and DNS setup.

## User instructions

### Notes
The script invokes two "gcloud compute ssh" commands per VM instance -- to copy data over and to reboot the machine.  For convenient cluster setup, it is recommended that "gcloud compute ssh" works without password from the machine where the users intend to run "kube-up.sh".  Please refer to ["gcloud compute ssh](https://cloud.google.com/sdk/gcloud/reference/compute/ssh) and ["Connecting to an instance using ssh"](https://cloud.google.com/compute/docs/instances#sshing) for more details.

### Commands
Please refer to [kubernetes docs](docs/getting-started-guides/binary_release.md) for instructions on checking out and building from sources.  It is assumed that the user can run a kubernetes cluster on GCE using "containervm" image for the VMs.

```bash
export KUBERNETES_PROVIDER="gce"
export KUBE_GCE_IMAGE="coreos-stable-557-2-0-v20150210"
export KUBE_GCE_IMAGE_PROJECT=coreos-cloud
cluster/kube-up.sh
```

Optionally, you can set the following environment variables to further control the cluster setup:
```bash
export KUBE_GCE_INSTANCE_PREFIX="$USER-k8s"
export KUBE_GCE_NETWORK="$USER-k8s-network"
export KUBE_GCE_ZONE="us-central1-f"
export NUM_MINIONS=2
```

For a list of available CoreOS images:
```bash
gcloud compute images list --project coreos-cloud
```

For basic sanity-checking of the cluster:
```bash
cluster/kubectl.sh get minions
cluster/kubectl.sh get events
```

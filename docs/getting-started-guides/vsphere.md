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
[here](http://releases.k8s.io/release-1.0/docs/getting-started-guides/vsphere.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
Getting started with vSphere
-------------------------------

The example below creates a Kubernetes cluster with 4 worker node Virtual
Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This
cluster is set up and controlled from your workstation (or wherever you find
convenient).

**Table of Contents**

- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Starting a cluster](#starting-a-cluster)
- [Extra: debugging deployment failure](#extra-debugging-deployment-failure)

### Prerequisites

1. You need administrator credentials to an ESXi machine or vCenter instance.
2. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
3. You must have your `GOPATH` set up and include `$GOPATH/bin` in your `PATH`.

   ```sh
   export GOPATH=$HOME/src/go
   mkdir -p $GOPATH
   export PATH=$PATH:$GOPATH/bin
   ```

4. Install the govc tool to interact with ESXi/vCenter:

   ```sh
   go get github.com/vmware/govmomi/govc
   ```

5. Get or build a [binary release](binary_release.md)

### Setup

Download a prebuilt Debian 7.7 VMDK that we'll use as a base image:

```sh
curl --remote-name-all https://storage.googleapis.com/govmomi/vmdk/2014-11-11/kube.vmdk.gz{,.md5}
md5sum -c kube.vmdk.gz.md5
gzip -d kube.vmdk.gz
```

Import this VMDK into your vSphere datastore:

```sh
export GOVC_URL='user:pass@hostname'
export GOVC_INSECURE=1 # If the host above uses a self-signed cert
export GOVC_DATASTORE='target datastore'
export GOVC_RESOURCE_POOL='resource pool or cluster with access to datastore'

govc import.vmdk kube.vmdk ./kube/
```

Verify that the VMDK was correctly uploaded and expanded to ~3GiB:

```sh
govc datastore.ls ./kube/
```

Take a look at the file `cluster/vsphere/config-common.sh` fill in the required
parameters. The guest login for the image that you imported is `kube:kube`.

### Starting a cluster

Now, let's continue with deploying Kubernetes.
This process takes about ~10 minutes.

```sh
cd kubernetes # Extracted binary release OR repository root
export KUBERNETES_PROVIDER=vsphere
cluster/kube-up.sh
```

Refer to the top level README and the getting started guide for Google Compute
Engine. Once you have successfully reached this point, your vSphere Kubernetes
deployment works just as any other one!

**Enjoy!**

### Extra: debugging deployment failure

The output of `kube-up.sh` displays the IP addresses of the VMs it deploys. You
can log into any VM as the `kube` user to poke around and figure out what is
going on (find yourself authorized with your SSH key, or use the password
`kube` otherwise).


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/vsphere.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

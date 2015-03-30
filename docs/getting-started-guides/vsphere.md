## Getting started with vSphere

The example below creates a LMKTFY cluster with 4 worker node Virtual
Machines and a master Virtual Machine (i.e. 5 VMs in your cluster). This
cluster is set up and controlled from your workstation (or wherever you find
convenient).

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
curl --remote-name-all https://storage.googleapis.com/govmomi/vmdk/2014-11-11/lmktfy.vmdk.gz{,.md5}
md5sum -c lmktfy.vmdk.gz.md5
gzip -d lmktfy.vmdk.gz
```

Import this VMDK into your vSphere datastore:

```sh
export GOVC_URL='user:pass@hostname'
export GOVC_INSECURE=1 # If the host above uses a self-signed cert
export GOVC_DATASTORE='target datastore'
export GOVC_RESOURCE_POOL='resource pool or cluster with access to datastore'

govc import.vmdk lmktfy.vmdk ./lmktfy/
```

Verify that the VMDK was correctly uploaded and expanded to ~3GiB:

```sh
govc datastore.ls ./lmktfy/
```

Take a look at the file `cluster/vsphere/config-common.sh` fill in the required
parameters. The guest login for the image that you imported is `lmktfy:lmktfy`.

### Starting a cluster

Now, let's continue with deploying LMKTFY.
This process takes about ~10 minutes.

```sh
cd lmktfy # Extracted binary release OR repository root
export LMKTFYRNETES_PROVIDER=vsphere
cluster/lmktfy-up.sh
```

Refer to the top level README and the getting started guide for Google Compute
Engine. Once you have successfully reached this point, your vSphere LMKTFY
deployment works just as any other one!

**Enjoy!**

### Extra: debugging deployment failure

The output of `lmktfy-up.sh` displays the IP addresses of the VMs it deploys. You
can log into any VM as the `lmktfy` user to poke around and figure out what is
going on (find yourself authorized with your SSH key, or use the password
`lmktfy` otherwise).

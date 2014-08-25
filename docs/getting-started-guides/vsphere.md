## Getting started with vSphere

### Prerequisites

1. You need administrator credentials to an ESXi machine or vCenter instance.
2. You must have Go (version 1.2 or later) installed: [www.golang.org](http://www.golang.org).
3. Install the govc tool to interact with ESXi/vCenter:

   ```sh
   go get github.com/vmware/govmomi/govc
   ```
4. Install godep:

   ```sh
   export GOBIN=/usr/local/go/bin
   go get github.com/tools/godep
   ```

5. Get the Kubernetes source:

   ```sh
   git clone https://github.com/GoogleCloudPlatform/kubernetes.git
   ```

### Setup

Download a prebuilt Debian VMDK to be used as base image:

```sh
wget http://storage.googleapis.com/govmomi/vmdk/kube.vmdk.gz{,.md5}
md5sum -c kube.vmdk.gz.md5
gzip -d kube.vmdk.gz
```

Upload this VMDK to your vSphere instance:

```sh
export GOVC_URL='https://user:pass@hostname/sdk'
export GOVC_DATASTORE='target datastore'
export GOVC_RESOURCE_POOL='resource pool with access to datastore'

govc datastore.import kube.vmdk
```

Verify that the VMDK was correctly uploaded and expanded to 10GiB:

```sh
govc datastore.ls
```

Take a look at the file `cluster/vsphere/config-common.sh` fill in the required
parameters. The guest login for the image that you imported is `kube:kube`.

Now, let's continue with deploying Kubernetes:

```
cd kubernetes

# Build a release
release/build-release.sh

# Deploy Kubernetes (takes ~5 minutes, provided everything works out)
export KUBERNETES_PROVIDER=vsphere
cluster/kube-up.sh
```

Refer to the top level README and the getting started guide for Google Compute
Engine. Once you have successfully reached this point, your vSphere Kubernetes
deployment works just as any other one!

**Enjoy!**

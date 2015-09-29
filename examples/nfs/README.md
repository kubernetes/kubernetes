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
[here](http://releases.k8s.io/release-1.0/examples/nfs/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Example of NFS volume

See [nfs-web-pod.yaml](nfs-web-pod.yaml) for a quick example, how to use NFS volume
in a pod.

## Complete setup

The example below shows how to export a NFS share from a pod and import it
into another one.

### Prerequisites

The nfs server pod creates a privileged container, so if you are using a Salt based KUBERNETES_PROVIDER (**gce**, **vagrant**, **aws**), you have to enable the ability to create privileged containers by API.

```sh
#At the root of Kubernetes source code
$ vi cluster/saltbase/pillar/privilege.sls

# If true, allow privileged containers to be created by API
allow_privileged: true
```

For other non-salt based provider, you can set `--allow-privileged=true` for both api-server and kubelet, and then restart these components.

Rebuild the Kubernetes and spin up a cluster using your preferred KUBERNETES_PROVIDER.

### NFS server part

Define [NFS server pod](nfs-server-pod.yaml) and
[NFS service](nfs-server-service.yaml):

    $ kubectl create -f nfs-server-pod.yaml
    $ kubectl create -f nfs-server-service.yaml

The server exports `/mnt/data` directory as `/` (fsid=0). The directory contains
dummy `index.html`. Wait until the pod is running!

### NFS client

[WEB server pod](nfs-web-pod.yaml) uses the NFS share exported above as a NFS
volume and runs simple web server on it. The pod assumes your DNS is configured
and the NFS service is reachable as `nfs-server.default.kube.local`. Edit the
yaml file to supply another name or directly its IP address (use
`kubectl get services` to get it).

Define the pod:

    $ kubectl create -f nfs-web-pod.yaml

Now the pod serves `index.html` from the NFS server:

    $ curl http://<the container IP address>/
    Hello World!


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

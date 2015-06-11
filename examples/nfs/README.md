# Example of NFS volume

See [nfs-web-pod.yaml](nfs-web-pod.yaml) for a quick example, how to use NFS volume
in a pod.

## Complete setup

The example below shows how to export a NFS share from a pod and import it
into another one.

###Prerequisites
The nfs server pod creates a privileged container, so if you are using a Salt based KUBERNETES_PROVIDER (**gce**, **vagrant**, **aws**), you have to enable the ability to create privileged containers by API.

```shell
#At the root of Kubernetes source code
$ vi cluster/saltbase/pillar/privilege.sls

# If true, allow privileged containers to be created by API
allow_privileged: true
```

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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/nfs/README.md?pixel)]()

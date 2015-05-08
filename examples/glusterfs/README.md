## Glusterfs

[Glusterfs](http://www.gluster.org) is an open source scale-out filesystem. These examples provide information about how to allow containers use Glusterfs volumes.

The example assumes that the Glusterfs client package is installed on all nodes.

### Prerequisites

Install Glusterfs client package on the Kubernetes hosts.

### Create a POD

The following *volume* spec illustrates a sample configuration.

```js
{
     "name": "glusterfsvol",
     "glusterfs": {
        "endpoints": "glusterfs-cluster",
        "path": "kube_vol",
        "readOnly": true
    }
}
```

The parameters are explained as the followings. 

- **endpoints** is endpoints name that represents a Gluster cluster configuration. *kubelet* is optimized to avoid mount storm, it will randomly pick one from the endpoints to mount. If this host is unresponsive, the next Gluster host in the endpoints is automatically selected. 
- **path** is the Glusterfs volume name. 
- **readOnly** is the boolean that sets the mountpoint readOnly or readWrite. 

Detailed POD and Gluster cluster endpoints examples can be found at [v1beta3/](v1beta3/) and [endpoints/](endpoints/)

```shell
# create gluster cluster endpoints
$ kubectl create -f examples/glusterfs/endpoints/glusterfs-endpoints.json
# create a container using gluster volume
$ kubectl create -f examples/glusterfs/v1beta3/glusterfs.json
```
Once that's up you can list the pods and endpoint in the cluster, to verify that the master is running:

```shell
$ kubectl get endpoints
$ kubectl get pods
```

If you ssh to that machine, you can run `docker ps` to see the actual pod and `mount` to see if the Glusterfs volume is mounted.

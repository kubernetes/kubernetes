## GlusterFS

[GlusterFS](http://www.gluster.org) is an open source scale-out filesystem. These examples provide information about how to allow containers use GlusterFS volumes.

There are couple of ways to use GlusterFS as a persistent data store in application pods.

*) Static Provisioning of GlusterFS Volumes.
*) Dynamic Provisioning of GlusterFS Volumes.

### Static Provisioning

Static Provisioning of GlusterFS Volumes is analogues to creation of a PV ( Persistent Volume) resource by specifying the parameters in it. This
also need a working GlusterFS cluster/trusted pool available to carve out GlusterFS volumes.

The example assumes that you have already set up a GlusterFS server cluster and have a working GlusterFS volume ready to use in the containers.

#### Prerequisites

* Set up a GlusterFS server cluster
* Create a GlusterFS volume
* If you are not using hyperkube, you may need to install the GlusterFS client package on the Kubernetes nodes ([Guide](http://gluster.readthedocs.io/en/latest/Administrator%20Guide/))

#### Create endpoints

The first step is to create the GlusterFS endpoints definition in Kubernetes. Here is a snippet of [glusterfs-endpoints.json](glusterfs-endpoints.json):

```
  "subsets": [
    {
      "addresses": [{ "ip": "10.240.106.152" }],
      "ports": [{ "port": 1 }]
    },
    {
      "addresses": [{ "ip": "10.240.79.157" }],
      "ports": [{ "port": 1 }]
    }
  ]
```

The `subsets` field should be populated with the addresses of the nodes in the GlusterFS cluster. It is fine to provide any valid value (from 1 to 65535) in the `port` field.

Create the endpoints:

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-endpoints.json
```

You can verify that the endpoints are successfully created by running

```sh
$ kubectl get endpoints
NAME                ENDPOINTS
glusterfs-cluster   10.240.106.152:1,10.240.79.157:1
```

We also need to create a service for these endpoints, so that they will persist. We will add this service without a selector to tell Kubernetes we want to add its endpoints manually. You can see [glusterfs-service.json](glusterfs-service.json) for details.

Use this command to create the service:

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-service.json
```


#### Create a Pod

The following *volume* spec in [glusterfs-pod.json](glusterfs-pod.json) illustrates a sample configuration:

```json
"volumes": [
  {
    "name": "glusterfsvol",
    "glusterfs": {
      "endpoints": "glusterfs-cluster",
      "path": "kube_vol",
      "readOnly": true
    }
  }
]
```

The parameters are explained as the followings.

- **endpoints** is the name of the Endpoints object that represents a Gluster cluster configuration. *kubelet* is optimized to avoid mount storm, it will randomly pick one from the endpoints to mount. If this host is unresponsive, the next Gluster host in the endpoints is automatically selected.
- **path** is the Glusterfs volume name.
- **readOnly** is the boolean that sets the mountpoint readOnly or readWrite.

Create a pod that has a container using Glusterfs volume,

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-pod.json
```

You can verify that the pod is running:

```sh
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
glusterfs        1/1       Running   0          3m
```

You may execute the command `mount` inside the container to see if the GlusterFS volume is mounted correctly:

```sh
$ kubectl exec glusterfs -- mount | grep gluster
10.240.106.152:kube_vol on /mnt/glusterfs type fuse.glusterfs (rw,relatime,user_id=0,group_id=0,default_permissions,allow_other,max_read=131072)
```

You may also run `docker ps` on the host to see the actual container.

### Dynamic Provisioning of GlusterFS Volumes:

Dynamic Provisioning means provisioning of GlusterFS volumes based on a Storage class. Please refer [this guide](./../../persistent-volume-provisioning/README.md)
.
<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/glusterfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

## Glusterfs

[Glusterfs](http://www.gluster.org) is an open source scale-out filesystem. These examples provide information about how to allow containers use Glusterfs volumes.

The example assumes that you have already set up a Glusterfs server cluster and the Glusterfs client package is installed on all Kubernetes nodes.

### Prerequisites

Set up Glusterfs server cluster; install Glusterfs client package on the Kubernetes nodes. ([Guide](https://www.howtoforge.com/high-availability-storage-with-glusterfs-3.2.x-on-debian-wheezy-automatic-file-replication-mirror-across-two-storage-servers))

### Create endpoints

Here is a snippet of [glusterfs-endpoints.json](glusterfs-endpoints.json),

```
      "addresses": [
        {
          "IP": "10.240.106.152"
        }
      ],
      "ports": [
        {
          "port": 1
        }
      ]

```

The "IP" field should be filled with the address of a node in the Glusterfs server cluster. In this example, it is fine to give any valid value (from 1 to 65535) to the "port" field.

Create the endpoints,

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-endpoints.json
```

You can verify that the endpoints are successfully created by running

```sh
$ kubectl get endpoints
NAME                ENDPOINTS
glusterfs-cluster   10.240.106.152:1,10.240.79.157:1
```

We need also create a service for this endpoints, so that the endpoints will be persistented. We will add this service without a selector to tell Kubernetes we want to add its endpoints manually. You can see [glusterfs-service.json](glusterfs-service.json) for details.

Use this command to create the service:

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-service.json
```


### Create a POD

The following *volume* spec in [glusterfs-pod.json](glusterfs-pod.json) illustrates a sample configuration.

```json
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

Create a pod that has a container using Glusterfs volume,

```sh
$ kubectl create -f examples/volumes/glusterfs/glusterfs-pod.json
```

You can verify that the pod is running:

```sh
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
glusterfs        1/1       Running   0          3m

$ kubectl get pods glusterfs -t '{{.status.hostIP}}{{"\n"}}'
10.240.169.172
```

You may ssh to the host (the hostIP) and run 'mount' to see if the Glusterfs volume is mounted,

```sh
$ mount | grep kube_vol
10.240.106.152:kube_vol on /var/lib/kubelet/pods/f164a571-fa68-11e4-ad5c-42010af019b7/volumes/kubernetes.io~glusterfs/glusterfsvol type fuse.glusterfs (rw,relatime,user_id=0,group_id=0,default_permissions,allow_other,max_read=131072)
```

You may also run `docker ps` on the host to see the actual container.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/glusterfs/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

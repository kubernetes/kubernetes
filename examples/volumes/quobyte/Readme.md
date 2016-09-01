<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Quobyte Volume](#quobyte-volume)
  - [Quobyte](#quobyte)
    - [Prerequisites](#prerequisites)
    - [Fixed user Mounts](#fixed-user-mounts)
  - [Creating a pod](#creating-a-pod)

<!-- END MUNGE: GENERATED_TOC -->

# Quobyte Volume

## Quobyte

[Quobyte](http://www.quobyte.com) is software that turns commodity servers into a reliable and highly automated multi-data center file system.

The example assumes that you already have a running Kubernetes cluster and you already have setup Quobyte-Client (1.3+) on each Kubernetes node.

### Prerequisites

- Running Quobyte storage cluster
- Quobyte client (1.3+) installed on the Kubernetes nodes more information how you can install Quobyte on your Kubernetes nodes, can be found in the [documentation](https://support.quobyte.com) of Quobyte.
- To get access to Quobyte and the documentation please [contact us](http://www.quobyte.com/get-quobyte)
- Already created Quobyte Volume
- Added the line `allow-usermapping-in-volumename` in `/etc/quobyte/client.cfg` to allow the fixed user mounts

### Fixed user Mounts

Quobyte supports since 1.3 fixed user mounts. The fixed-user mounts simply allow to mount all Quobyte Volumes inside one directory and use them as different users. All access to the Quobyte Volume will be rewritten to the specified user and group – both are optional, independent of the user inside the container. You can read more about it [here](https://blog.inovex.de/docker-plugins) under the section "Quobyte Mount and Docker — what’s special"

## Creating a pod

See example:

<!-- BEGIN MUNGE: EXAMPLE ./quobyte-pod.yaml -->

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: quobyte
spec:
  containers:
  - name: quobyte
    image: kubernetes/pause
    volumeMounts:
    - mountPath: /mnt
      name: quobytevolume
  volumes:
  - name: quobytevolume
    quobyte:
      registry: registry:7861
      volume: testVolume
      readOnly: false
      user: root
      group: root
```

[Download example](quobyte-pod.yaml?raw=true)
<!-- END MUNGE: EXAMPLE ./quobyte-pod.yaml -->

Parameters:
* **registry** Quobyte registry to use to mount the volume. You can specifiy the registry as <host>:<port> pair or if you want to specify multiple registries you just have to put a semicolon between them e.q. <host1>:<port>,<host2>:<port>,<host3>:<port>. The host can be an IP address or if you have a working DNS you can also provide the DNS names.
* **volume** volume represents a Quobyte volume which must be created before usage.
* **readOnly** is the boolean that sets the mountpoint readOnly or readWrite.
* **user** maps all access to this user. Default is root.
* **group** maps all access to this group. Default is empty.

Creating the pod:

```bash
$ kubectl create -f examples/volumes/quobyte/quobyte-pod.yaml
```

Verify that the pod is running:

```bash
$ kubectl get pods quobyte
NAME      READY     STATUS    RESTARTS   AGE
quobyte   1/1       Running   0          48m

$ kubectl get pods quobyte --template '{{.status.hostIP}}{{"\n"}}'
10.245.1.3
```

SSH onto the Machine and validate that quobyte is mounted:

```bash
$ mount | grep quobyte
quobyte@10.239.10.21:7861/ on /var/lib/kubelet/plugins/kubernetes.io~quobyte type fuse (rw,nosuid,nodev,noatime,user_id=0,group_id=0,default_permissions,allow_other)

$ docker inspect --format '{{ range .Mounts }}{{ if eq .Destination "/mnt"}}{{ .Source }}{{ end }}{{ end }}' 55ab97593cd3
/var/lib/kubelet/plugins/kubernetes.io~quobyte/root#root@testVolume
```



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/volumes/quobyte/Readme.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

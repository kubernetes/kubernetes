<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<h1>PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)
![WARNING](http://kubernetes.io/img/warning.png)

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes 101 - Kubectl CLI & Pods

For Kubernetes 101, we will cover kubectl, pods, volumes, and multiple containers

In order for the kubectl usage examples to work, make sure you have an examples directory locally, either from [a release](https://github.com/GoogleCloudPlatform/kubernetes/releases) or [the source](https://github.com/GoogleCloudPlatform/kubernetes).

Table of Contents
- [Kubectl CLI](#kubectl-cli)
  - [Install Kubectl](#install-kubectl)
  - [Configure Kubectl](#configure-kubectl)
- [Pods](#pods)
  - [Pod Definition](#pod-definition)
  - [Pod Management](#pod-management)
  - [Volumes](#volumes)
  - [Multiple Containers](#multiple-containers)
- [What's Next?](#whats-next)


## Kubectl CLI

The easiest way to interact with Kubernetes is via the command-line interface.

If you downloaded a pre-compiled release, kubectl should be under `platforms/<os>/<arch>`.

If you built from source, kubectl should be either under `_output/local/bin/<os>/<arch>` or `_output/dockerized/bin/<os>/<arch>`.

For more info about kubectl, including its usage, commands, and parameters, see the [kubectl CLI reference](../kubectl/kubectl.md).

#### Install Kubectl

The kubectl binary doesn't have to be installed to be executable, but the rest of the walkthrough will assume that it's in your PATH.

The simplest way to install is to copy or move kubectl into a dir already in PATH (like `/usr/local/bin`).

An alternate method, useful if you're building from source and want to rebuild without re-installing is to use `./cluster/kubectl.sh` instead of kubectl. That script will auto-detect the location of kubectl and proxy commands to it (ex: `./cluster/kubectl.sh cluster-info`).

#### Configure Kubectl

If you used `./cluster/kube-up.sh` to deploy your Kubernetes cluster, kubectl should already be locally configured.

By default, kubectl configuration lives at `~/.kube/config`.

If your cluster was deployed by other means (e.g. a [getting started guide](../../getting-started-guides/README.md)), you may want to configure the path to the Kubernetes apiserver in your shell environment:

```sh
export KUBERNETES_MASTER=http://<ip>:<port>/api
```

Check that kubectl is properly configured by getting the cluster state:

```sh
kubectl cluster-info
```


## Pods
In Kubernetes, a group of one or more containers is called a _pod_. Containers in a pod are deployed together, and are started, stopped, and replicated as a group.

See [pods](../../../docs/user-guide/pods.md) for more details.


#### Pod Definition

The simplest pod definition describes the deployment of a single container.  For example, an nginx web server pod might be defined as such:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: nginx
spec:
  containers:
  - name: nginx
    image: nginx
    ports:
    - containerPort: 80
```

A pod definition is a declaration of a _desired state_.  Desired state is a very important concept in the Kubernetes model.  Many things present a desired state to the system, and it is Kubernetes' responsibility to make sure that the current state matches the desired state.  For example, when you create a Pod, you declare that you want the containers in it to be running.  If the containers happen to not be running (e.g. program failure, ...), Kubernetes will continue to (re-)create them for you in order to drive them to the desired state. This process continues until the Pod is deleted.

See the [design document](../../../DESIGN.md) for more details.


#### Pod Management

Create a pod containing an nginx server ([pod-nginx.yaml](pod-nginx.yaml)):

```sh
kubectl create -f docs/user-guide/walkthrough/pod-nginx.yaml
```

List all pods:

```sh
kubectl get pods
```

On most providers, the pod IPs are not externally accessible. The easiest way to test that the pod is working is to create a busybox pod and exec commands on it remotely. See the [command execution documentation](../kubectl/kubectl_exec.md) for details.

Provided the pod IP is accessible, you should be able to access its http endpoint with curl on port 80:

```sh
curl http://$(kubectl get pod nginx -o=template -t={{.status.podIP}})
```

Delete the pod by name:

```sh
kubectl delete pod nginx
```


#### Volumes

That's great for a simple static web server, but what about persistent storage?

The container file system only lives as long as the container does. So if your app's state needs to survive relocation, reboots, and crashes, you'll need to configure some persistent storage.

For this example, we'll be creating a Redis pod, with a named volume and volume mount that defines the path to mount the volume.

1. Define a volume:

  ```yaml
    volumes:
    - name: redis-persistent-storage
      emptyDir: {}
  ```

1. Define a volume mount within a container definition:

  ```yaml
    volumeMounts:
    # name must match the volume name below
    - name: redis-persistent-storage
      # mount path within the container
      mountPath: /data/redis
  ```

Example Redis pod definition with a persistent storage volume ([pod-redis.yaml](pod-redis.yaml)):

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: redis
spec:
  containers:
  - name: redis
    image: redis
    volumeMounts:
    - name: redis-persistent-storage
      mountPath: /data/redis
  volumes:
  - name: redis-persistent-storage
    emptyDir: {}
```

Notes:
- The volume mount name is a reference to a specific empty dir volume.
- The volume mount path is the path to mount the empty dir volume within the container.

##### Volume Types

- **EmptyDir**: Creates a new directory that will persist across container failures and restarts.
- **HostPath**: Mounts an existing directory on the minion's file system (e.g. `/var/logs`).

See [volumes](../../../docs/user-guide/volumes.md) for more details.


#### Multiple Containers

_Note:
The examples below are syntactically correct, but some of the images (e.g. kubernetes/git-monitor) don't exist yet.  We're working on turning these into working examples._


However, often you want to have two different containers that work together.  An example of this would be a web server, and a helper job that polls a git repository for new updates:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: www
spec:
  containers:
  - name: nginx
    image: nginx
    volumeMounts:
    - mountPath: /srv/www
      name: www-data
      readOnly: true
  - name: git-monitor
    image: kubernetes/git-monitor
    env:
    - name: GIT_REPO
      value: http://github.com/some/repo.git
    volumeMounts:
    - mountPath: /data
      name: www-data
  volumes:
  - name: www-data
    emptyDir: {}
```

Note that we have also added a volume here.  In this case, the volume is mounted into both containers.  It is marked `readOnly` in the web server's case, since it doesn't need to write to the directory.

Finally, we have also introduced an environment variable to the `git-monitor` container, which allows us to parameterize that container with the particular git repository that we want to track.


## What's Next?

Continue on to [Kubernetes 201](k8s201.md) or
for a complete application see the [guestbook example](../../../examples/guestbook/README.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/walkthrough/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

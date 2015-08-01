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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/walkthrough/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes 101 - Kubectl CLI and Pods

For Kubernetes 101, we will cover kubectl, pods, volumes, and multiple containers

In order for the kubectl usage examples to work, make sure you have an examples directory locally, either from [a release](https://github.com/GoogleCloudPlatform/kubernetes/releases) or [the source](https://github.com/GoogleCloudPlatform/kubernetes).

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes 101 - Kubectl CLI and Pods](#kubernetes-101---kubectl-cli-and-pods)
  - [Kubectl CLI](#kubectl-cli)
  - [Pods](#pods)
      - [Pod Definition](#pod-definition)
      - [Pod Management](#pod-management)
      - [Volumes](#volumes)
        - [Volume Types](#volume-types)
      - [Multiple Containers](#multiple-containers)
  - [What's Next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

## Kubectl CLI

The easiest way to interact with Kubernetes is via the [kubectl](../kubectl/kubectl.md) command-line interface.

For more info about kubectl, including its usage, commands, and parameters, see the [kubectl CLI reference](../kubectl/kubectl.md).

If you haven't installed and configured kubectl, finish the [prerequisites](../prereqs.md) before continuing.

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
$ kubectl create -f docs/user-guide/walkthrough/pod-nginx.yaml
```

List all pods:

```sh
$ kubectl get pods
```

On most providers, the pod IPs are not externally accessible. The easiest way to test that the pod is working is to create a busybox pod and exec commands on it remotely. See the [command execution documentation](../kubectl/kubectl_exec.md) for details.

Provided the pod IP is accessible, you should be able to access its http endpoint with curl on port 80:

```sh
$ curl http://$(kubectl get pod nginx -o=template -t={{.status.podIP}})
```

Delete the pod by name:

```sh
$ kubectl delete pod nginx
```


#### Volumes

That's great for a simple static web server, but what about persistent storage?

The container file system only lives as long as the container does. So if your app's state needs to survive relocation, reboots, and crashes, you'll need to configure some persistent storage.

For this example we'll be creating a Redis pod with a named volume and volume mount that defines the path to mount the volume.

1. Define a volume:

  ```yaml
    volumes:
    - name: redis-persistent-storage
      emptyDir: {}
  ```

2. Define a volume mount within a container definition:

  ```yaml
    volumeMounts:
    # name must match the volume name below
    - name: redis-persistent-storage
      # mount path within the container
      mountPath: /data/redis
  ```

Example Redis pod definition with a persistent storage volume ([pod-redis.yaml](pod-redis.yaml)):

<!-- BEGIN MUNGE: EXAMPLE pod-redis.yaml -->

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

[Download example](pod-redis.yaml)
<!-- END MUNGE: EXAMPLE pod-redis.yaml -->

Notes:
- The volume mount name is a reference to a specific empty dir volume.
- The volume mount path is the path to mount the empty dir volume within the container.

##### Volume Types

- **EmptyDir**: Creates a new directory that will persist across container failures and restarts.
- **HostPath**: Mounts an existing directory on the node's file system (e.g. `/var/logs`).

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

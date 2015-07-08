# Kubernetes 101 - Walkthrough

## Pods
The first atom of Kubernetes is a _pod_.  A pod is a collection of containers that are symbiotically grouped.

See [pods](../../docs/pods.md) for more details.

### Intro

Trivially, a single container might be a pod.  For example, you can express a simple web server as a pod:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: www
spec:
  containers:
    - name: nginx
      image: nginx
```

A pod definition is a declaration of a _desired state_.  Desired state is a very important concept in the Kubernetes model.  Many things present a desired state to the system, and it is Kubernetes' responsibility to make sure that the current state matches the desired state.  For example, when you create a Pod, you declare that you want the containers in it to be running.  If the containers happen to not be running (e.g. program failure, ...), Kubernetes will continue to (re-)create them for you in order to drive them to the desired state. This process continues until you delete the Pod.

See the [design document](../../DESIGN.md) for more details.

### Volumes

Now that's great for a static web server, but what about persistent storage?  We know that the container file system only lives as long as the container does, so we need more persistent storage.  To do this, you also declare a ```volume``` as part of your pod, and mount it into a container:
```yaml
apiVersion: v1
kind: Pod
metadata:
  name: storage
spec:
  containers:
    - name: redis
      image: redis
      volumeMounts:
          # name must match the volume name below
        - name: redis-persistent-storage
          # mount path within the container
          mountPath: /data/redis
  volumes:
    - name: redis-persistent-storage
      emptyDir: {}
```

Ok, so what did we do? We added a volume to our pod:
```
  volumes:
    - name: redis-persistent-storage
      emptyDir: {}
```

And we added a reference to that volume to our container:
```
      volumeMounts:
          # name must match the volume name below
        - name: redis-persistent-storage
          # mount path within the container
          mountPath: /data/redis
```

In Kubernetes, ```emptyDir``` Volumes live for the lifespan of the Pod, which is longer than the lifespan of any one container, so if the container fails and is restarted, our persistent storage will live on.

If you want to mount a directory that already exists in the file system (e.g. ```/var/logs```) you can use the ```hostPath``` directive.

See [volumes](../../docs/volumes.md) for more details.

### Multiple Containers

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

Note that we have also added a volume here.  In this case, the volume is mounted into both containers.  It is marked ```readOnly``` in the web server's case, since it doesn't need to write to the directory.

Finally, we have also introduced an environment variable to the ```git-monitor``` container, which allows us to parameterize that container with the particular git repository that we want to track.


### What's next?
Continue on to [Kubernetes 201](../walkthrough/k8s201.md) or
for a complete application see the [guestbook example](../guestbook/README.md)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/walkthrough/README.md?pixel)]()

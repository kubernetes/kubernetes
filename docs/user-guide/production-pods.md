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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/production-pods.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Working with pods and containers in production

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications: Working with pods and containers in production](#kubernetes-user-guide-managing-applications-working-with-pods-and-containers-in-production)
  - [Persistent storage](#persistent-storage)
  - [Distributing credentials](#distributing-credentials)
  - [Authenticating with a private image registry](#authenticating-with-a-private-image-registry)
  - [Helper containers](#helper-containers)
  - [Resource management](#resource-management)
  - [Liveness and readiness probes (aka health checks)](#liveness-and-readiness-probes-aka-health-checks)
  - [Lifecycle hooks and termination notice](#lifecycle-hooks-and-termination-notice)
  - [Termination message](#termination-message)
  - [What's next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

You’ve seen [how to configure and deploy pods and containers](configuring-containers.md), using some of the most common configuration parameters. This section dives into additional features that are especially useful for running applications in production.

## Persistent storage

The container file system only lives as long as the container does, so when a container crashes and restarts, changes to the filesystem will be lost and the container will restart from a clean slate. To access more-persistent storage, outside the container file system, you need a [*volume*](volumes.md). This is especially important to stateful applications, such as key-value stores and databases.

For example, [Redis](http://redis.io/) is a key-value cache and store, which we use in the [guestbook](../../examples/guestbook/) and other examples. We can add a volume to it to store persistent data as follows:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis
spec:
  template:
    metadata:
      labels:
        app: redis
        tier: backend
    spec:
      # Provision a fresh volume for the pod
      volumes:
        - name: data
          emptyDir: {}
      containers:
      - name: redis
        image: kubernetes/redis:v1
        ports:
        - containerPort: 6379
        # Mount the volume into the pod
        volumeMounts:
        - mountPath: /redis-master-data
          name: data   # must match the name of the volume, above
```

`emptyDir` volumes live for the lifespan of the [pod](pods.md), which is longer than the lifespan of any one container, so if the container fails and is restarted, our storage will live on.

In addition to the local disk storage provided by `emptyDir`, Kubernetes supports many different network-attached storage solutions, including PD on GCE and EBS on EC2, which are preferred for critical data, and will handle details such as mounting and unmounting the devices on the nodes. See [the volumes doc](volumes.md) for more details.

## Distributing credentials

Many applications need credentials, such as passwords, OAuth tokens, and TLS keys, to authenticate with other applications, databases, and services. Storing these credentials in container images or environment variables is less than ideal, since the credentials can then be copied by anyone with access to the image, pod/container specification, host file system, or host Docker daemon.

Kubernetes provides a mechanism, called [*secrets*](secrets.md), that facilitates delivery of sensitive credentials to applications. A `Secret` is a simple resource containing a map of data. For instance, a simple secret with a username and password might look as follows:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: mysecret
type: Opaque
data:
  password: dmFsdWUtMg0K
  username: dmFsdWUtMQ0K
```

As with other resources, this secret can be instantiated using `create` and can be viewed with `get`:

```console
$ kubectl create -f ./secret.yaml
secrets/mysecret
$ kubectl get secrets
NAME                  TYPE                                  DATA
default-token-v9pyz   kubernetes.io/service-account-token   2
mysecret              Opaque                                2
```

To use the secret, you need to reference it in a pod or pod template. The `secret` volume source enables you to mount it as an in-memory directory into your containers.

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: redis
spec:
  template:
    metadata:
      labels:
        app: redis
        tier: backend
    spec:
      volumes:
        - name: data
          emptyDir: {}
        - name: supersecret
          secret:
            secretName: mysecret
      containers:
      - name: redis
        image: kubernetes/redis:v1
        ports:
        - containerPort: 6379
        # Mount the volume into the pod
        volumeMounts:
        - mountPath: /redis-master-data
          name: data   # must match the name of the volume, above
        - mountPath: /var/run/secrets/super
          name: supersecret
```

For more details, see the [secrets document](secrets.md), [example](secrets/) and [design doc](../../docs/design/secrets.md).

## Authenticating with a private image registry

Secrets can also be used to pass [image registry credentials](images.md#using-a-private-registry).

First, create a `.dockercfg` file, such as running `docker login <registry.domain>`.
Then put the resulting `.dockercfg` file into a [secret resource](secrets.md).  For example:

```console
$ docker login
Username: janedoe
Password: ●●●●●●●●●●●
Email: jdoe@example.com
WARNING: login credentials saved in /Users/jdoe/.dockercfg.
Login Succeeded

$ echo $(cat ~/.dockercfg)
{ "https://index.docker.io/v1/": { "auth": "ZmFrZXBhc3N3b3JkMTIK", "email": "jdoe@example.com" } }

$ cat ~/.dockercfg | base64
eyAiaHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEvIjogeyAiYXV0aCI6ICJabUZyWlhCaGMzTjNiM0prTVRJSyIsICJlbWFpbCI6ICJqZG9lQGV4YW1wbGUuY29tIiB9IH0K

$ cat > /tmp/image-pull-secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: myregistrykey
data:
  .dockercfg: eyAiaHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEvIjogeyAiYXV0aCI6ICJabUZyWlhCaGMzTjNiM0prTVRJSyIsICJlbWFpbCI6ICJqZG9lQGV4YW1wbGUuY29tIiB9IH0K
type: kubernetes.io/dockercfg
EOF

$ kubectl create -f ./image-pull-secret.yaml
secrets/myregistrykey
```

Now, you can create pods which reference that secret by adding an `imagePullSecrets`
section to a pod definition.

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: foo
spec:
  containers:
    - name: foo
      image: janedoe/awesomeapp:v1
  imagePullSecrets:
    - name: myregistrykey
```

## Helper containers

[Pods](pods.md) support running multiple containers co-located together. They can be used to host vertically integrated application stacks, but their primary motivation is to support auxiliary helper programs that assist the primary application. Typical examples are data pullers, data pushers, and proxies.

Such containers typically need to communicate with one another, often through the file system. This can be achieved by mounting the same volume into both containers. An example of this pattern would be a web server with a [program that polls a git repository](http://releases.k8s.io/HEAD/contrib/git-sync/) for new updates:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  template:
    metadata:
      labels:
        app: nginx
    spec:
      volumes:
      - name: www-data
        emptyDir: {}
      containers:
      - name: nginx
        image: nginx
        # This container reads from the www-data volume
        volumeMounts:
        - mountPath: /srv/www
          name: www-data
          readOnly: true
      - name: git-monitor
        image: myrepo/git-monitor
        env:
        - name: GIT_REPO
          value: http://github.com/some/repo.git
        # This container writes to the www-data volume
        volumeMounts:
        - mountPath: /data
          name: www-data
```

More examples can be found in our [blog article](http://blog.kubernetes.io/2015/06/the-distributed-system-toolkit-patterns.html) and [presentation slides](http://www.slideshare.net/Docker/slideshare-burns).

## Resource management

Kubernetes’s scheduler will place applications only where they have adequate CPU and memory, but it can only do so if it knows how much [resources they require](compute-resources.md). The consequence of specifying too little CPU is that the containers could be starved of CPU if too many other containers were scheduled onto the same node. Similarly, containers could die unpredictably due to running out of memory if no memory were requested, which can be especially likely for large-memory applications.

If no resource requirements are specified, a nominal amount of resources is assumed. (This default is applied via a [LimitRange](../admin/limitrange/) for the default [Namespace](namespaces.md). It can be viewed with `kubectl describe limitrange limits`.) You may explicitly specify the amount of resources required as follows:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
        resources:
          limits:
            # cpu units are cores
            cpu: 500m
            # memory units are bytes
            memory: 64Mi
		  requests:
			# cpu units are cores
		    cpu: 500m
			# memory units are bytes
			memory: 64Mi
```

The container will die due to OOM (out of memory) if it exceeds its specified limit, so specifying a value a little higher than expected generally improves reliability. By specifying request, pod is guaranteed to be able to use that much of resource when needed. See [Resource QoS](../proposals/resource-qos.md) for the difference between resource limits and requests.

If you’re not sure how much resources to request, you can first launch the application without specifying resources, and use [resource usage monitoring](monitoring.md) to determine appropriate values.

## Liveness and readiness probes (aka health checks)

Many applications running for long periods of time eventually transition to broken states, and cannot recover except by restarting them. Kubernetes provides [*liveness probes*](pod-states.md#container-probes) to detect and remedy such situations.

A common way to probe an application is using HTTP, which can be specified as follows:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
        livenessProbe:
          httpGet:
            # Path to probe; should be cheap, but representative of typical behavior
            path: /index.html
            port: 80
          initialDelaySeconds: 30
          timeoutSeconds: 1
```

Other times, applications are only temporarily unable to serve, and will recover on their own. Typically in such cases you’d prefer not to kill the application, but don’t want to send it requests, either, since the application won’t respond correctly or at all. A common such scenario is loading large data or configuration files during application startup. Kubernetes provides *readiness probes* to detect and mitigate such situations. Readiness probes are configured similarly to liveness probes, just using the `readinessProbe` field. A pod with containers reporting that they are not ready will not receive traffic through Kubernetes [services](connecting-applications.md).

For more details (e.g., how to specify command-based probes), see the [example in the walkthrough](walkthrough/k8s201.md#health-checking), the [standalone example](liveness/), and the [documentation](pod-states.md#container-probes).

## Lifecycle hooks and termination notice

Of course, nodes and applications may fail at any time, but many applications benefit from clean shutdown, such as to complete in-flight requests, when the termination of the application is deliberate. To support such cases, Kubernetes supports two kinds of notifications:

* Kubernetes will send SIGTERM to applications, which can be handled in order to effect graceful termination. SIGKILL is sent a configurable number of seconds later if the application does not terminate sooner (defaults to 30 seconds, controlled by `spec.terminationGracePeriodSeconds`).
* Kubernetes supports the (optional) specification of a [*pre-stop lifecycle hook*](container-environment.md#container-hooks), which will execute prior to sending SIGTERM.

The specification of a pre-stop hook is similar to that of probes, but without the timing-related parameters. For example:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 2
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        ports:
        - containerPort: 80
        lifecycle:
          preStop:
            exec:
              # SIGTERM triggers a quick exit; gracefully terminate instead
              command: ["/usr/sbin/nginx","-s","quit"]
```

## Termination message

In order to achieve a reasonably high level of availability, especially for actively developed applications, it’s important to debug failures quickly. Kubernetes can speed debugging by surfacing causes of fatal errors in a way that can be display using [`kubectl`](kubectl/kubectl.md) or the [UI](ui.md), in addition to general [log collection](logging.md). It is possible to specify a `terminationMessagePath` where a container will write its “death rattle”, such as assertion failure messages, stack traces, exceptions, and so on. The default path is `/dev/termination-log`.

Here is a toy example:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: pod-w-message
spec:
  containers:
  - name: messager
    image: "ubuntu:14.04"
    command: ["/bin/sh","-c"]
    args: ["sleep 60 && /bin/echo Sleep expired > /dev/termination-log"]
```

The message is recorded along with the other state of the last (i.e., most recent) termination:

```console
$ kubectl create -f ./pod.yaml
pods/pod-w-message
$ sleep 70
$ kubectl get pods/pod-w-message -o go-template="{{range .status.containerStatuses}}{{.lastState.terminated.message}}{{end}}"
Sleep expired
$ kubectl get pods/pod-w-message -o go-template="{{range .status.containerStatuses}}{{.lastState.terminated.exitCode}}{{end}}"
0
```

## What's next?

[Learn more about managing deployments.](managing-deployments.md)


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/production-pods.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

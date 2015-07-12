<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<h1>*** PLEASE NOTE: This document applies to the HEAD of the source
tree only. If you are using a released version of Kubernetes, you almost
certainly want the docs that go with that version.</h1>

<strong>Documentation for specific releases can be found at
[releases.k8s.io](http://releases.k8s.io).</strong>

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->
# Kubernetes User Guide: Managing Applications: Working with pods and containers in production

You’ve seen [how to configure and deploy pods and containers](configuring-containers.md), using some of the most common configuration parameters. This section dives into additional features that are especially useful for running applications in production.

## Persistent storage

The container file system only lives as long as the container does, so when a container crashes and restarts, changes to the filesystem will be lost and the container will restart from a clean slate. To access more-persistent storage, outside the container file system, you need a [*volume*](../../docs/volumes.md). This is especially important to stateful applications, such as key-value stores and databases.

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
`emptyDir` volumes live for the lifespan of the [pod](../../docs/pods.md), which is longer than the lifespan of any one container, so if the container fails and is restarted, our storage will live on.

In addition to the local disk storage provided by `emptyDir`, Kubernetes supports many different network-attached storage solutions, including PD on GCE and EBS on EC2, which are preferred for critical data, and will handle details such as mounting and unmounting the devices on the nodes. See [the volumes doc](../../docs/volumes.md) for more details.

## Distributing credentials

Many applications need credentials, such as passwords, OAuth tokens, and TLS keys, to authenticate with other applications, databases, and services. Storing these credentials in container images or environment variables is less than ideal, since the credentials can then be copied by anyone with access to the image, pod/container specification, host file system, or host Docker daemon. 

Kubernetes provides a mechanism, called [*secrets*](../../docs/secrets.md), that facilitates delivery of sensitive credentials to applications. A `Secret` is a simple resource containing a map of data. For instance, a simple secret with a username and password might look as follows:
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
```bash
$ kubectl create -f secret.yaml
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

For more details, see the [secrets document](../../docs/secrets.md), [example](../../examples/secrets/) and [design doc](../../docs/design/secrets.md).

## Authenticating with a private image registry

Secrets can also be used to pass [image registry credentials](../../docs/images.md#using-a-private-registry).

First, create a `.dockercfg` file, such as running `docker login <registry.domain>`.
Then put the resulting `.dockercfg` file into a [secret resource](../../docs/secrets.md).  For example:
```
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

$ cat > image-pull-secret.yaml <<EOF
apiVersion: v1
kind: Secret
metadata:
  name: myregistrykey
data:
  .dockercfg: eyAiaHR0cHM6Ly9pbmRleC5kb2NrZXIuaW8vdjEvIjogeyAiYXV0aCI6ICJabUZyWlhCaGMzTjNiM0prTVRJSyIsICJlbWFpbCI6ICJqZG9lQGV4YW1wbGUuY29tIiB9IH0K
type: kubernetes.io/dockercfg
EOF

$ kubectl create -f image-pull-secret.yaml
secrets/myregistrykey
```

Now, you can create pods which reference that secret by adding an `imagePullSecrets`
section to a pod definition.
```
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


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/production-pods.md?pixel)]()

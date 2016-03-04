<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# ConfigMap

Many applications require configuration via some combination of config files, command line
arguments, and environment variables.  These configuration artifacts should be decoupled from image
content in order to keep containerized applications portable.  The ConfigMap API resource provides
mechanisms to inject containers with configuration data while keeping containers agnostic of
Kubernetes.  ConfigMap can be used to store fine-grained information like individual properties or
coarse-grained information like entire config files or JSON blobs.

**Table of Contents**

<!-- BEGIN MUNGE: GENERATED_TOC -->

- [ConfigMap](#configmap)
  - [Overview of ConfigMap](#overview-of-configmap)
  - [Creating ConfigMaps](#creating-configmaps)
    - [Creating from directories](#creating-from-directories)
    - [Creating from files](#creating-from-files)
    - [Creating from literal values](#creating-from-literal-values)
  - [Consuming ConfigMap in pods](#consuming-configmap-in-pods)
    - [Use-Case: Consume ConfigMap in environment variables](#use-case-consume-configmap-in-environment-variables)
    - [Use-Case: Set command-line arguments with ConfigMap](#use-case-set-command-line-arguments-with-configmap)
    - [Use-Case: Consume ConfigMap via volume plugin](#use-case-consume-configmap-via-volume-plugin)
  - [Real World Example: Configuring Redis](#real-world-example-configuring-redis)
  - [Restrictions](#restrictions)

<!-- END MUNGE: GENERATED_TOC -->

## Overview of ConfigMap

The ConfigMap API resource holds key-value pairs of configuration data that can be consumed in pods
or used to store configuration data for system components such as controllers.  ConfigMap is similar
to [Secrets](secrets.md), but designed to more conveniently support working with strings that do not
contain sensitive information.

Let's look at a made-up example:

```yaml
kind: ConfigMap
apiVersion: v1
metadata:
  creationTimestamp: 2016-02-18T19:14:38Z
  name: example-config
  namespace: default
data:
  example.property.1: hello
  example.property.2: world
  example.property.file: |-
    property.1=value-1
    property.2=value-2
    property.3=value-3
```

The `data` field contains the configuration data.  As you can see, ConfigMaps can be used to hold
fine-grained information like individual properties or coarse-grained information like the contents
of configuration files.

Configuration data can be consumed in pods in a variety of ways.  ConfigMaps can be used to:

1.  Populate the value of environment variables
2.  Set command-line arguments in a container
3.  Populate config files in a volume

Both users and system components may store configuration data in ConfigMap.

## Creating ConfigMaps

You can use the `kubectl create configmap` command to create configmaps easily from literal values,
files, or directories.

Let's take a look at some different ways to create a ConfigMap:

### Creating from directories

Say that we have a directory with some files that already contain the data we want to populate a ConfigMap with:

```console

$ ls docs/user-guide/configmap/kubectl/
game.properties
ui.properties

$ cat docs/user-guide/configmap/kubectl/game.properties
enemies=aliens
lives=3
enemies.cheat=true
enemies.cheat.level=noGoodRotten
secret.code.passphrase=UUDDLRLRBABAS
secret.code.allowed=true
secret.code.lives=30

$ cat docs/user-guide/configmap/kubectl/ui.properties
color.good=purple
color.bad=yellow
allow.textmode=true
how.nice.to.look=fairlyNice

```

The `kubectl create configmap` command can be used to create a ConfigMap holding the content of each
file in this directory:

```console

$ kubectl create configmap game-config --from-file=docs/user-guide/configmap/kubectl

```

When `--from-file` points to a directory, each file directly in that directory is used to populate a
key in the ConfigMap, where the name of the key is the filename, and the value of the key is the
content of the file.

Let's take a look at the ConfigMap that this command created:

```console

$ cluster/kubectl.sh describe configmaps game-config
Name:           game-config
Namespace:      default
Labels:         <none>
Annotations:    <none>

Data
====
game.properties:        121 bytes
ui.properties:          83 bytes

```

You can see the two keys in the map are created from the filenames in the directory we pointed
kubectl to.  Since the content of those keys may be large, in the output of `kubectl describe`,
you'll see only the names of the keys and their sizes.

If we want to see the values of the keys, we can simply `kubectl get` the resource:

```console

$ kubectl get configmaps game-config -o yaml
apiVersion: v1
data:
  game.properties: |-
    enemies=aliens
    lives=3
    enemies.cheat=true
    enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
  ui.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true
    how.nice.to.look=fairlyNice
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T18:34:05Z
  name: game-config
  namespace: default
  resourceVersion: "407"-
  selfLink: /api/v1/namespaces/default/configmaps/game-config
  uid: 30944725-d66e-11e5-8cd0-68f728db1985

```

### Creating from files

We can also pass `--from-file` a specific file, and pass it multiple times to kubectl.  The
following command yields equivalent results to the above example:

```console

$ kubectl create configmap game-config-2 --from-file=docs/user-guide/configmap/kubectl/game.properties --from-file=docs/user-guide/configmap/kubectl/ui.properties

$ cluster/kubectl.sh get configmaps game-config-2 -o yaml
apiVersion: v1
data:
  game.properties: |-
    enemies=aliens
    lives=3
    enemies.cheat=true
    enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
  ui.properties: |
    color.good=purple
    color.bad=yellow
    allow.textmode=true
    how.nice.to.look=fairlyNice
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T18:52:05Z
  name: game-config-2
  namespace: default
  resourceVersion: "516"
  selfLink: /api/v1/namespaces/default/configmaps/game-config-2
  uid: b4952dc3-d670-11e5-8cd0-68f728db1985

```

We can also set the key to use for an individual file with `--from-file` by passing an expression
of `key=value`: `--from-file=game-special-key=docs/user-guide/configmap/kubectl/game.properties`:

```console

$ kubectl create configmap game-config-3 --from-file=game-special-key=docs/user-guide/configmap/kubectl/game.properties

$ kubectl get configmaps game-config-3 -o yaml
apiVersion: v1
data:
  game-special-key: |-
    enemies=aliens
    lives=3
    enemies.cheat=true
    enemies.cheat.level=noGoodRotten
    secret.code.passphrase=UUDDLRLRBABAS
    secret.code.allowed=true
    secret.code.lives=30
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T18:54:22Z
  name: game-config-3
  namespace: default
  resourceVersion: "530"
  selfLink: /api/v1/namespaces/default/configmaps/game-config-3
  uid: 05f8da22-d671-11e5-8cd0-68f728db1985

```

### Creating from literal values

It is also possible to supply literal values for ConfigMaps using `kubectl create configmap`.  The
`--from-literal` option takes a `key=value` syntax that allows literal values to be supplied
directly on the command line:

```console

$ kubectl create configmap special-config --from-literal=special.how=very --from-literal=special.type=charm

$ kubectl get configmaps special-config -o yaml
apiVersion: v1
data:
  special.how: very
  special.type: charm
kind: ConfigMap
metadata:
  creationTimestamp: 2016-02-18T19:14:38Z
  name: special-config
  namespace: default
  resourceVersion: "651"
  selfLink: /api/v1/namespaces/default/configmaps/special-config
  uid: dadce046-d673-11e5-8cd0-68f728db1985

```

## Consuming ConfigMap in pods

### Use-Case: Consume ConfigMap in environment variables

ConfigMaps can be used to populate the value of command line arguments.  As an example, consider
the following ConfigMap:

```yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config
  namespace: default
data:
  special.how: very
  special.type: charm

```

We can consume the keys of this ConfigMap in a pod like so:

```yaml

apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "env" ]
      env:
        - name: SPECIAL_LEVEL_KEY
          valueFrom:
            configMapKeyRef:
              name: special-configmap
              key: special.how
        - name: SPECIAL_TYPE_KEY
          valueFrom:
            configMapKeyRef:
              name: special-config
              key: data-1
  restartPolicy: Never

```

When this pod is run, its output will include the lines:

```console

SPECIAL_LEVEL_KEY=very
SPECIAL_TYPE_KEY=charm

```

### Use-Case: Set command-line arguments with ConfigMap

ConfigMaps can also be used to set the value of the command or arguments in a container.  This is
accomplished using the kubernetes substitution syntax `$(VAR_NAME)`.  Consider the ConfigMap:

```yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config
  namespace: default
data:
  special.how: very
  special.type: charm

```

In order to inject values into the command line, we must consume the keys we want to use as
environment variables, as in the last example.  Then we can refer to them in a container's command
using the `$(VAR_NAME)` syntax.

```yaml

apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "-c", "echo $(SPECIAL_LEVEL_KEY) $(SPECIAL_TYPE_KEY)" ]
      env:
        - name: SPECIAL_LEVEL_KEY
          valueFrom:
            configMapKeyRef:
              name: special-configmap
              key: special.how
        - name: SPECIAL_TYPE_KEY
          valueFrom:
            configMapKeyRef:
              name: special-config
              key: data-1
  restartPolicy: Never

```

When this pod is run, the output from the `test-container` container will be:

```console

very charm

```

### Use-Case: Consume ConfigMap via volume plugin

ConfigMaps can also be consumed in volumes.  Returning again to our example ConfigMap:

```yaml

apiVersion: v1
kind: ConfigMap
metadata:
  name: special-config
  namespace: default
data:
  special.how: very
  special.type: charm

```

We have a couple different options for consuming this ConfigMap in a volume.  The most basic
way is to populate the volume with files where the key is the filename and the content of the file
is the value of the key:

```yaml

apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "cat", "/etc/config/special.how" ]
      volumeMounts:
      - name: config-volume
        mountPath: /etc/config
  volumes:
    - name: config-volume
      configMap:
        name: special-config
  restartPolicy: Never

```

When this pod is run, the output will be:

```console

very

```

We can also control the paths within the volume where ConfigMap keys are projected:

```yaml

apiVersion: v1
kind: Pod
metadata:
  name: dapi-test-pod
spec:
  containers:
    - name: test-container
      image: gcr.io/google_containers/busybox
      command: [ "/bin/sh", "cat", "/etc/config/path/to/special-key" ]
      volumeMounts:
      - name: config-volume
        mountPath: /etc/config
  volumes:
    - name: config-volume
      configMap:
        name: special-config
        items:
        - key: special.how
          path: path/to/special-key
  restartPolicy: Never

```

When this pod is run, the output will be:

```console

very

```

## Real World Example: Configuring Redis

Let's take a look at a real-world example: configuring redis using ConfigMap.  Say we want to inject
redis with the recommendation configuration for using redis as a cache.  The redis config file
should contain:

```
maxmemory 2mb
maxmemory-policy allkeys-lru
```

Such a file is in `docs/user-guide/configmap/redis`; we can use the following command to create a
ConfigMap instance with it:

```console
$ kubectl create configmap example-redis-config --from-file=docs/user-guide/configmap/redis/redis-config

$ kubectl get configmap redis-config -o yaml
{
    "kind": "ConfigMap",
    "apiVersion": "v1",
    "metadata": {
        "name": "example-redis-config",
        "namespace": "default",
        "selfLink": "/api/v1/namespaces/default/configmaps/example-redis-config",
        "uid": "07fd0419-d97b-11e5-b443-68f728db1985",
        "resourceVersion": "15",
        "creationTimestamp": "2016-02-22T15:43:34Z"
    },
    "data": {
        "redis-config": "maxmemory 2mb\nmaxmemory-policy allkeys-lru\n"
    }
}
```

Now, let's create a pod that uses this config:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: redis
spec:
  containers:
  - name: redis
    image: kubernetes/redis:v1
    env:
    - name: MASTER
      value: "true"
    ports:
    - containerPort: 6379
    resources:
      limits:
        cpu: "0.1"
    volumeMounts:
    - mountPath: /redis-master-data
      name: data
    - mountPath: /redis-master
      name: config
  volumes:
    - name: data
      emptyDir: {}
    - name: config
      configMap:
        name: example-redis-config
        items:
        - key: redis-config
          path: redis.conf
```

Notice that this pod has a ConfigMap volume that places the `redis-config` key of the
`example-redis-config` ConfigMap into a file called `redis.conf`.  This volume is mounted into the
`/redis-master` directory in the redis container, placing our config file at
`/redis-master/redis.conf`, which is where the image looks for the redis config file for the master.

```console
$ kubectl create -f docs/user-guide/configmap/redis/redis-pod.yaml
```

If we `kubectl exec` into this pod and run the `redis-cli` tool, we can check that our config was
applied correctly:

```console
$ kubectl exec -it redis redis-cli
127.0.0.1:6379> CONFIG GET maxmemory
1) "maxmemory"
2) "2097152"
127.0.0.1:6379> CONFIG GET maxmemory-policy
1) "maxmemory-policy"
2) "allkeys-lru"
```

## Restrictions

ConfigMaps must be created before they are consumed in pods.  Controllers may be written to tolerate
missing configuration data; consult individual components configured via ConfigMap on a case-by-case
basis.

ConfigMaps reside in a namespace.   They can only be referenced by pods in the same namespace.

Quota for ConfigMap size is a planned feature.

Kubelet only supports use of ConfigMap for pods it gets from the API server.  This includes any pods
created using kubectl, or indirectly via a replication controller.  It does not include pods created
via the Kubelet's `--manifest-url` flag, its `--config` flag, or its REST API (these are not common
ways to create pods.)



<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/configmap.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

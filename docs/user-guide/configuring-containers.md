<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Configuring and launching containers

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications: Configuring and launching containers](#kubernetes-user-guide-managing-applications-configuring-and-launching-containers)
  - [Configuration in Kubernetes](#configuration-in-kubernetes)
  - [Launching a container using a configuration file](#launching-a-container-using-a-configuration-file)
  - [Validating configuration](#validating-configuration)
  - [Environment variables and variable expansion](#environment-variables-and-variable-expansion)
  - [Viewing pod status](#viewing-pod-status)
  - [Viewing pod output](#viewing-pod-output)
  - [Deleting pods](#deleting-pods)
  - [What's next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

## Configuration in Kubernetes

In addition to the imperative-style commands, such as `kubectl run` and `kubectl expose`, described [elsewhere](quick-start.md), Kubernetes supports declarative configuration. Often times, configuration files are preferable to imperative commands, since they can be checked into version control and changes to the files can be code reviewed, which is especially important for more complex configurations, producing a more robust, reliable and archival system.

In the declarative style, all configuration is stored in YAML or JSON configuration files using Kubernetes's API resource schemas as the configuration schemas. `kubectl` can create, update, delete, and get API resources. The `apiVersion` (currently “v1”), resource `kind`, and resource `name` are used by `kubectl` to construct the appropriate API path to invoke for the specified operation.

## Launching a container using a configuration file

Kubernetes executes containers in [*Pods*](pods.md). A pod containing a simple Hello World container can be specified in YAML as follows:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:  # specification of the pod’s contents
  restartPolicy: Never
  containers:
  - name: hello
    image: "ubuntu:14.04"
    command: ["/bin/echo","hello”,”world"]
```

The value of `metadata.name`, `hello-world`, will be the name of the pod resource created, and must be unique within the cluster, whereas `containers[0].name` is just a nickname for the container within that pod. `image` is the name of the Docker image, which Kubernetes expects to be able to pull from a registry, the [Docker Hub](https://registry.hub.docker.com/) by default.

`restartPolicy: Never` indicates that we just want to run the container once and then terminate the pod.

The [`command`](containers.md#containers-and-commands) overrides the Docker container’s `Entrypoint`. Command arguments (corresponding to Docker’s `Cmd`) may be specified using `args`, as follows:

```yaml
    command: ["/bin/echo"]
    args: ["hello","world"]
```

This pod can be created using the `create` command:

```console
$ kubectl create -f ./hello-world.yaml
pods/hello-world
```

`kubectl` prints the resource type and name of the resource created when successful.

## Validating configuration

If you’re not sure you specified the resource correctly, you can ask `kubectl` to validate it for you:

```console
$ kubectl create -f ./hello-world.yaml --validate
```

Let’s say you specified `entrypoint` instead of `command`. You’d see output as follows:

```console
I0709 06:33:05.600829   14160 schema.go:126] unknown field: entrypoint
I0709 06:33:05.600988   14160 schema.go:129] this may be a false alarm, see http://issue.k8s.io/6842
pods/hello-world
```

`kubectl create --validate` currently warns about problems it detects, but creates the resource anyway, unless a required field is absent or a field value is invalid. Unknown API fields are ignored, so be careful. This pod was created, but with no `command`, which is an optional field, since the image may specify an `Entrypoint`.
View the [Pod API
object](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/release-1.1/docs/api-reference/v1/definitions.html#_v1_pod)
to see the list of valid fields.

## Environment variables and variable expansion

Kubernetes [does not automatically run commands in a shell](https://github.com/kubernetes/kubernetes/wiki/User-FAQ#use-of-environment-variables-on-the-command-line) (not all images contain shells). If you would like to run your command in a shell, such as to expand environment variables (specified using `env`), you could do the following:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: hello-world
spec:  # specification of the pod’s contents
  restartPolicy: Never
  containers:
  - name: hello
    image: "ubuntu:14.04"
    env:
    - name: MESSAGE
      value: "hello world"
    command: ["/bin/sh","-c"]
    args: ["/bin/echo \"${MESSAGE}\""]
```

However, a shell isn’t necessary just to expand environment variables. Kubernetes will do it for you if you use [`$(ENVVAR)` syntax](../../docs/design/expansion.md):

```yaml
    command: ["/bin/echo"]
    args: ["$(MESSAGE)"]
```

## Viewing pod status

You can see the pod you created (actually all of your cluster's pods) using the `get` command.

If you’re quick, it will look as follows:

```console
$ kubectl get pods
NAME          READY     STATUS    RESTARTS   AGE
hello-world   0/1       Pending   0          0s
```

Initially, a newly created pod is unscheduled -- no node has been selected to run it. Scheduling happens after creation, but is fast, so you normally shouldn’t see pods in an unscheduled state unless there’s a problem.

After the pod has been scheduled, the image may need to be pulled to the node on which it was scheduled, if it hadn’t been pulled already. After a few seconds, you should see the container running:

```console
$ kubectl get pods
NAME          READY     STATUS    RESTARTS   AGE
hello-world   1/1       Running   0          5s
```

The `READY` column shows how many containers in the pod are running.

Almost immediately after it starts running, this command will terminate. `kubectl` shows that the container is no longer running and displays the exit status:

```console
$ kubectl get pods
NAME          READY     STATUS       RESTARTS   AGE
hello-world   0/1       ExitCode:0   0          15s
```

## Viewing pod output

You probably want to see the output of the command you ran. As with [`docker logs`](https://docs.docker.com/userguide/usingdocker/), `kubectl logs` will show you the output:

```console
$ kubectl logs hello-world
hello world
```

## Deleting pods

When you’re done looking at the output, you should delete the pod:

```console
$ kubectl delete pod hello-world
pods/hello-world
```

As with `create`, `kubectl` prints the resource type and name of the resource deleted when successful.

You can also use the resource/name format to specify the pod:

```console
$ kubectl delete pods/hello-world
pods/hello-world
```

Terminated pods aren’t currently automatically deleted, so that you can observe their final status, so be sure to clean up your dead pods.

On the other hand, containers and their logs are eventually deleted automatically in order to free up disk space on the nodes.

## What's next?

[Learn about deploying continuously running applications.](deploying-applications.md)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/configuring-containers.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

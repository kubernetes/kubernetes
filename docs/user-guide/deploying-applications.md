<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Deploying continuously running applications

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications: Deploying continuously running applications](#kubernetes-user-guide-managing-applications-deploying-continuously-running-applications)
  - [Launching a set of replicas using a configuration file](#launching-a-set-of-replicas-using-a-configuration-file)
  - [Viewing replication controller status](#viewing-replication-controller-status)
  - [Deleting replication controllers](#deleting-replication-controllers)
  - [Labels](#labels)
  - [What's next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

You previously read about how to quickly deploy a simple replicated application using [`kubectl run`](quick-start.md) and how to configure and launch single-run containers using pods ([Configuring containers](configuring-containers.md)). Here you’ll use the configuration-based approach to deploy a continuously running, replicated application.

## Launching a set of replicas using a configuration file

Kubernetes creates and manages sets of replicated containers (actually, replicated [Pods](pods.md)) using [*Replication Controllers*](replication-controller.md).

A replication controller simply ensures that a specified number of pod "replicas" are running at any one time. If there are too many, it will kill some. If there are too few, it will start more. It’s analogous to Google Compute Engine’s [Instance Group Manager](https://cloud.google.com/compute/docs/instance-groups/manager/) or AWS’s [Auto-scaling Group](http://docs.aws.amazon.com/AutoScaling/latest/DeveloperGuide/AutoScalingGroup.html) (with no scaling policies).

The replication controller created to run nginx by `kubectl run` in the [Quick start](quick-start.md) could be specified using YAML as follows:

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
```

Some differences compared to specifying just a pod are that the `kind` is `ReplicationController`, the number of `replicas` desired is specified, and the pod specification is under the `template` field. The names of the pods don’t need to be specified explicitly because they are generated from the name of the replication controller.
View the [replication controller API
object](https://htmlpreview.github.io/?https://github.com/kubernetes/kubernetes/blob/v1.1.0/docs/api-reference/definitions.html#_v1_replicationcontroller)
to view the list of supported fields.

This replication controller can be created using `create`, just as with pods:

```console
$ kubectl create -f ./nginx-rc.yaml
replicationcontrollers/my-nginx
```

Unlike in the case where you directly create pods, a replication controller replaces pods that are deleted or terminated for any reason, such as in the case of node failure. For this reason, we recommend that you use a replication controller for a continuously running application even if your application requires only a single pod, in which case you can omit `replicas` and it will default to a single replica.

## Viewing replication controller status

You can view the replication controller you created using `get`:

```console
$ kubectl get rc
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR    REPLICAS
my-nginx     nginx          nginx      app=nginx   2
```

This tells you that your controller will ensure that you have two nginx replicas.

You can see those replicas using `get`, just as with pods you created directly:

```console
$ kubectl get pods
NAME             READY     STATUS    RESTARTS   AGE
my-nginx-065jq   1/1       Running   0          51s
my-nginx-buaiq   1/1       Running   0          51s
```

## Deleting replication controllers

When you want to kill your application, delete your replication controller, as in the [Quick start](quick-start.md):

```console
$ kubectl delete rc my-nginx
replicationcontrollers/my-nginx
```

By default, this will also cause the pods managed by the replication controller to be deleted. If there were a large number of pods, this may take a while to complete. If you want to leave the pods running, specify `--cascade=false`.

If you try to delete the pods before deleting the replication controller, it will just replace them, as it is supposed to do.

## Labels

Kubernetes uses user-defined key-value attributes called [*labels*](labels.md) to categorize and identify sets of resources, such as pods and replication controllers. The example above specified a single label in the pod template, with key `app` and value `nginx`. All pods created carry that label, which can be viewed using `-L`:

```console
$ kubectl get pods -L app
NAME             READY     STATUS    RESTARTS   AGE       APP
my-nginx-afv12   0/1       Running   0          3s        nginx
my-nginx-lg99z   0/1       Running   0          3s        nginx
```

The labels from the pod template are copied to the replication controller’s labels by default, as well -- all resources in Kubernetes support labels:

```console
$ kubectl get rc my-nginx -L app
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR    REPLICAS   APP
my-nginx     nginx          nginx      app=nginx   2          nginx
```

More importantly, the pod template’s labels are used to create a [`selector`](labels.md#label-selectors) that will match pods carrying those labels. You can see this field by requesting it using the [Go template output format of `kubectl get`](kubectl/kubectl_get.md):

```console
$ kubectl get rc my-nginx -o template --template="{{.spec.selector}}"
map[app:nginx]
```

You could also specify the `selector` explicitly, such as if you wanted to specify labels in the pod template that you didn’t want to select on, but you should ensure that the selector will match the labels of the pods created from the pod template, and that it won’t match pods created by other replication controllers. The most straightforward way to ensure the latter is to create a unique label value for the replication controller, and to specify it in both the pod template’s labels and in the selector.

## What's next?

[Learn about exposing applications to users and clients, and connecting tiers of your application together.](connecting-applications.md)




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/deploying-applications.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

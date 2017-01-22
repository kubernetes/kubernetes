---
layout: docwithnav
title: "Kubernetes User Guide: Managing Applications: Managing deployments"
---
<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Kubernetes User Guide: Managing Applications: Managing deployments

You’ve deployed your application and exposed it via a service. Now what? Kubernetes provides a number of tools to help you manage your application deployment, including scaling and updating. Among the features we’ll discuss in more depth are [configuration files](configuring-containers.html#configuration-in-kubernetes) and [labels](deploying-applications.html#labels).

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [Kubernetes User Guide: Managing Applications: Managing deployments](#kubernetes-user-guide-managing-applications-managing-deployments)
  - [Organizing resource configurations](#organizing-resource-configurations)
  - [Bulk operations in kubectl](#bulk-operations-in-kubectl)
  - [Using labels effectively](#using-labels-effectively)
  - [Canary deployments](#canary-deployments)
  - [Updating labels](#updating-labels)
  - [Scaling your application](#scaling-your-application)
  - [Updating your application without a service outage](#updating-your-application-without-a-service-outage)
  - [In-place updates of resources](#in-place-updates-of-resources)
  - [Disruptive updates](#disruptive-updates)
  - [What's next?](#whats-next)

<!-- END MUNGE: GENERATED_TOC -->

## Organizing resource configurations

Many applications require multiple resources to be created, such as a Replication Controller and a Service. Management of multiple resources can be simplified by grouping them together in the same file (separated by `---` in YAML). For example:

{% highlight yaml %}
{% raw %}
apiVersion: v1
kind: Service
metadata:
  name: my-nginx-svc
  labels:
    app: nginx
spec:
  type: LoadBalancer
  ports:
  - port: 80
  selector:
    app: nginx
---
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
{% endraw %}
{% endhighlight %}

Multiple resources can be created the same way as a single resource:

{% highlight console %}
{% raw %}
$ kubectl create -f ./nginx-app.yaml
services/my-nginx-svc
replicationcontrollers/my-nginx
{% endraw %}
{% endhighlight %}

The resources will be created in the order they appear in the file. Therefore, it's best to specify the service first, since that will ensure the scheduler can spread the pods associated with the service as they are created by the replication controller(s).

`kubectl create` also accepts multiple `-f` arguments:

{% highlight console %}
{% raw %}
$ kubectl create -f ./nginx-svc.yaml -f ./nginx-rc.yaml
{% endraw %}
{% endhighlight %}

And a directory can be specified rather than or in addition to individual files:

{% highlight console %}
{% raw %}
$ kubectl create -f ./nginx/
{% endraw %}
{% endhighlight %}

`kubectl` will read any files with suffixes `.yaml`, `.yml`, or `.json`.

It is a recommended practice to put resources related to the same microservice or application tier into the same file, and to group all of the files associated with your application in the same directory. If the tiers of your application bind to each other using DNS, then you can then simply deploy all of the components of your stack en masse.

A URL can also be specified as a configuration source, which is handy for deploying directly from configuration files checked into github:

{% highlight console %}
{% raw %}
$ kubectl create -f https://raw.githubusercontent.com/GoogleCloudPlatform/kubernetes/master/docs/user-guide/replication.yaml
replicationcontrollers/nginx
{% endraw %}
{% endhighlight %}

## Bulk operations in kubectl

Resource creation isn’t the only operation that `kubectl` can perform in bulk. It can also extract resource names from configuration files in order to perform other operations, in particular to delete the same resources you created:

{% highlight console %}
{% raw %}
$ kubectl delete -f ./nginx/
replicationcontrollers/my-nginx
services/my-nginx-svc
{% endraw %}
{% endhighlight %}

In the case of just two resources, it’s also easy to specify both on the command line using the resource/name syntax:

{% highlight console %}
{% raw %}
$ kubectl delete replicationcontrollers/my-nginx services/my-nginx-svc
{% endraw %}
{% endhighlight %}

For larger numbers of resources, one can use labels to filter resources. The selector is specified using `-l`:

{% highlight console %}
{% raw %}
$ kubectl delete all -lapp=nginx
replicationcontrollers/my-nginx
services/my-nginx-svc
{% endraw %}
{% endhighlight %}

Because `kubectl` outputs resource names in the same syntax it accepts, it’s easy to chain operations using `$()` or `xargs`:

{% highlight console %}
{% raw %}
$ kubectl get $(kubectl create -f ./nginx/ | grep my-nginx)
CONTROLLER   CONTAINER(S)   IMAGE(S)   SELECTOR    REPLICAS
my-nginx     nginx          nginx      app=nginx   2
NAME           LABELS      SELECTOR    IP(S)          PORT(S)
my-nginx-svc   app=nginx   app=nginx   10.0.152.174   80/TCP
{% endraw %}
{% endhighlight %}

## Using labels effectively

The examples we’ve used so far apply at most a single label to any resource. There are many scenarios where multiple labels should be used to distinguish sets from one another.

For instance, different applications would use different values for the `app` label, but a multi-tier application, such as the [guestbook example](../../examples/guestbook/), would additionally need to distinguish each tier. The frontend could carry the following labels:

{% highlight yaml %}
{% raw %}
     labels:
        app: guestbook
        tier: frontend
{% endraw %}
{% endhighlight %}

while the Redis master and slave would have different `tier` labels, and perhaps even an additional `role` label:

{% highlight yaml %}
{% raw %}
     labels:
        app: guestbook
        tier: backend
        role: master
{% endraw %}
{% endhighlight %}

and

{% highlight yaml %}
{% raw %}
     labels:
        app: guestbook
        tier: backend
        role: slave
{% endraw %}
{% endhighlight %}

The labels allow us to slice and dice our resources along any dimension specified by a label:

{% highlight console %}
{% raw %}
$ kubectl create -f ./guestbook-fe.yaml -f ./redis-master.yaml -f ./redis-slave.yaml
replicationcontrollers/guestbook-fe
replicationcontrollers/guestbook-redis-master
replicationcontrollers/guestbook-redis-slave
$ kubectl get pods -Lapp -Ltier -Lrole
NAME                           READY     STATUS    RESTARTS   AGE       APP         TIER       ROLE
guestbook-fe-4nlpb             1/1       Running   0          1m        guestbook   frontend   <n/a>
guestbook-fe-ght6d             1/1       Running   0          1m        guestbook   frontend   <n/a>
guestbook-fe-jpy62             1/1       Running   0          1m        guestbook   frontend   <n/a>
guestbook-redis-master-5pg3b   1/1       Running   0          1m        guestbook   backend    master
guestbook-redis-slave-2q2yf    1/1       Running   0          1m        guestbook   backend    slave
guestbook-redis-slave-qgazl    1/1       Running   0          1m        guestbook   backend    slave
my-nginx-divi2                 1/1       Running   0          29m       nginx       <n/a>      <n/a>
my-nginx-o0ef1                 1/1       Running   0          29m       nginx       <n/a>      <n/a>
$ kubectl get pods -lapp=guestbook,role=slave
NAME                          READY     STATUS    RESTARTS   AGE
guestbook-redis-slave-2q2yf   1/1       Running   0          3m
guestbook-redis-slave-qgazl   1/1       Running   0          3m
{% endraw %}
{% endhighlight %}

## Canary deployments

Another scenario where multiple labels are needed is to distinguish deployments of different releases or configurations of the same component. For example, it is common practice to deploy a *canary* of a new application release (specified via image tag) side by side with the previous release so that the new release can receive live production traffic before fully rolling it out. For instance, a new release of the guestbook frontend might carry the following labels:

{% highlight yaml %}
{% raw %}
     labels:
        app: guestbook
        tier: frontend
        track: canary
{% endraw %}
{% endhighlight %}

and the primary, stable release would have a different value of the `track` label, so that the sets of pods controlled by the two replication controllers would not overlap:

{% highlight yaml %}
{% raw %}
     labels:
        app: guestbook
        tier: frontend
        track: stable
{% endraw %}
{% endhighlight %}

The frontend service would span both sets of replicas by selecting the common subset of their labels, omitting the `track` label:

{% highlight yaml %}
{% raw %}
  selector:
     app: guestbook
     tier: frontend
{% endraw %}
{% endhighlight %}

## Updating labels

Sometimes existing pods and other resources need to be relabeled before creating new resources. This can be done with `kubectl label`. For example:

{% highlight console %}
{% raw %}
$ kubectl label pods -lapp=nginx tier=fe
NAME                READY     STATUS    RESTARTS   AGE
my-nginx-v4-9gw19   1/1       Running   0          14m
NAME                READY     STATUS    RESTARTS   AGE
my-nginx-v4-hayza   1/1       Running   0          13m
NAME                READY     STATUS    RESTARTS   AGE
my-nginx-v4-mde6m   1/1       Running   0          17m
NAME                READY     STATUS    RESTARTS   AGE
my-nginx-v4-sh6m8   1/1       Running   0          18m
NAME                READY     STATUS    RESTARTS   AGE
my-nginx-v4-wfof4   1/1       Running   0          16m
$ kubectl get pods -lapp=nginx -Ltier
NAME                READY     STATUS    RESTARTS   AGE       TIER
my-nginx-v4-9gw19   1/1       Running   0          15m       fe
my-nginx-v4-hayza   1/1       Running   0          14m       fe
my-nginx-v4-mde6m   1/1       Running   0          18m       fe
my-nginx-v4-sh6m8   1/1       Running   0          19m       fe
my-nginx-v4-wfof4   1/1       Running   0          16m       fe
{% endraw %}
{% endhighlight %}

## Scaling your application

When load on your application grows or shrinks, it’s easy to scale with `kubectl`. For instance, to increase the number of nginx replicas from 2 to 3, do:

{% highlight console %}
{% raw %}
$ kubectl scale rc my-nginx --replicas=3
scaled
$ kubectl get pods -lapp=nginx
NAME             READY     STATUS    RESTARTS   AGE
my-nginx-1jgkf   1/1       Running   0          3m
my-nginx-divi2   1/1       Running   0          1h
my-nginx-o0ef1   1/1       Running   0          1h
{% endraw %}
{% endhighlight %}

## Updating your application without a service outage

At some point, you’ll eventually need to update your deployed application, typically by specifying a new image or image tag, as in the canary deployment scenario above. `kubectl` supports several update operations, each of which is applicable to different scenarios.

To update a service without an outage, `kubectl` supports what is called [“rolling update”](kubectl/kubectl_rolling-update.html), which updates one pod at a time, rather than taking down the entire service at the same time. See the [rolling update design document](../design/simple-rolling-update.html) and the [example of rolling update](update-demo/) for more information.

Let’s say you were running version 1.7.9 of nginx:

{% highlight yaml %}
{% raw %}
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx
spec:
  replicas: 5
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.7.9
        ports:
        - containerPort: 80
{% endraw %}
{% endhighlight %}

To update to version 1.9.1, you can use [`kubectl rolling-update --image`](../../docs/design/simple-rolling-update.html):

{% highlight console %}
{% raw %}
$ kubectl rolling-update my-nginx --image=nginx:1.9.1
Creating my-nginx-ccba8fbd8cc8160970f63f9a2696fc46
{% endraw %}
{% endhighlight %}

In another window, you can see that `kubectl` added a `deployment` label to the pods, whose value is a hash of the configuration, to distinguish the new pods from the old:

{% highlight console %}
{% raw %}
$ kubectl get pods -lapp=nginx -Ldeployment
NAME                                              READY     STATUS    RESTARTS   AGE       DEPLOYMENT
my-nginx-1jgkf                                    1/1       Running   0          1h        2d1d7a8f682934a254002b56404b813e
my-nginx-ccba8fbd8cc8160970f63f9a2696fc46-k156z   1/1       Running   0          1m        ccba8fbd8cc8160970f63f9a2696fc46
my-nginx-ccba8fbd8cc8160970f63f9a2696fc46-v95yh   1/1       Running   0          35s       ccba8fbd8cc8160970f63f9a2696fc46
my-nginx-divi2                                    1/1       Running   0          2h        2d1d7a8f682934a254002b56404b813e
my-nginx-o0ef1                                    1/1       Running   0          2h        2d1d7a8f682934a254002b56404b813e
my-nginx-q6all                                    1/1       Running   0          8m        2d1d7a8f682934a254002b56404b813e
{% endraw %}
{% endhighlight %}

`kubectl rolling-update` reports progress as it progresses:

{% highlight console %}
{% raw %}
Updating my-nginx replicas: 4, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 1
At end of loop: my-nginx replicas: 4, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 1
At beginning of loop: my-nginx replicas: 3, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 2
Updating my-nginx replicas: 3, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 2
At end of loop: my-nginx replicas: 3, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 2
At beginning of loop: my-nginx replicas: 2, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 3
Updating my-nginx replicas: 2, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 3
At end of loop: my-nginx replicas: 2, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 3
At beginning of loop: my-nginx replicas: 1, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 4
Updating my-nginx replicas: 1, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 4
At end of loop: my-nginx replicas: 1, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 4
At beginning of loop: my-nginx replicas: 0, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 5
Updating my-nginx replicas: 0, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 5
At end of loop: my-nginx replicas: 0, my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 replicas: 5
Update succeeded. Deleting old controller: my-nginx
Renaming my-nginx-ccba8fbd8cc8160970f63f9a2696fc46 to my-nginx
my-nginx
{% endraw %}
{% endhighlight %}

If you encounter a problem, you can stop the rolling update midway and revert to the previous version using `--rollback`:

{% highlight console %}
{% raw %}
$ kubectl kubectl rolling-update my-nginx  --image=nginx:1.9.1 --rollback
Found existing update in progress (my-nginx-ccba8fbd8cc8160970f63f9a2696fc46), resuming.
Found desired replicas.Continuing update with existing controller my-nginx.
Stopping my-nginx-02ca3e87d8685813dbe1f8c164a46f02 replicas: 1 -> 0
Update succeeded. Deleting my-nginx-ccba8fbd8cc8160970f63f9a2696fc46
my-nginx
{% endraw %}
{% endhighlight %}

This is one example where the immutability of containers is a huge asset.

If you need to update more than just the image (e.g., command arguments, environment variables), you can create a new replication controller, with a new name and distinguishing label value, such as:

{% highlight yaml %}
{% raw %}
apiVersion: v1
kind: ReplicationController
metadata:
  name: my-nginx-v4
spec:
  replicas: 5
  selector:
    app: nginx
    deployment: v4
  template:
    metadata:
      labels:
        app: nginx
        deployment: v4
    spec:
      containers:
      - name: nginx
        image: nginx:1.9.2
        args: ["nginx","-T"]
        ports:
        - containerPort: 80
{% endraw %}
{% endhighlight %}

and roll it out:

{% highlight console %}
{% raw %}
$ kubectl rolling-update my-nginx -f ./nginx-rc.yaml
Creating my-nginx-v4
At beginning of loop: my-nginx replicas: 4, my-nginx-v4 replicas: 1
Updating my-nginx replicas: 4, my-nginx-v4 replicas: 1
At end of loop: my-nginx replicas: 4, my-nginx-v4 replicas: 1
At beginning of loop: my-nginx replicas: 3, my-nginx-v4 replicas: 2
Updating my-nginx replicas: 3, my-nginx-v4 replicas: 2
At end of loop: my-nginx replicas: 3, my-nginx-v4 replicas: 2
At beginning of loop: my-nginx replicas: 2, my-nginx-v4 replicas: 3
Updating my-nginx replicas: 2, my-nginx-v4 replicas: 3
At end of loop: my-nginx replicas: 2, my-nginx-v4 replicas: 3
At beginning of loop: my-nginx replicas: 1, my-nginx-v4 replicas: 4
Updating my-nginx replicas: 1, my-nginx-v4 replicas: 4
At end of loop: my-nginx replicas: 1, my-nginx-v4 replicas: 4
At beginning of loop: my-nginx replicas: 0, my-nginx-v4 replicas: 5
Updating my-nginx replicas: 0, my-nginx-v4 replicas: 5
At end of loop: my-nginx replicas: 0, my-nginx-v4 replicas: 5
Update succeeded. Deleting my-nginx
my-nginx-v4
{% endraw %}
{% endhighlight %}

You can also run the [update demo](update-demo/) to see a visual representation of the rolling update process.

## In-place updates of resources

Sometimes it’s necessary to make narrow, non-disruptive updates to resources you’ve created. For instance, you might want to add an [annotation](annotations.html) with a description of your object. That’s easiest to do with `kubectl patch`:

{% highlight console %}
{% raw %}
$ kubectl patch rc my-nginx-v4 -p '{"metadata": {"annotations": {"description": "my frontend running nginx"}}}'
my-nginx-v4
$ kubectl get rc my-nginx-v4 -o yaml
apiVersion: v1
kind: ReplicationController
metadata:
  annotations:
    description: my frontend running nginx
...
{% endraw %}
{% endhighlight %}

The patch is specified using json.

For more significant changes, you can `get` the resource, edit it, and then `replace` the resource with the updated version:

{% highlight console %}
{% raw %}
$ kubectl get rc my-nginx-v4 -o yaml > /tmp/nginx.yaml
$ vi /tmp/nginx.yaml
$ kubectl replace -f /tmp/nginx.yaml
replicationcontrollers/my-nginx-v4
$ rm $TMP
{% endraw %}
{% endhighlight %}

The system ensures that you don’t clobber changes made by other users or components by confirming that the `resourceVersion` doesn’t differ from the version you edited. If you want to update regardless of other changes, remove the `resourceVersion` field when you edit the resource. However, if you do this, don’t use your original configuration file as the source since additional fields most likely were set in the live state.

## Disruptive updates

In some cases, you may need to update resource fields that cannot be updated once initialized, or you may just want to make a recursive change immediately, such as to fix broken pods created by a replication controller. To change such fields, use `replace --force`, which deletes and re-creates the resource. In this case, you can simply modify your original configuration file:

{% highlight console %}
{% raw %}
$ kubectl replace -f ./nginx-rc.yaml --force
replicationcontrollers/my-nginx-v4
replicationcontrollers/my-nginx-v4
{% endraw %}
{% endhighlight %}

## What's next?

- [Learn about how to use `kubectl` for application introspection and debugging.](introspection-and-debugging.html)
- [Tips and tricks when working with config](config-best-practices.html)


<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/managing-deployments.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

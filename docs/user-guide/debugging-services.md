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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/debugging-services.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# My Service is not working - how to debug

An issue that comes up rather frequently for new installations of Kubernetes is
that `Services` are not working properly.  You've run all your `Pod`s and
`ReplicationController`s, but you get no response when you try to access them.
This document will hopefully help you to figure out what's going wrong.

**Table of Contents**
<!-- BEGIN MUNGE: GENERATED_TOC -->

- [My Service is not working - how to debug](#my-service-is-not-working---how-to-debug)
  - [Conventions](#conventions)
  - [Running commands in a Pod](#running-commands-in-a-pod)
  - [Setup](#setup)
  - [Does the Service exist?](#does-the-service-exist)
  - [Does the Service work by DNS?](#does-the-service-work-by-dns)
    - [Does any Service exist in DNS?](#does-any-service-exist-in-dns)
  - [Does the Service work by IP?](#does-the-service-work-by-ip)
  - [Is the Service correct?](#is-the-service-correct)
  - [Does the Service have any Endpoints?](#does-the-service-have-any-endpoints)
  - [Are the Pods working?](#are-the-pods-working)
  - [Is the kube-proxy working?](#is-the-kube-proxy-working)
    - [Is kube-proxy running?](#is-kube-proxy-running)
    - [Is kube-proxy writing iptables rules?](#is-kube-proxy-writing-iptables-rules)
    - [Is kube-proxy proxying?](#is-kube-proxy-proxying)
  - [Seek help](#seek-help)
  - [More information](#more-information)

<!-- END MUNGE: GENERATED_TOC -->

## Conventions

Throughout this doc you will see various commands that you can run.  Some
commands need to be run within `Pod`, others on a Kubernetes `Node`, and others
can run anywhere you have `kubectl` and credentials for the cluster.  To make it
clear what is expected, this document will use the following conventions.

If the command "COMMAND" is expected to run in a `Pod` and produce "OUTPUT":

```console
u@pod$ COMMAND
OUTPUT
```

If the command "COMMAND" is expected to run on a `Node` and produce "OUTPUT":

```console
u@node$ COMMAND
OUTPUT
```

If the command is "kubectl ARGS":

```console
$ kubectl ARGS
OUTPUT
```

## Running commands in a Pod

For many steps here you will want to see what a `Pod` running in the cluster
sees.  Kubernetes does not directly support interactive `Pod`s (yet), but you can
approximate it:

```console
$ cat <<EOF | kubectl create -f -
apiVersion: v1
kind: Pod
metadata:
  name: busybox-sleep
spec:
  containers:
  - name: busybox
    image: busybox
    args:
    - sleep
    - "1000000"
EOF
pods/busybox-sleep
```

Now, when you need to run a command (even an interactive shell) in a `Pod`-like
context, use:

```console
$ kubectl exec busybox-sleep -- <COMMAND>
```

or

```console
$ kubectl exec -ti busybox-sleep sh
/ #
```

## Setup

For the purposes of this walk-through, let's run some `Pod`s.  Since you're
probably debugging your own `Service` you can substitute your own details, or you
can follow along and get a second data point.

```console
$ kubectl run hostnames --image=gcr.io/google_containers/serve_hostname \
                        --labels=app=hostnames \
                        --port=9376 \
                        --replicas=3
CONTROLLER   CONTAINER(S)   IMAGE(S)                                  SELECTOR        REPLICAS
hostnames    hostnames      gcr.io/google_containers/serve_hostname   app=hostnames   3
```

Note that this is the same as if you had started the `ReplicationController` with
the following YAML:

```yaml
apiVersion: v1
kind: ReplicationController
metadata:
  name: hostnames
spec:
  selector:
    app: hostnames
  replicas: 3
  template:
    metadata:
      labels:
        app: hostnames
    spec:
      containers:
      - name: hostnames
        image: gcr.io/google_containers/serve_hostname
        ports:
        - containerPort: 9376
          protocol: TCP
```

Confirm your `Pod`s are running:

```console
$ kubectl get pods -l app=hostnames
NAME              READY     STATUS    RESTARTS   AGE
hostnames-0uton   1/1       Running   0          12s
hostnames-bvc05   1/1       Running   0          12s
hostnames-yp2kp   1/1       Running   0          12s
```

## Does the Service exist?

The astute reader will have noticed that we did not actually create a `Service`
yet - that is intentional.  This is a step that sometimes gets forgotten, and
is the first thing to check.

So what would happen if I tried to access a non-existent `Service`?  Assuming you
have another `Pod` that consumes this `Service` by name you would get something
like:

```console
u@pod$ wget -qO- hostnames
wget: bad address 'hostname'
```

or:

```console
u@pod$ echo $HOSTNAMES_SERVICE_HOST
```

So the first thing to check is whether that `Service` actually exists:

```console
$ kubectl get svc hostnames
Error from server: service "hostnames" not found
```

So we have a culprit, let's create the `Service`.  As before, this is for the
walk-through - you can use your own `Service`'s details here.

```console
$ kubectl expose rc hostnames --port=80 --target-port=9376
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
hostnames         10.0.0.1         <none>            80/TCP        run=hostnames          1h
```

And read it back, just to be sure:

```console
$ kubectl get svc hostnames
NAME              CLUSTER_IP       EXTERNAL_IP       PORT(S)       SELECTOR               AGE
hostnames         10.0.0.1         <none>            80/TCP        run=hostnames          1h
```

As before, this is the same as if you had started the `Service` with YAML:

```yaml
apiVersion: v1
kind: Service
metadata:
  name: hostnames
spec:
  selector:
    app: hostnames
  ports:
  - name: default
    protocol: TCP
    port: 80
    targetPort: 9376
```

Now you can confirm that the `Service` exists.

## Does the Service work by DNS?

From a `Pod` in the same `Namespace`:

```console
u@pod$ nslookup hostnames
Server:         10.0.0.10
Address:        10.0.0.10#53

Name:   hostnames
Address: 10.0.1.175
```

If this fails, perhaps your `Pod` and `Service` are in different
`Namespace`s, try a namespace-qualified name:

```console
u@pod$ nslookup hostnames.default
Server:         10.0.0.10
Address:        10.0.0.10#53

Name:   hostnames.default
Address: 10.0.1.175
```

If this works, you'll need to ensure that `Pod`s and `Service`s run in the same
`Namespace`.  If this still fails, try a fully-qualified name:

```console
u@pod$ nslookup hostnames.default.svc.cluster.local
Server:         10.0.0.10
Address:        10.0.0.10#53

Name:   hostnames.default.svc.cluster.local
Address: 10.0.1.175
```

Note the suffix here: "default.svc.cluster.local".  The "default" is the
`Namespace` we're operating in.  The "svc" denotes that this is a `Service`.
The "cluster.local" is your cluster domain.

You can also try this from a `Node` in the cluster (note: 10.0.0.10 is my DNS
`Service`):

```console
u@node$ nslookup hostnames.default.svc.cluster.local 10.0.0.10
Server:         10.0.0.10
Address:        10.0.0.10#53

Name:   hostnames.default.svc.cluster.local
Address: 10.0.1.175
```

If you are able to do a fully-qualified name lookup but not a relative one, you
need to check that your `kubelet` is running with the right flags.
The `--cluster-dns` flag needs to point to your DNS `Service`'s IP and the
`--cluster-domain` flag needs to be your cluster's domain - we assumed
"cluster.local" in this document, but yours might be different, in which case
you should change that in all of the commands above.

### Does any Service exist in DNS?

If the above still fails - DNS lookups are not working for your `Service` - we
can take a step back and see what else is not working.  The Kubernetes master
`Service` should always work:

```console
u@pod$ nslookup kubernetes.default
Server:    10.0.0.10
Address 1: 10.0.0.10

Name:      kubernetes
Address 1: 10.0.0.1
```

If this fails, you might need to go to the kube-proxy section of this doc, or
even go back to the top of this document and start over, but instead of
debugging your own `Service`, debug DNS.

## Does the Service work by IP?

The next thing to test is whether your `Service` works at all.  From a
`Node` in your cluster, access the `Service`'s IP (from `kubectl get` above).

```console
u@node$ curl 10.0.1.175:80
hostnames-0uton

u@node$ curl 10.0.1.175:80
hostnames-yp2kp

u@node$ curl 10.0.1.175:80
hostnames-bvc05
```

If your `Service` is working, you should get correct responses.  If not, there
are a number of things that could be going wrong.  Read on.

## Is the Service correct?

It might sound silly, but you should really double and triple check that your
`Service` is correct and matches your `Pods`.  Read back your `Service` and
verify it:

```console
$ kubectl get service hostnames -o json
{
    "kind": "Service",
    "apiVersion": "v1",
    "metadata": {
        "name": "hostnames",
        "namespace": "default",
        "selfLink": "/api/v1/namespaces/default/services/hostnames",
        "uid": "428c8b6c-24bc-11e5-936d-42010af0a9bc",
        "resourceVersion": "347189",
        "creationTimestamp": "2015-07-07T15:24:29Z",
        "labels": {
            "app": "hostnames"
        }
    },
    "spec": {
        "ports": [
            {
                "name": "default",
                "protocol": "TCP",
                "port": 80,
                "targetPort": 9376,
                "nodePort": 0
            }
        ],
        "selector": {
            "app": "hostnames"
        },
        "clusterIP": "10.0.1.175",
        "type": "ClusterIP",
        "sessionAffinity": "None"
    },
    "status": {
        "loadBalancer": {}
    }
}
```

Is the port you are trying to access in `spec.ports[]`?  Is the `targetPort`
correct for your `Pod`s?  If you meant it to be a numeric port, is it a number
(9376) or a string "9376"?  If you meant it to be a named port, do your `Pod`s
expose a port with the same name?  Is the port's `protocol` the same as the
`Pod`'s?

## Does the Service have any Endpoints?

If you got this far, we assume that you have confirmed that your `Service`
exists and resolves by DNS.  Now let's check that the `Pod`s you ran are
actually being selected by the `Service`.

Earlier we saw that the `Pod`s were running.  We can re-check that:

```console
$ kubectl get pods -l app=hostnames
NAME              READY     STATUS    RESTARTS   AGE
hostnames-0uton   1/1       Running   0          1h
hostnames-bvc05   1/1       Running   0          1h
hostnames-yp2kp   1/1       Running   0          1h
```

The "AGE" column says that these `Pod`s are about an hour old, which implies that
they are running fine and not crashing.

The `-l app=hostnames` argument is a label selector - just like our `Service`
has.  Inside the Kubernetes system is a control loop which evaluates the
selector of every `Service` and save the results into an `Endpoints` object.

```console
$ kubectl get endpoints hostnames
NAME        ENDPOINTS
hostnames   10.244.0.5:9376,10.244.0.6:9376,10.244.0.7:9376
```

This confirms that the control loop has found the correct `Pod`s for your
`Service`.  If the `hostnames` row is blank, you should check that the
`spec.selector` field of your `Service` actually selects for `metadata.labels`
values on your `Pod`s.

## Are the Pods working?

At this point, we know that your `Service` exists and has selected your `Pod`s.
Let's check that the `Pod`s are actually working - we can bypass the `Service`
mechanism and go straight to the `Pod`s.

```console
u@pod$ wget -qO- 10.244.0.5:9376
hostnames-0uton

pod $ wget -qO- 10.244.0.6:9376
hostnames-bvc05

u@pod$ wget -qO- 10.244.0.7:9376
hostnames-yp2kp
```

We expect each `Pod` in the `Endpoints` list to return its own hostname.  If
this is not what happens (or whatever the correct behavior is for your own
`Pod`s), you should investigate what's happening there.  You might find
`kubectl logs` to be useful or `kubectl exec` directly to your `Pod`s and check
service from there.

## Is the kube-proxy working?

If you get here, your `Service` is running, has `Endpoints`, and your `Pod`s
are actually serving.  At this point, the whole `Service` proxy mechanism is
suspect.  Let's confirm it, piece by piece.

### Is kube-proxy running?

Confirm that `kube-proxy` is running on your `Node`s.  You should get something
like the below:

```console
u@node$ ps auxw | grep kube-proxy
root  4194  0.4  0.1 101864 17696 ?    Sl Jul04  25:43 /usr/local/bin/kube-proxy --master=https://kubernetes-master --kubeconfig=/var/lib/kube-proxy/kubeconfig --v=2
```

Next, confirm that it is not failing something obvious, like contacting the
master.  To do this, you'll have to look at the logs.  Accessing the logs
depends on your `Node` OS.  On some OSes it is a file, such as
/var/log/kube-proxy.log, while other OSes use `journalctl` to access logs.  You
should see something like:

```console
I0707 17:34:53.945651   30031 server.go:88] Running in resource-only container "/kube-proxy"
I0707 17:34:53.945921   30031 proxier.go:121] Setting proxy IP to 10.240.115.247 and initializing iptables
I0707 17:34:54.053023   30031 roundrobin.go:262] LoadBalancerRR: Setting endpoints for default/kubernetes: to [10.240.169.188:443]
I0707 17:34:54.053175   30031 roundrobin.go:262] LoadBalancerRR: Setting endpoints for default/hostnames:default to [10.244.0.5:9376 10.244.0.6:9376 10.244.0.7:9376]
I0707 17:34:54.053284   30031 roundrobin.go:262] LoadBalancerRR: Setting endpoints for default/kube-dns:dns to [10.244.3.3:53]
I0707 17:34:54.053310   30031 roundrobin.go:262] LoadBalancerRR: Setting endpoints for default/kube-dns:dns-tcp to [10.244.3.3:53]
I0707 17:34:54.054780   30031 proxier.go:306] Adding new service "default/kubernetes:" at 10.0.0.1:443/TCP
I0707 17:34:54.054903   30031 proxier.go:247] Proxying for service "default/kubernetes:" on TCP port 40074
I0707 17:34:54.079181   30031 proxier.go:306] Adding new service "default/hostnames:default" at 10.0.1.175:80/TCP
I0707 17:34:54.079273   30031 proxier.go:247] Proxying for service "default/hostnames:default" on TCP port 48577
I0707 17:34:54.113665   30031 proxier.go:306] Adding new service "default/kube-dns:dns" at 10.0.0.10:53/UDP
I0707 17:34:54.113776   30031 proxier.go:247] Proxying for service "default/kube-dns:dns" on UDP port 34149
I0707 17:34:54.120224   30031 proxier.go:306] Adding new service "default/kube-dns:dns-tcp" at 10.0.0.10:53/TCP
I0707 17:34:54.120297   30031 proxier.go:247] Proxying for service "default/kube-dns:dns-tcp" on TCP port 53476
I0707 17:34:54.902313   30031 proxysocket.go:130] Accepted TCP connection from 10.244.3.3:42670 to 10.244.3.1:40074
I0707 17:34:54.903107   30031 proxysocket.go:130] Accepted TCP connection from 10.244.3.3:42671 to 10.244.3.1:40074
I0707 17:35:46.015868   30031 proxysocket.go:246] New UDP connection from 10.244.3.2:57493
I0707 17:35:46.017061   30031 proxysocket.go:246] New UDP connection from 10.244.3.2:55471
```

If you see error messages about not being able to contact the master, you
should double-check your `Node` configuration and installation steps.

### Is kube-proxy writing iptables rules?

One of the main responsibilities of `kube-proxy` is to write the `iptables`
rules which implement `Service`s.  Let's check that those rules are getting
written.

```console
u@node$ iptables-save | grep hostnames
-A KUBE-PORTALS-CONTAINER -d 10.0.1.175/32 -p tcp -m comment --comment "default/hostnames:default" -m tcp --dport 80 -j REDIRECT --to-ports 48577
-A KUBE-PORTALS-HOST -d 10.0.1.175/32 -p tcp -m comment --comment "default/hostnames:default" -m tcp --dport 80 -j DNAT --to-destination 10.240.115.247:48577
```

There should be 2 rules for each port on your `Service` (just one in this
example) - a "KUBE-PORTALS-CONTAINER" and a "KUBE-PORTALS-HOST".  If you do
not see these, try restarting `kube-proxy` with the `-V` flag set to 4, and
then look at the logs again.

### Is kube-proxy proxying?

Assuming you do see the above rules, try again to access your `Service` by IP:

```console
u@node$ curl 10.0.1.175:80
hostnames-0uton
```

If this fails, we can try accessing the proxy directly.  Look back at the
`iptables-save` output above, and extract the port number that `kube-proxy` is
using for your `Service`.  In the above examples it is "48577".  Now connect to
that:

```console
u@node$ curl localhost:48577
hostnames-yp2kp
```

If this still fails, look at the `kube-proxy` logs for specific lines like:

```console
Setting endpoints for default/hostnames:default to [10.244.0.5:9376 10.244.0.6:9376 10.244.0.7:9376]
```

If you don't see those, try restarting `kube-proxy` with the `-V` flag set to 4, and
then look at the logs again.

## Seek help

If you get this far, something very strange is happening.  Your `Service` is
running, has `Endpoints`, and your `Pod`s are actually serving.  You have DNS
working, `iptables` rules installed, and `kube-proxy` does not seem to be
misbehaving.  And yet your `Service` is not working.  You should probably let
us know, so we can help investigate!

Contact us on
[Slack](../troubleshooting.md#slack) or
[email](https://groups.google.com/forum/#!forum/google-containers) or
[GitHub](https://github.com/kubernetes/kubernetes).

## More information

Visit [troubleshooting document](../troubleshooting.md) for more information.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/debugging-services.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

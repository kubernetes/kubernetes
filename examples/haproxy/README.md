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
[here](http://releases.k8s.io/release-1.0/examples/haproxy/README.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# haproxy

This example creates an haproxy service that listens on a ClusterIP and reverse
proxies connections to backend services. Configuration for haproxy is added to a
secrets volume which can then be updated independently of the kubernetes
configuration and applied by restarting the pods. The only thing that can't be
changed in this manner is the port on which haproxy listens, which must be
specificed in the pod configuration.

The [included configuration](haproxy-cfg/haproxy.cfg) will log to the [example
rsyslog service](../rsyslog/README.md).

## Creating Secrets Volume

The configuration for haproxy is stored in a secrets volume which is attached
when the pod is started. Use the `make_secrets.go` script to help create the
volume:

```sh
% go run examples/utils/make_secrets.go examples/haproxy/haproxy-cfg | \
	kubectl create -f -
secrets/haproxy-cfg
```

## Create the haproxy Service and Replication Controller

The example [Service](haproxy-service.yaml) and [Replication
Controller](haproxy-controller.yaml) should work without modification:

```sh
% kubectl create -f examples/haproxy
services/haproxy-example
replicationcontrollers/haproxy-example
```

The example configuration is a reverse proxy for google.com:

```sh
% kubectl get po -lapp=haproxy
NAME                       READY     STATUS    RESTARTS   AGE
haproxy-controller-ciw90   1/1       Running   0          22h

% kubectl port-forward -p haproxy-controller-ciw90 8080:80 &
I1028 16:36:27.368258   17012 portforward.go:225] Forwarding from 127.0.0.1:8080 -> 80
I1028 16:36:27.368510   17012 portforward.go:225] Forwarding from [::1]:8080 -> 80

% curl -Is localhost:8080 | grep Location
I1028 17:41:55.622429   21637 portforward.go:251] Handling connection for 8080
Location: http://www.google.com/

% kill %1
[1]  + exit 2     kubectl port-forward -p haproxy-controller-ciw90 8080:80
```

And logs to [the example rsyslog-service](../rsyslog/README.md):

```sh
% kubectl exec haproxy-controller-ciw90 -- kill -HUP 1
% kubectl exec rsyslog-controller-vgox2 tail /log/messages
2015-10-28T23:21:33+00:00 haproxy[1]: SIGHUP received, dumping servers states for proxy http-in.
2015-10-28T23:21:33+00:00 haproxy[1]: SIGHUP: Server http-in/google is UP. Conn: 0 act, 0 pend, 0 tot.
2015-10-28T23:21:33+00:00 haproxy[1]: SIGHUP: Proxy http-in has 1 active servers and 0 backup servers available. Conn: act(FE+BE): 0+0, 0 pend (0 unass), tot(FE+BE): 0+0.
```

## Updating the haproxy Configuration

Let's update our haproxy configuration to proxy wikipedia instead:

```sh
% ed examples/haproxy/haproxy-cfg/haproxy.cfg
194
s/google www.google.com/wikipedia wikipedia.org/
wq
199

% git diff examples/haproxy/haproxy-cfg/haproxy.cfg
diff --git a/examples/haproxy/haproxy-cfg/haproxy.cfg b/examples/haproxy/haproxy-cfg/haproxy.cfg
index d454f7d..c40f5ed 100644
--- a/examples/haproxy/haproxy-cfg/haproxy.cfg
+++ b/examples/haproxy/haproxy-cfg/haproxy.cfg
@@ -10,4 +10,4 @@ defaults
 
 listen http-in
   bind *:80
-  server google www.google.com:80
+  server wikipedia wikipedia.org:80

% go run examples/utils/make_secrets.go examples/haproxy/haproxy-cfg | kubectl replace -f -
secrets/haproxy-cfg
```

Now we've updated the configuration in the secrets volume, but this doesn't
affect running pods:

```sh
% curl -Is localhost:8080 | grep Location
I1028 17:41:55.622429   21637 portforward.go:251] Handling connection for 8080
Location: http://www.google.com/
```

If we stop the pod the replication controller will restart it using the updated
secrets volume:

```sh
% kubectl stop po -lapp=haproxy
pods/haproxy-controller-8d64q

% kubectl get po -lapp=haproxy
NAME                       READY     STATUS    RESTARTS   AGE
haproxy-controller-b59v7   1/1       Running   0          2m

% kubectl port-forward -p haproxy-controller-b59v7 8080:80 &
I1028 17:35:21.727891   21136 portforward.go:225] Forwarding from 127.0.0.1:8080 -> 80
I1028 17:35:21.728354   21136 portforward.go:225] Forwarding from [::1]:8080 -> 80

% curl -Is --resolve wikipedia.org:8080:127.0.0.1 wikipedia.org:8080 |  grep Location
I1028 17:35:47.160290   21136 portforward.go:251] Handling connection for 8080
Location: http://www.wikipedia.org/
```

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/haproxy/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

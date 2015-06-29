# Proxy a pod port or host port to a kubernetes Service

While Kubernetes provides the ability to map Services to ports on each node,
those ports are in a special range set aside for allocations.  This means you
can not not simply choose to expose a Service on port 80 on your nodes.  You
also can not choose to expose it on some nodes but not others.  These things
will be fixed in the future, but until then, here is a stop-gap measure you can
use.

The container image `gcr.io/google_containers/proxy-to-service:v2` is a very
small container that will do port-forwarding for you.  You can use it to
forward a pod port or a host port to a service.  Pods can choose any port or
host port, and are not limited in the same way Services are.

For example, suppose you want to forward a node's port 53 (DNS) to your
cluster's DNS service.  The following pod would do the trick:

```
apiVersion: v1
kind: Pod
metadata:
  name: dns-proxy
spec:
  containers:
  - name: proxy-udp
    image: gcr.io/google_containers/proxy-to-service:v2
    args: [ "udp", "53", "kube-dns.default", "1" ]
    ports:
    - name: udp
      protocol: UDP
      containerPort: 53
      hostPort: 53
  - name: proxy-tcp
    image: gcr.io/google_containers/proxy-to-service:v2
    args: [ "tcp", "53", "kube-dns.default" ]
    ports:
    - name: tcp
      protocol: TCP
      containerPort: 53
      hostPort: 53
```

This creates a pod with two containers (one for TCP, one for UDP).  Each
container receives traffic on a port (53 here) and forwards that traffic to the
`kube-dns` service.  You can run this on as many or as few nodes as you want.

Note that the UDP container has a 4th argument - this is a timeout.  Unlike
TCP, UDP does not really have a concept of "connection terminated".  If you
need to proxy UDP, you should choose an appropriate timeout.  You can specify a
timeout for TCP sessions too, which will close the session after the specified
number of seconds of inactivity.  In this case, DNS sessions are not really
ever reused, so a short timeout is appropriate.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/contrib/for-demos/proxy-to-service/README.md?pixel)]()

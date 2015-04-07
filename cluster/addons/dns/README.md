# DNS in Kubernetes

Kubernetes offers a DNS cluster addon, which most of the supported environments
enable by default.  We use [SkyDNS](https://github.com/skynetservices/skydns)
as the DNS server, with some custom logic to slave it to the kubernetes API
server.

## What things get DNS names?
The only objects to which we are assigning DNS names are Services.  Every
Kubernetes Service is assigned a virtual IP address which is stable as long as
the Service exists (as compared to Pod IPs which can change over time due to
crashes or scheduling changes).  This maps well to DNS, which has a long
history of clients that, on purpose or on accident, do not respect DNS TTLs
(see previous remark about Pod IPs changing).

## How do I find the DNS server?
The DNS server itself runs as a Kubernetes Service.  This gives it a stable IP
address.  When you run the SkyDNS service, you want to assign a static IP to use for
the Service.  For example, if you assign the DNS Service IP as `10.0.0.10`, you
can configure your kubelet to pass that on to each container as a DNS server.

Of course, giving services a name is just half of the problem - DNS names need a
domain also.  This implementation uses a configurable local domain, which can
also be passed to containers by kubelet as a DNS search suffix.

## How do I configure it?
The easiest way to use DNS is to use a supported kubernetes cluster setup,
which should have the required logic to read some config variables and plumb
them all the way down to kubelet.

Supported environments offer the following config flags, which are used at
cluster turn-up to create the SkyDNS pods and configure the kubelets.  For
example, see `cluster/gce/config-default.sh`.

```shell
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1
```

This enables DNS with a DNS Service IP of `10.0.0.10` and a local domain of
`kubernetes.local`, served by a single copy of SkyDNS.

If you are not using a supported cluster setup, you will have to replicate some
of this yourself.  First, each kubelet needs to run with the following flags
set:

```
--cluster_dns=<DNS service ip>
--cluster_domain=<default local domain>
```

Second, you need to start the DNS server ReplicationController and Service. See
the example files ([ReplicationController](skydns-rc.yaml.in) and
[Service](skydns-svc.yaml.in)), but keep in mind that these are templated for
Salt.  You will need to replace the `{{ <param> }}` blocks with your own values
for the config variables mentioned above.  Other than the templating, these are
normal kubernetes objects, and can be instantiated with `kubectl create`.

## How does it work?
SkyDNS depends on etcd for what to serve, but it doesn't really need all of
what etcd offers (at least not in the way we use it).  For simplicty, we run
etcd and SkyDNS together in a pod, and we do not try to link etcd instances
across replicas.  A helper container called [kube2sky](kube2sky/) also runs in
the pod and acts a bridge between Kubernetes and SkyDNS.  It finds the
Kubernetes master through the `kubernetes-ro` service (via environment
variables), pulls service info from the master, and writes that to etcd for
SkyDNS to find.

## Known issues
Kubernetes installs do not configure the nodes' resolv.conf files to use the
cluster DNS by default, because that process is inherently distro-specific.
This should probably be implemented eventually.

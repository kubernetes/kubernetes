# DNS in Kubernetes
[SkyDNS](https://github.com/skynetservices/skydns) can be configured
to automatically run in a Kubernetes cluster.

## What things get DNS names?
The only objects to which we are assigning DNS names are Services.  Every
Kubernetes Service is assigned a virtual IP address which is stable as long as
the Service exists.  This maps well to DNS, which has a long history of clients
that, on purpose or on accident, do not respect DNS TTLs.

## How do I find the DNS server?
The DNS server itself runs as a Kubernetes Service.  This gives it a stable IP
address.  When you run the SkyDNS service, you can assign a static IP to use for
the Service.  For example, if you assign `DNS_SERVER_IP` (see below) as
10.0.0.10, you can configure your docker daemon with the flag `--dns 10.0.0.10`.

Of course, giving services a name is just half of the problem - DNS names need a
domain also.  This implementation uses the variable `DNS_DOMAIN` (see below).
You can configure your docker daemon with the flag `--dns-search`.

## How do I configure it?
The following environment variables are used at cluster startup to create the SkyDNS pods and configure the kubelets. If you need to, you can reconfigure your provider as necessary (e.g. `cluster/gce/config-default.sh`):

```shell
ENABLE_CLUSTER_DNS=true
DNS_SERVER_IP="10.0.0.10"
DNS_DOMAIN="kubernetes.local"
DNS_REPLICAS=1
```

## How does it work?
SkyDNS depends on etcd for what to serve, but it doesn't really need all of
what etcd offers in the way we use it.  For simplicty, we run etcd and SkyDNS
together in a pod, and we do not try to link etcd instances across replicas.  A
helper container called `kube2sky` also runs in the pod and acts a bridge
between Kubernetes and SkyDNS.  It finds the Kubernetes master through the
`kubernetes-ro` service, it pulls service info from the master, and it writes
that to etcd for SkyDNS to find.

## Known issues
Kubernetes installs do not configure the nodes' resolv.conf files to use the
cluster DNS by default, because that process is inherently distro-specific.
This should probably be implemented eventually.

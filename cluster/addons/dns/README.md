# DNS in Kubernetes
This directory holds an example of how to run
[SkyDNS](https://github.com/skynetservices/skydns) in a Kubernetes cluster.

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

## How do I run it?
The first thing you have to do is substitute the variables into the
configuration.  You can then feed the result into `kubectl`.

```shell
DNS_SERVER_IP=10.0.0.10
DNS_DOMAIN=kubernetes.local
DNS_REPLICAS=2

sed -e "s/{DNS_DOMAIN}/$DNS_DOMAIN/g" \
    -e "s/{DNS_REPLICAS}/$DNS_REPLICAS/g" \
    ./cluster/addons/dns/skydns-rc.yaml.in \
    | ./cluster/kubectl.sh create -f -

sed -e "s/{DNS_SERVER_IP}/$DNS_SERVER_IP/g" \
    ./cluster/addons/dns/skydns-svc.yaml.in \
    | ./cluster/kubectl.sh create -f -
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

# DNS Integration with SkyDNS
Since Kubernetes services changed to assign a single IP address to each service, it is
now possible to use DNS to resolve a DNS name directly to a Kubernetes service, which 
would then use Kubernetes' proxy to connect to an appropriate pod running the application 
pointed to by the service definition.

## How it Works
Version 2.0.1a of [SkyDNS](https://github.com/skynetservices/skydns) added a change that
allows it to poll the Kubernetes API looking for changes to the service definitions.  Newly
added services are published in SkyDNS, and removed services are deleted from SkyDNS's 
internal registry.  

### Concrete Example
If you run the Guestbook example in the Kubernetes repository, you'll end up with a service
called `redismaster`.  If you were also running SkyDNS with the `-kubernetes=true` flag and
`-master=http://my.kubernetes.master:8080` you would immediately be able to run queries against
the SkyDNS server for the `redismaster` service.  By default, SkyDNS is authoratative for the
domain `skydns.local`, so a query to the SkyDNS server requesting redismaster.skydns.local will
return the IP Address of the `redismaster` service.

## Configuration
SkyDNS allows you to change the domain name that it will resolve by passing in a domain on the 
command line using `-domain=mydomain.com` or by setting an environment variable `SKYDNS_DOMAIN`.

If you change the Docker daemon on your Kubernetes minions to use SkyDNS for domain name resolution,
your pods will all be able to connect to services via DNS instead of using environment variables
or other configuration methods.  To change Docker to use SkyDNS resolution, add `--dns=ip.of.skydns.server`
to the Docker startup command.
```
docker -d --dns=10.2.0.5 ...
```

SkyDNS uses the etcd instance in Kubernetes as its storage backend, which means that you can run
multiple SkyDNS daemons if you wish to have more than one resolver on your cluster.  You could run
a SkyDNS instance on each node in your Kubernetes cluster, and set Docker to use 127.0.0.1 as the 
DNS resolver.

## Starting SkyDNS in a Kubernetes Cluster
At a minimum, you need to provide the `-kubernetes` flag, and the `-master=http://my.kubernetes.master.ip:8080` 
flag when you start SkyDNS.  You may also wish to use `-domain=mydomain.com` to change the domain that
SkyDNS resolves.

SkyDNS can act as your external resolver, too.  If you set your domain to use the external IP address of 
the server running SkyDNS and bind SkyDNS to listen on all interfaces, SkyDNS will serve DNS for
your domain.  You could then use a mixture of manually created hosts in SkyDNS and Kubernetes service
resolution to serve your various DNS endpoints.  A simple example might be to run a Wordpress pod in Kubernetes
and create a service called `blog` in Kubernetes.  Then external DNS requests to `blog.mydomain.com` will
automatically resolve to the service proxy and be forwarded to the pods running Wordpress.

Full documentation of the SkyDNS server is in the [SkyDNS repository](https://github.com/skydnsservices/skydns)
and abbreviated information is available by typing `skydns --help`.

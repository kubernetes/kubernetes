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
[here](http://releases.k8s.io/release-1.0/docs/user-guide/services-firewalls.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Services and Firewalls

Many cloud providers (e.g. Google Compute Engine) define firewalls that help prevent inadvertent
exposure to the internet.  When exposing a service to the external world, you may need to open up
one or more ports in these firewalls to serve traffic.  This document describes this process, as
well as any provider specific details that may be necessary.


### Google Compute Engine

When using a Service with `spec.type: LoadBalancer`, the firewall will be
opened automatically.  When using `spec.type: NodePort`, however, the firewall
is *not* opened by default.

Google Compute Engine firewalls are documented [elsewhere](https://cloud.google.com/compute/docs/networking#firewalls_1).

You can add a firewall with the `gcloud` command line tool:

```console
$ gcloud compute firewall-rules create my-rule --allow=tcp:<port>
```

**Note**
There is one important security note when using firewalls on Google Compute Engine:

as of kubernmetes v1.0.0, GCE firewalls are defined per-vm, rather than per-ip
address.  This means that when you open a firewall for a service's ports,
anything that serves on that port on that VM's host IP address may potentially
serve traffic.  Note that this is not a problem for other Kubernetes services,
as they listen on IP addresses that are different than the host node's external
IP address.

Consider:
   * You create a Service with an external load balancer (IP Address 1.2.3.4)
     and port 80
   * You open the firewall for port 80 for all nodes in your cluster, so that
     the external Service actually can deliver packets to your Service
   * You start an nginx server, running on port 80 on the host virtual machine
     (IP Address 2.3.4.5).  This nginx is **also** exposed to the internet on
     the VM's external IP address.

Consequently, please be careful when opening firewalls in Google Compute Engine
or Google Container Engine.  You may accidentally be exposing other services to
the wilds of the internet.

This will be fixed in an upcoming release of Kubernetes.

### Other cloud providers

Coming soon.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/user-guide/services-firewalls.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

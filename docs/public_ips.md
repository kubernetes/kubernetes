# Load balancing & publicly accessible services

This document describes how kubernetes services are made accessible outside the cluster.

There are two main mechanisms:

* The service can be exposed through a load-balancer managed by k8s.
* k8s can also expose services without an associated load-balancer; the service could be reached via DNS SRV records,
or by a load-balancer configured by a helper process, or even by a client using the k8s API directly for discovery.
_(new functionality)_

## Public ports

To be exposed outside the k8s cluster, the service must be on known ports on externally reachable
IPs.  For version 1 public services, k8s exposes each public service port on a common port on each
node.

## Important scenarios

The names in bold are the names we will use in the API.

1. **cluster** The service is reachable from within the cluster on a virtual IP.  _(Existing functionality)_
1. **public** The service is exposed on a set of nodes, on a set of ports.  _(New functionality)_.
 For V1 services are exposed on all nodes.  The service endpoints would be discoverable via the k8s API,
 and helper processes could then expose that information via DNS or configure external systems
 (e.g. physical load balancers)  The intention is that this service should then be reachable from outside
 the k8s cluster; a cloud-provider would automatically configure firewall rules (or equivalents)
 so that a public service is reachable from the public Internet.
1. **loadbalancer** The service is exposed via a k8s-managed load-balancer.  _(Existing functionality)_

Expected future functionality:

1. **namespace** The service is only accessible to pods belonging to the same namespace.  This is similar
to _cluster_, but with additional firewall rules.
1. **dmz** The service is accessible to pods that are in the DMZ.  This is similar to _cluster_,
except that pods marked 'DMZ' are prevented from accessing _cluster_ services, and can only access services explicitly marked
as dmz-reachable that are in their namespace.

Currently these rules (and even the planned rules) nest very nicely (namespace < dmz < cluster < public < loadbalancer),
so for V1 we will have a single visibility setting.  In future, we may convert this to a variably-typed argument 
(StringOrStringList, or StringOrVisibility) or just add a richer field.

This does mean that loadbalancer => public; i.e. a service that is exposed through a load-balancer can also be
reached at the nodes, bypassing the load-balancer.  It does not seem that any of the cloud load balancers
currently offer any functionality that would make this problematic (e.g. if this were a way to bypass
DDoS protection).  Further, because cloud load-balancers may have a financial and performance cost, there are
legitimate reasons where a user might choose to send traffic _not_ through the load-balancer.  Future versions of
the API will likely have a richer model.

## API sketch

```
ServiceSpec:
  - portalIP
  - createExternalLoadBalancer  // deprecated
  - publicIPs[]  // deprecated
  - visibility   // "cluster" or "public" or "loadbalancer"
  - ports[] {
      - protocol
      - port
      - hostPort  // if visibility==cluster, must be omitted or zero.
                  // otherwise: if == 0, try to assign a port.  if != 0, try to assign the specified port.
   }
```

```
ServiceStatus:
  - publicIPs[] // mirrors publicIPs in Service today, replaces them in future
```

We have to differentiate between the load-balancer as seen externally (by the k8s end-user) vs. as seen internally
(e.g. by kube-proxy).
For example, if we were using AWS subnets to identify ELBs (which we're not), the external load-balancer name
that the k8s-end-user cares about would be something like `myelbname.amazonaws.com`, the internal ip-ranges would be
something like `10.244.250.192/30`.  In the case of GCE these are the same, and it looks like we don't need to track
an internal name at the moment, but we do not want to bake this assumption into the API.  So publicIPs is _external_.

publicIPs in the ServiceSpec will continue to work as it does today. The user can specify a publicIP to use an existing
load-balancer/VIP (in practice this only works for OpenStack today).  This will be populated once the load-balancer
is created.  This can be deprecated in future (once the OpenStack plugin does dynamic VIP creation).

Each cloud-provider can keep track of the mapping from service to load-balancer:
* GCE uses the name
* AWS can use name and/or tags
* OpenStack uses the name
* Rackspace can use the name and/or metadata

Because the load-balancer IPs can be recreated, we are able to expose the load-balancer information through the status.
The ServiceStatus will be extended to include publicIPs.  This will be the user-facing IP addresses or hostnames.

If we require additional state in future for kube-proxy routing (for example internal load-balancer IPs),
that will be added to a new Status field.

By putting this information on Status, an external load-balancer provisioner can use the same mechanism for securely
updating Status as kube-proxy/kubelet.



## Creation flow

### visibility: public

1. User `POST`s `/api/v1/namespaces/foo/services/bar` `Service{ Spec{ visibility: public, ports: [{port: 80, hostPort: 10080}] } }`
1. API server verifies that port 10080 can be assigned, accepts and persists
1. kube-proxy wakes up and sets iptables to receive on hostPort 10080

### visibility: loadbalancer (GCE)

1. User `POST`s `/api/v1/namespaces/foo/services/bar` `Service{ Spec{ visibility: loadbalancer, ports: [{port: 80}] } }`
1. API server assigns a hostPort, accepts and persists
1. LB Controller wakes up, allocates a load-balancer, and `POST`s `/api/v1/namespaces/foo/services/bar` `Service{ Status{ publicIps: [ "1.2.3.4" ] }}`
1. Service REST sets `Service.Status.publicIps = [ "1.2.3.4" ]`
1. kube-proxy wakes up and sets iptables to receive on 1.2.3.4 (for visiblity==loadbalancer), and on hostPort (for visibility==public)

### visibility: loadbalancer (AWS)

1. User `POST`s `/api/v1/namespaces/foo/services/bar` `Service{ Spec{ visibility: loadbalancer, ports: [{port: 80}] } }`
1. API server assigns a hostPort, accepts and persists
1. LB Controller wakes up, allocates a load-balancer, and `POST`s `/api/v1/namespaces/foo/services/bar` `Service{ Status{ publicIps: [ "mylb.amazonaws.com" ] }}`
1. Service REST sets `Service.Status.publicIps = [ "mylb.amazonaws.com" ]`
1. kube-proxy wakes up and sets iptables to receive on hostPort (for visibility==public & visibility==loadbalancer)


(These are just examples; it is valid to specify a hostPort with visiblity==loadbalancer, or to omit hostPort
with visibility==public)

In order to verify that the port can be assigned, the API server must check that the port is within a permitted range,
and that it is not already assigned to another service.  This is similar to the allocation of internal cluster-IPs.
The error codes & messages (in the case of port conflicts, out-of-range ports or port-exhaustion) will be analogous
to the equivalent IP address errors.

Advanced clouds (e.g. GCE) can support pure load-balancing without requiring a public port assignment, but
for V1 visibility==loadbalancer incorporates visibility==public, so one will be assigned.

## New flags

We will add a flag `public_service_ports` to the apiserver, which will determine the port-range allowed for public services.

The syntax is `--public_service_ports=30000-32767` to specify the range 30000-32767 _inclusive_.

(Inside the code we will likely use a different representation)

The default value is `--public_service_ports=30000-32767`.  This excludes most well-known ports, and certainly excludes
the <1024 ports.  We should try to keep the minion using ports <1024.

TODO: figure out which >1024 ports are needed on the minion

A null range can be specified (e.g. `--public_service_ports=0-0`).  This will have the effect of preventing any public port allocations.

In future, we could allow disjoint port ranges (`--public_service_ports=10000-20000,30000-32767`), but not for V1

The range should not overlap the range of ports which kube-proxy uses internally for _cluster_ services;
that range is set in /proc/sys/net/ipv4/ip_local_port_range.  We cannot know on the apiserver what the
setting is on the kube-proxy machines (the minions), so we cannot prevent the user setting an overlapping range.
The salt scripts could check this.  If an overlapping range is set, conflicts will randomly happen between
cluster services and public services on some kube-proxy instances, and those kube-proxies will log errors and
fail to serve that service.  This will likely result in degraded service availability.

## API evolution

The main changes:

* we deprecate createExternalLoadBalancer, and replace it with visibility
* we move publicIPs to the Status
* we add a field hostPort to each service port
 
Mappings:

`visibility` is mappable:

```
new.visibility = old.createExternalLoadBalancer ? "loadbalancer" : "cluster"
```

```
old.createExternalLoadBalancer = (new.visibility == 'loadbalancer')
```

`publicIPs` will be copied to the Spec if set on the Status, and from the Spec to the Status if not set on the Status.
Once each cloud provider sets the Status, we will stop copying from the Spec to the Status.

For v1 final:
* we should remove publicIPs from the Spec, and require that this information is retrieved through the Status.
However, we would need to change the openstack code to auto-allocate a VIP.
* we should remove createExternalLoadBalancer.

## kube-proxy pseudocode


```
for each service {
    for each service.port {
      // listen on cluster
      listen on <internalMagicIp>:0
      proxyPort = figureOutTheRandomPortWeJustGotAssigned()
      openPortal(service.portalIP, port.port, proxyPort)

      for each service.publicIP {  // deprecated
        openPublicPortal(publicIP, port.port, proxyPort) // uses cloud-provider to decide srcip vs dstip (gclb vs rax)
      }

      if service.spec.visibility == "loadbalancer" {
        if (port.hostPort != 0) {
          openLoadBalancerPortals(service, port.port, port.hostPort) // uses cloud-provider to figure out correct iptables
        }
      } else if service.spec.visibility == "public" {
        if (port.hostPort != 0) {
          openPublicPortal(<publicMagicIp>, port.port, port.hostPort)
        }
      }
   }
}
```


## kube-proxy iptables

The current implementation of kube-proxy allocates ports for cluster services, randomly from the range
in /proc/sys/net/ipv4/ip_local_port_range.  By allocating ports from a non-overlapping range specified by
the public_service_ports argument, we should avoid conflicts.

(There may be changes we can make to avoid this in future, but they are orthogonal to the public-ports work)

It is possible that there would be an unexpected port-conflict meaning we could not open the port.  This will be treated
as an unexpected failure (like out-of-fds); it will be logged and kube-proxy will continue and likely retry.  This should
only happen if we run new services on the minions or specify public_service_ports overlapping with ip_local_port_range.


## Sidecar-pod implementations

In future, we may allow sidecar-pods to be responsible for configuring load-balancers; these could be outside
of the main k8s codebase.  We would likely configure the apiserver to permit load-balancer creation, assign a port
(probably), but k8s would not create the load balancer.  kube-proxy would create the service as a public service.
The loadbalancer sidecar-pod would create the load-balancer and
call the API to report events.

We might be able to move the cloud-provider load-balancer functionality into a sidecar-pod, but not for V1.

Likely examples of load-balancer sidecar-pods:
* hardware load-balancers (Cisco, Big5 etc)
* 3rd party hosted load-balancers/CDNs (Akamai, CloudFlare)

We could also have sidecar-pods that configure DNS, either k8s-hosted or 3rd-party.  For now, these could be triggered
by labels on the service (i.e. would not require API support).  In future, having learned what is useful, we can
add any required functionality to k8s itself.

## Provider specifics

GCE load balancer: traffic dest-ip is load-balancer IP, does not perform port-translation. 
_visibility==loadbalancer_ will not set the hostPort (so that the hostPort could be set if we _also_ opened the service publicly).
The publicIP reported is the load-balancer IP.

AWS load balancer: an AWS load balancer has a dynamic set of IP addresses, users must CNAME their DNS name to it (or
use Route53).  It forwards traffic, so the source IP is not meaningful; the dest IP is the instance IP.  We will
therefore use public-ports for services that have visibility==loadbalancer on AWS.
_visibility==loadbalancer_ will set the hostPort, but will configure AWS security groups so that the port is only accessible
to the load balancer.
The publicIP reported is the load-balancer's DNS name.

OpenStack load balancer: same as GCE.  (The openstack provider currently uses the VIP support, although OpenStack also
supports a layer-7 load-balancer, which would be more like AWS).

Rackspace load balancer: Not entirely clear, but looks to support port translation, and I think it preserves the source address.
It may only support a single port per load-balancer. Probably a combination of AWS & GCE (?)

Digital Ocean load-balancer: DO does not support load balancers, nor does it support floating IPs.  It has two networks: the public
network and an internal backend network (which is not secure as it is shared across all DO customers,
but should be faster than the public network).  public services will be supported; those
service will be exposed on both networks.  A DO cloudprovider would presumably expose both the internal & external IPs
for the minions in the Node status.  A DNS-publishing sidecar-pod could expose public services.  A sidecar that configures
an external load-balancer would also be useful.  The best we can do seems to be to expose the service directly and
drop traffic that hits failed nodes, or rely on a hosted load-balancer; both would be configured through sidecar-pods.

Bare metal: There is currently no 'baremetal' cloud provider, so we will just support public services.  Likely
a sidecar-pod would be configured for configuration of a hardware load-balancer / switch.

## Future evolution (post-V1)

V1 makes two big simplifying assumptions for public-ports: we bind public services to _every_ IP, and we assign the _same_ port across all
IPs.  Both these simplifications could be eliminated in future versions of the API:

* To listen on a subset of IP addresses, we would just bind selectively, and then publish those addresses through the API.  We could
use publicIPs for this.
* We could then associate a port with each IP address.  However, while having a port-per-IP is easier to implement
than a cluster-wide port-assignment, this would likely be less useful for end-users: firewalls often think
in terms of ports, and some load-balancers (like AWS ELB) assume that back-ends all have the same port.  To allow more than 65k
ports in a cluster therefore, we would probably want to partition our IP addresses, and assign ports within each partition.

We will likely want a richer model for load-balancer lifecycle, where we can see the progress of creation and errors/events are surfaced more nicely.

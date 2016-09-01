<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Service externalName

Author: Tim Hockin (@thockin), Rodrigo Campos (@rata), Rudi C (@therc)

Date: August 2016

Status: Implementation in progress

# Goal

Allow a service to have a CNAME record in the cluster internal DNS service. For
example, the lookup for a `db` service could return a CNAME that points to the
RDS resource `something.rds.aws.amazon.com`. No proxying is involved.

# Motivation

There were many related issues, but we'll try to summarize them here. More info
is on GitHub issues/PRs: #13748, #11838, #13358, #23921

One motivation is to present as native cluster services, services that are
hosted externally. Some cloud providers, like AWS, hand out hostnames (IPs are
not static) and the user wants to refer to these services using regular
Kubernetes tools. This was requested in bugs, at least for AWS, for RedShift,
RDS, Elasticsearch Service, ELB, etc.

Other users just want to use an external service, for example `oracle`, with dns
name `oracle-1.testdev.mycompany.com`, without having to keep DNS in sync, and
are fine with a CNAME.

Another use case is to "integrate" some services for local development. For
example, consider a search service running in Kubernetes in staging, let's say
`search-1.stating.mycompany.com`. It's running on AWS, so it resides behind an
ELB (which has no static IP, just a hostname). A developer is building an app
that consumes `search-1`, but doesn't want to run it on their machine (before
Kubernetes, they didn't, either). They can just create a service that has a
CNAME to the `search-1` endpoint in staging and be happy as before.

Also, Openshift needs this for "service refs". Service ref is really just the
three use cases mentioned above, but in the future a way to automatically inject
"service ref"s into namespaces via "service catalog"[1] might be considered. And
service ref is the natural way to integrate an external service, since it takes
advantage of native DNS capabilities already in wide use.

[1]: https://github.com/kubernetes/kubernetes/pull/17543

# Alternatives considered

In the issues linked above, some alternatives were also considered. A partial
summary of them follows.

One option is to add the hostname to endpoints, as proposed in
https://github.com/kubernetes/kubernetes/pull/11838. This is problematic, as
endpoints are used in many places and users assume the required fields (such as
IP address) are always present and valid (and check that, too). If the field is
not required anymore or if there is just a hostname instead of the IP,
applications could break. Even assuming those cases could be solved, the
hostname will have to be resolved, which presents further questions and issues:
the timeout to use, whether the lookup is synchronous or asynchronous, dealing
with DNS TTL and more. One imperfect approach was to only resolve the hostname
upon creation, but this was considered not a great idea. A better approach
would be at a higher level, maybe a service type.

There are more ideas described in #13748, but all raised further issues,
ranging from using another upstream DNS server to creating a Name object
associated with DNSs.

# Proposed solution

The proposed solution works at the service layer, by adding a new `externalName`
type for services. This will create a CNAME record in the internal cluster DNS
service. No virtual IP or proxying is involved.

Using a CNAME gets rid of unnecessary DNS lookups. There's no need for the
Kubernetes control plane to issue them, to pick a timeout for them and having to
refresh them when the TTL for a record expires. It's way simpler to implement,
while solving the right problem. And addressing it at the service layer avoids
all the complications mentioned above about doing it at the endpoints layer.

The solution was outlined by Tim Hockin in
https://github.com/kubernetes/kubernetes/issues/13748#issuecomment-230397975

Currently a ServiceSpec looks like this, with comments edited for clarity:

```
type ServiceSpec struct {
    Ports []ServicePort

    // If not specified, the associated Endpoints object is not automatically managed
    Selector map[string]string

    // "", a real IP, or "None".  If not specified, this is default allocated.  If "None", this Service is not load-balanced
    ClusterIP string

    // ClusterIP, NodePort, LoadBalancer.  Only applies if clusterIP != "None"
    Type ServiceType

    // Only applies if clusterIP != "None"
    ExternalIPs []string
    SessionAffinity ServiceAffinity

    // Only applies to type=LoadBalancer
    LoadBalancerIP string
    LoadBalancerSourceRanges []string
```

The proposal is to change it to:

```
type ServiceSpec struct {
    Ports []ServicePort

    // If not specified, the associated Endpoints object is not automatically managed
+   // Only applies if type is ClusterIP, NodePort, or LoadBalancer.  If type is ExternalName, this is ignored.
    Selector map[string]string

    // "", a real IP, or "None".  If not specified, this is default allocated.  If "None", this Service is not load-balanced.
+   // Only applies if type is ClusterIP, NodePort, or LoadBalancer.  If type is ExternalName, this is ignored.
    ClusterIP string

-   // ClusterIP, NodePort, LoadBalancer.  Only applies if clusterIP != "None"
+   // ExternalName, ClusterIP, NodePort, LoadBalancer.  Only applies if clusterIP != "None"
    Type ServiceType

+   // Only applies if type is ExternalName
+   ExternalName string

    // Only applies if clusterIP != "None"
    ExternalIPs []string
    SessionAffinity ServiceAffinity

    // Only applies to type=LoadBalancer
    LoadBalancerIP string
    LoadBalancerSourceRanges []string
```

For example, it can be used like this:

```
apiVersion: v1
kind: Service
metadata:
  name: my-rds
spec:
  ports:
  - port: 12345
type: ExternalName
externalName: myapp.rds.whatever.aws.says
```

There is one issue to take into account, that no other alternative considered
fixes, either: TLS. If the service is a CNAME for an endpoint that uses TLS,
connecting with the Kubernetes name `my-service.my-ns.svc.cluster.local` may
result in a failure during server certificate validation. This is acknowledged
and left for future consideration. For the time being, users and administrators
might need to ensure that the server certificates also mentions the Kubernetes
name as an alternate host name.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-external-name.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

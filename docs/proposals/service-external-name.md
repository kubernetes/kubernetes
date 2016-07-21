<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->

<!-- BEGIN STRIP_FOR_RELEASE -->

<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">
<img src="http://kubernetes.io/kubernetes/img/warning.png" alt="WARNING"
     width="25" height="25">

<h2>PLEASE NOTE: This document applies to the HEAD of the source tree</h2>

If you are using a released version of Kubernetes, you should
refer to the docs that go with that version.

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Service externalName

Author: Tim Hockin (@thockin), Rodrigo Campos (@rata)

Date: July 2016

Status: Waiting LGTM

# Goal

Allow a service to have a CNAME record in the cluster internal DNS service. For
example, a `db` service can have a CNAME to `something.rds.aws.amazon.com`
pointing to an RDS resource.

# Motivation

There were tons of issues about this, I'd try to summarize motivation here. More
info is on github issues/PRs: #13748, #11838, #13358, #23921

One motivation is to present as cluster services, services that are hosted by
some provider. Some cloud providers, like AWS, give a hostname (IPs are not
static) and the user wants to refer using regular kubernetes tools to these
services. This was asked in bugs, at least for AWS in RedShift, RDS,
Elasticsearch Service, ELB, etc.

Some others just want an external service, for example "oracle", with dns name
"oracle-1.testdev.mycompany.com", without having to keep DNS in sync, and just
want a CNAME.

Another use case is to "integrate" some services to local development. For
example, I have a search service running in kubernetes in staging, let's say
`search-1.stating.mycompany.com`. Let's say it's on AWS, so it's behind an ELB
(which doesn't have a static IP, it has a hostname). I'm building an app that
consumes `search-1` and I don't want to run it on my local PC (before kubernetes
I didn't). I can just create a service that has a CNAME to the `search-1`
endpoint in staging and be happy like I was before.

Also, openshift needs this for "service refs". Service ref is really just the
three use cases mentioned above, but in the future a way to automatically inject
"service ref"s into namespaces via "service catalog"[1] might be considered. And
service ref is the natural way to integrate an external service, since it takes
advantage of native DNS capabilities already in wide use.

[1]: https://github.com/kubernetes/kubernetes/pull/17543

# Alternatives considered

In the issues linked above, there is also some alternatives considered. I will
try to sum them up here, but the list might not be complete or have all the
discussion.

One option is to add the hostname to endpoints. This was proposed in:
https://github.com/kubernetes/kubernetes/pull/11838. This is problematic as
endpoints are used in tons of places and users assume the required fields (like
IP, for example) are always present and a valid IP (and check that). If the
field is not required anymore, they can break, or if there is a hostname instead
of the IP, they can break too.  But assuming that can be solved, it was also
discussed that the hostname will have to be resolved, with a timeout, sync/async
and the DNS entry has a TTL and presents other problems. One option, not
perfect, was to only resolve the hostname on creation. But this was considered
not a good idea. The best thing was to do this at a higher level, maybe a
service type.

There are more ideas on how to approach this problem on #13748, but all pointed
to some problem. Ranging from using another upstream DNS server to creating a
Name object assoaciated with DNSs.

# Proposed solution

The proposed solution is to add this at the service layer by adding a new
`externalName` type to the service. This will create a CNAME record in the
internal cluster DNS service.

Using a CNAME avoids having to do the lookup, decide a timeout for it, and
having to lookup for it when the TTL expires. It's way simpler to implement,
while solving the right problem. And doing it at the service layer avoids all
the problem discussed with doing the change at the endpoints layer.

The proposed solution is the one by Tim Hockin here:
https://github.com/kubernetes/kubernetes/issues/13748#issuecomment-230397975

Currently a ServiceSpec looks like this (comments stripped down):

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

So, the proposal is to change it to:

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

So, for example, it can be used like this:

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

There is one thing to take into account (that no other alternative considered
fixes either): TLS. If the service is a CNAME for an endpoint that uses TLS,
connecting with another name may fail cert validation. This is acknowledged and
left for future consideration.


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-external-name.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

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

<!-- TAG RELEASE_LINK, added by the munger automatically -->
<strong>
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.3/docs/proposals/templates.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

Idling and Unidling
===================

In certain cluster scenarios, it is valuable to "idle" an application, scaling
it down to zero and marking it to "wake up" when needed.  Because the
application had no pods, it would not consume compute resources, but would
still be available when needed.  Use cases for such functionality include
public clouds, in which many low traffic or "fire-and-forget" applications
may be created, and other situations where a high application density is
desired.

There are two parts to this proposal: "idling", the act of figuring out what
can be scaled down to zero to be woken up later, and "unidling", the mechanism
by which the "waking up" actually occurs.

For the purposes of this document, we will consider an "application" to be some
scalable resource or resources (e.g. ReplicationControllers or Deployments)
with a service or services which point to them.

Idling
------

There are two parts of idling -- discovery of applications which meet some
criteria for idling, and the actual marking of the services as idled. In the
initial idling implementation, these will both be done via a `kubectl` command.

Intially, the criteria supported by the `kubectl idle` commad would be limitted
simply be a threshold for a given metric (say, network traffic on services).
Given the criteria, the command would query Heapster's historical access API
(Oldtimer) to check to see which services were at or below the threshold, and
would return the resulting list.  The admin could then apply custom logic to
filter out any applications that should never be idled.

Then, the admin could pass that list back to the `idle` command (or simply idle
everything in a namespace, or any service matching a label selector).  The
command would discover which scalable resources corresponded to the given
services (by examining the endpoints, and the corresponding pods, for target
and creator references, respectively), and would apply annotations to each
endpoints object denoting when it was idled, and which scalable resources
should be woken up upon unidling (as well as their current replica count).  The
corresponding scalable resources would then be scaled down to zero replicas.

See below for options on recording the information.

Unidling
--------

Once an endpoints object is marked as "idled" and has no populated subsets, it
would be a candidate for unidling.  Unidling would involve a special "hybrid"
proxy: non-idled services would use a normal version of the proxy (either
userspace or iptables), while idled services would use a special variant of the
userspace proxy (henceforth referred to as the "unidling proxy").

A service is considered idled by the hybrid proxy only if its endpoints object
has no populated subets *and* is marked with an idled-at time.  If either of
these conditions is false, the "normal" proxy will be used.

The unidling proxy is a variant of the userspace proxy with a custom
ProxySocket.  When it receives TCP connections, it accepts them, and then
triggers unidling (see below).  Once it sees that subsets have become
populated, it triggers proxies held connections similarly to the normal
userspace proxy.  New connections after this point will use the "normal" proxy,
since the hybrid proxy will switch over as soon as it receives the endpoints
updated showing populated subsets.  Existing connects will continue to be
proxied in userspace until they close.  Only a certain number of connections
will be held open -- old connections eventually will be closed if too many
connections come in or the the unidling is not sucessful.

For UDP connections, the UDP packets are dropped (since UDP does not guarantee
delivery), and the signal is sent out.  Once endpoints appear, the hybrid proxy
will activate the "normal" proxy for that service, which will proxy the UDP
traffic as normal.

Kubernetes-aware applications which do not normally communicate though the
service proxy (such as the OpenShift router) would need to observe that
a particular service was idled, and either send its first connection
through the service proxy or emit the unidling signal itself.

A controller loop would then watch for the signal, load up the endpoints
object, read the annotation containing the the scalable resources to unidle,
scale the resources back up to their previous replica counts, and remove the
idling annotations.

The controller's logic looks like this, and is designed with the goal of not
idling something and then getting into a state where we never unidle:

1. Notice the Signal[Endpoints ep]
2. Check to make sure ep is idled, and the signal occurred after ep was idled
3. Unmarshal the list of targets to scale, and for each:
    1. Get the object's scale (on not found, remove it from list list)
    2. if not idle, continue
    3. scale the object up to the original replica count (on not found, remove
       from the list
4. For all targets still on the list, remove them from the targets annotation
5. If the annotation is empty, also remove the idled-at annotation
6. Update the endpoints object (to save the new annotation values)

In the current WIP implementation in OpenShift, the signal is a "NeedPods"
Event with the related object set as the service upon which the traffic
occurred.  Alternatively, the signal could also be an annotation on the
endpoints object, which would more in line with the declarative model of
Kubernetes.

Caveats
-------

- because idling deals with two different objects (the endpoints and the
  scalable resource), it is possible for a user to manually scale up a scalable
  resource and leave the idling annotations on the endpoints object.

- if a connection comes in while a pod is shutting down, but the pod still
  accepts the connection, that connection will not end up triggering unidling
  (and can potentially appear to be immediately closed, if the pod does not
  gracefully shut down).

- if a connection comes in between marking the service as idle and scaling the
  scalable resources down to zero, that connection will not trigger unidling
  (however, the other order causes a period in which we do not hold
  connections, so connections could just get rejected).

- traffic could come in between discovery and idling.  This is not too big of a
  problem, since any future traffic will trigger unidling.

Recording Idling State and the Scale Subresource
------------------------------------------------

In the current prototype in OpenShift, use annotations on the endpoints to
record "canonical" information, and then manually apply purely
informational annotations to the scalable resource in certain cases.  The
annotations applied to the endpoints object look like:

- `idling.kubernetes.io/idled-at`: the idled-at time.  Used to indicate that
  idling is active, and to ensure that that we don't accidentally mix up older
  signals to unidle with newer idling.

- `idling.kubernetes.io/unidle-targets`: an JSON-encoded array of targets
  paired with their former replica counts.  Used to determine which scalable
  resources to unidle.  Looks like `[{"name": "somedeployment", "kind":
  "Deployment", "replicas": 2}]`.

The annotations applied to the scalable resources (bypassing the scale
object) are important for clients that wish to see idling information, but
do not wish to perform the associations between endpoints and scalable
objects themselves.  They look like:

- `idling.kubernetes.io/idled-at`: the same as above
- `idling.kubernetes.io/previous-scale`: records the previous scale before
  idling

However, manually applying the annotations to scalable objects means we
need to know how to deal with individual resources, instead of just
dealing with the scale subresource, and makes it harder for clients to
manually unidle (for example, a web dashboard might want to have an
"unidle" button to allow individuals to manually unidle applications).

For this reason, it could be adventageous to modify the scale subresource
(this could be prototyped by passing a certain prefix of annotations, such
as `scale.kubernetes.io/`, through the scale subresource to the underlying
objects).  The following fields would be involved:

- `previous-scale`: this would record the previous scale of the scalable
  resource, and would be set/determined automatically.

- `scale-reason`: this would be a machine-readable reason for the most
  recent scale.  Any scale via the scale subresource without explicitly
  setting this field would cause it to clear.

- `scale-time`: this would record the last time a scale operation was
  performed, and would be set/determined automatically.

These would work as follows: the idler would set the `scale-reason` field
to `Idled`.  The unidling controller would then be able to confirm whether
or not a scalable resource was idled by looking at the `scale-reason` and
`scale-time` (if `scale-time` is before the endpoints' `idled-at` value,
then idling is still in process.  Otherwise we're either idled or unidled,
depending on the `scale-reason).  When it came time to unidle, the
unidling controller could look at the `previous-scale` annotation to get
the target scale for unidling, instead of storing it on the endpoints
object.

Furthermore, other clients could simply use the scale subresource
to unidle by inspecting the `previous-scale` field.  Any scale which did
not explicitly set the `scale-reason` would clear it, and thus the
unidling controller could notice the change in endpoints, fetch the
corresponding scale subresource objects, check if the `scale-reason` was
cleared, and clean up the annotations on the endpoints.

Future Improvements and Alternatives
------------------------------------

### An Idler Object ###

To bring idling more in line with autoscaling, an idler object could be
created.  This nicely exposes to the user that idling is enabled, but has the
downside that it requires an idler object to be created for all applications or
namespaces.  Since it is expected that admins will want to create some sort of
custom logic around what gets idled, this could be a bit tricky.

Alternatively, the HorizontalPodAutoscaler object could be modified to also
have an idling option.  Since autoscaling is somewhat related to idling,
this makes sense.  However, there could be cases where idling is desired
but autoscaling is not (for example, if you decide to apply idling
globally across the cluster, certain ReplicationControllers might require
a fixed scale instead of just being scalable to an arbitrary number of
replicas), and thus having an autoscaling object with no autoscaling
enabled would be counterintuitive.

### Netfilter Packet Queues ###

The proposed unidling proxy works great for short-lived connections like HTTP
requests -- one or two requests go through the userspace proxy, and the rest go
through the iptables proxy.  However, for long-lived TCP connections that
initial connection would be stuck going through the userspace proxy for all of
its traffic.  It was suggested that it may be possible to alleviate this issue
by using libnetfilter queues.  This complicates the logic somewhat, but could
bear investigation as a potential future improvement.

### Adding an "unidle" endpoint ###

It would probably be valuable to have an "unidle" subresource for the endpoints
or service object.  This would make it easier for Kubernetes-aware applications
to trigger unidling themselves.

### Adding a last-scale-reason field to the Scale object ###

Recording the last scale reason (either some identifier like "autoscaler" or
"idler", or nothing to indicate a manual scale) would easily allow tools
to see if a scalable resource was idled without having to examine all
endpoints objects in namespace, and without requiring the idling components
to circumvent the scale endpoint.  This is proposed in #29143.

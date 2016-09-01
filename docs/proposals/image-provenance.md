<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->


# Overview

Organizations wish to avoid running "unapproved" images.

The exact nature of "approval" is beyond the scope of Kubernetes, but may include reasons like:

 - only run images that are scanned to confirm they do not contain vulnerabilities
 - only run images that use a "required" base image
 - only run images that contain binaries which were built from peer reviewed, checked-in source
   by a trusted compiler toolchain.
 - only allow images signed by certain public keys.

 - etc...

Goals of the design include:
* Block creation of pods that would cause "unapproved" images to run.
* Make it easy for users or partners to build "image provenance checkers" which check whether images are "approved".
  *  We expect there will be multiple implementations.
* Allow users to request an "override" of the policy in a convenient way (subject to the override being allowed).
  * "overrides" are needed to allow "emergency changes", but need to not happen accidentally, since they may
    require tedious after-the-fact justification and affect audit controls.

Non-goals include:
* Encoding image policy into Kubernetes code.
* Implementing objects in core kubernetes which describe complete policies for what images are approved.
  * A third-party implementation of an image policy checker could optionally use ThirdPartyResource to store its policy.
* Kubernetes core code dealing with concepts of image layers, build processes, source repositories, etc.
  * We expect there will be multiple PaaSes and/or de-facto programming environments, each with different takes on
    these concepts.  At any rate, Kuberenetes is not ready to be opinionated on these concepts.
* Sending more information than strictly needed to a third-party service.
  * Information sent by Kubernetes to a third-party service constitutes an API of Kubernetes, and we want to
    avoid making these broader than necessary, as it restricts future evolution of Kubernetes, and makes
    Kubernetes harder to reason about.  Also, excessive information limits cache-ability of decisions.  Caching
    reduces latency and allows short outages of the backend to be tolerated.


Detailed discussion in [Ensuring only images are from approved sources are run](
https://github.com/kubernetes/kubernetes/issues/22888).

# Implementation

A new admission controller will be added.  That will be the only change.

## Admission controller

An `ImagePolicyWebhook` admission controller will be written.  The admission controller examines all pod objects which are
created or updated.  It can either admit the pod, or reject it.  If it is rejected, the request sees a `403 FORBIDDEN`

The admission controller code will go in `plugin/pkg/admission/imagepolicy`.

There will be a cache of decisions in the admission controller.

If the apiserver cannot reach the webhook backend, it will log a warning and either admit or deny the pod.
A flag will control whether it admits or denys on failure.
The rationale for deny is that an attacker could DoS the backend or wait for it to be down, and then sneak a
bad pod into the system.  The rationale for allow here is that, if the cluster admin also does
after-the-fact auditing of what images were run (which we think will be common), this will catch
any bad images run during periods of backend failure.  With default-allow, the availability of Kubernetes does
not depend on the availability of the backend.

# Webhook Backend

The admission controller code in that directory does not contain logic to make an admit/reject decision.  Instead, it extracts
relevant fields from the Pod creation/update request and sends those fields to a Backend (which we have been loosely calling "WebHooks"
in Kubernetes).  The request the admission controller sends to the backend is called a WebHook request to distinguish it from the
request being admission-controlled.  The server that accepts the WebHook request from Kubernetes is called the "Backend"
to distinguish it from the WebHook request itself, and from the API server.

The whole system will work similarly to the [Authentication WebHook](
https://github.com/kubernetes/kubernetes/pull/24902
) or the [AuthorizationWebHook](
https://github.com/kubernetes/kubernetes/pull/20347).

The WebHook request can optionally authenticate itself to its backend using a token from a `kubeconfig` file.

The WebHook request and response are JSON, and correspond to the following `go` structures:

```go
// Filename: pkg/apis/imagepolicy.k8s.io/register.go
package imagepolicy

// ImageReview checks if the set of images in a pod are allowed.
type ImageReview struct {
 	unversioned.TypeMeta
 
 	// Spec holds information about the pod being evaluated
 	Spec ImageReviewSpec
 
 	// Status is filled in by the backend and indicates whether the pod should be allowed.
 	Status ImageReviewStatus
 }
 
// ImageReviewSpec is a description of the pod creation request.
type ImageReviewSpec struct {
	// Containers is a list of a subset of the information in each container of the Pod being created.
	Containers []ImageReviewContainerSpec
	// Annotations is a list of key-value pairs extracted from the Pod's annotations.
	// It only includes keys which match the pattern `*.image-policy.k8s.io/*`.
	// It is up to each webhook backend to determine how to interpret these annotations, if at all.
	Annotations map[string]string
	// Namespace is the namespace the pod is being created in.
	Namespace string
}

// ImageReviewContainerSpec is a description of a container within the pod creation request.
type ImageReviewContainerSpec struct {
	Image string
	// In future, we may add command line overrides, exec health check command lines, and so on.
}

// ImageReviewStatus is the result of the token authentication request.
type ImageReviewStatus struct {
	// Allowed indicates that all images were allowed to be run.
	Allowed bool
	// Reason should be empty unless Allowed is false in which case it
	// may contain a short description of what is wrong.  Kubernetes
	// may truncate excessively long errors when displaying to the user.
	Reason string
}
```

## Extending with Annotations

All annotations on a Pod that match `*.image-policy.k8s.io/*` are sent to the webhook.
Sending annotations allows users who are aware of the image policy backend to send
extra information to it, and for different backends implementations to accept
different information.

Examples of information you might put here are

- request to "break glass" to override a policy, in case of emergency.
- a ticket number from a ticket system that documents the break-glass request
- provide a hint to the policy server as to the imageID of the image being provided, to save it a lookup

In any case, the annotations are provided by the user and are not validated by Kubernetes in any way.  In the future, if an annotation is determined to be widely
useful, we may promote it to a named field of ImageReviewSpec.

In the case of a Pod update, Kubernetes may send the backend either all images in the updated image, or only the ones that
changed, at its discretion.

## Interaction with Controllers

In the case of a Deployment object, no image check is done when the Deployment object is created or updated.
Likewise, no check happens when the Deployment controller creates a ReplicaSet.  The check only happens
when the ReplicaSet controller creates a Pod.  Checking Pod is necessary since users can directly create pods,
and since third-parties can write their own controllers, which kubernetes might not be aware of or even contain
pod templates.

The ReplicaSet, or other controller, is responsible for recognizing when a 403 has happened
(whether due to user not having permission due to bad image, or some other permission reason)
and throttling itself and surfacing the error in a way that CLIs and UIs can show to the user.

Issue [22298](https://github.com/kubernetes/kubernetes/issues/22298) needs to be resolved to
propagate Pod creation errors up through a stack of controllers.

## Changes in policy over time

The Backend might change the policy over time.  For example, yesterday `redis:v1` was allowed, but today `redis:v1` is not allowed
due to a CVE that just came out (fictional scenario).  In this scenario:
.

- a newly created replicaSet will be unable to create Pods.
- updating a deployment will be safe in the sense that it will detect that the new ReplicaSet is not scaling
  up and not scale down the old one.
- an existing replicaSet will be unable to create Pods that replace ones which are terminated.  If this is due to
  slow loss of nodes, then there should be time to react before significant loss of capacity.
- For non-replicated things (size 1 ReplicaSet, PetSet), a single node failure may disable it.
- a node rolling update will eventually check for liveness of replacements, and would be throttled if
  in the case when the image was no longer allowed and so replacements could not be started.
- rapid node restarts will cause existing pod objects to be restarted by kubelet.
- slow node restarts or network partitions will cause node controller to delete pods and there will be no replacement

It is up to the Backend implementor, and the cluster administrator who decides to use that backend, to decide
whether the Backend should be allowed to change its mind.  There is a tradeoff between responsiveness
to changes in policy, versus keeping existing services running.  The two models that make sense are:

- never change a policy, unless some external process has ensured no active objects depend on the to-be-forbidden
  images.
- change a policy and assume that transition to new image happens faster than the existing pods decay.

## Ubernetes

If two clusters share an image policy backend, then they will have the same policies.

The clusters can pass different tokens to the backend, and the backend can use this to distinguish
between different clusters.

## Image tags and IDs

Image tags are like: `myrepo/myimage:v1`.

Image IDs are like: `myrepo/myimage@sha256:beb6bd6a68f114c1dc2ea4b28db81bdf91de202a9014972bec5e4d9171d90ed`.
You can see image IDs with `docker images --no-trunc`.

The Backend needs to be able to resolve tags to IDs (by talking to the images repo).
If the Backend resolves tags to IDs, there is some risk that the tag-to-ID mapping will be
modified after approval by the Backend, but before Kubelet pulls the image.  We will not address this
race condition at this time.

We will wait and see how much demand there is for closing this hole. If the community demands a solution,
we may suggest one of these:

1.  Use a backend that refuses to accept images that are specified with tags, and require users to resolve to IDs
    prior to creating a pod template.
   - [kubectl could be modified to automate this process](https://github.com/kubernetes/kubernetes/issues/1697)
   - a CI/CD system or templating system could be used that maps IDs to tags before Deployment modification/creation.
1. Audit logs from kubelets to see image IDs were actually run, to see if any unapproved images slipped through.
1. Monitor tag changes in image repository for suspicious activity, or restrict remapping of tags after initial application.

If none of these works well, we could do the following:

- Image Policy Admission Controller adds new field to Pod, e.g. `pod.spec.container[i].imageID` (or an annotation).
  and kubelet will enforce that both the imageID and image match the image pulled.

Since this adds complexity and interacts with imagePullPolicy, we avoid adding the above feature initially.

### Caching

There will be a cache of decisions in the admission controller.
TTL will be user-controllable, but default to 1 hour for allows and 30s for denies.
Low TTL for deny allows user to correct a setting on the backend and see the fix
rapidly.  It is assumed that denies are infrequent.
Caching allows permits RC to scale up services even during short unavailability of the webhook backend.
The ImageReviewSpec is used as the key to the cache.

In the case of a cache miss and timeout talking to the backend, the default is to allow Pod creation.
Keeping services running is more important than a hypothetical threat from an un-verified image.


### Post-pod-creation audit

There are several cases where an image not currently allowed might still run.  Users wanting a
complete audit solution are advised to also do after-the-fact auditing of what images
ran.  This can catch:

- images allowed due to backend not reachable
- images that kept running after policy change (e.g. CVE discovered)
- images started via local files or http option of kubelet
- checking SHA of images allowed by a tag which was remapped

This proposal does not include post-pod-creation audit.

## Alternatives considered

### Admission Control on Controller Objects

We could have done admission control on Deployments, Jobs, ReplicationControllers, and anything else that creates a Pod, directly or indirectly.
This approach is good because it provides immediate feedback to the user that the image is not allowed.  However, we do not expect disallowed images
to be used often.  And controllers need to be able to surface problems creating pods for a variety of other reasons anyways.

Other good things about this alternative are:

- Fewer calls to Backend, once per controller rather than once per pod creation. Caching in backend should be able to help with this, though.
- End user that created the object is seen, rather than the user of the controller process.  This can be fixed by implementing `Impersonate-User` for controllers.

Other problems are:

- Works only with "core" controllers.  Need to update admission controller if we add more "core" controllers.  Won't work with "third party controllers", e.g. how we open-source distributed systems like hadoop, spark, zookeeper, etc running on kubernetes.  Because those controllers don't have config that can be "admission controlled", or if they do, schema is not known to admission controller, have to "search" for pod templates in json.  Yuck.
- How would it work if user created pod directly, which is allowed, and the recommended way to run something at most once.

### Sending User to Backend

We could have sent the username of the pod creator to the backend.  The username could be used to allow different users to run
different categories of images.  This would require propagating the username from e.g. Deployment creation, through to
Pod creation via, e.g. the `Impersonate-User:` header.  This feature is [not ready](https://github.com/kubernetes/kubernetes/issues/27152).
 When it is, we will re-evaluate adding user as a field of `ImagePolicyRequest`.

### Enforcement at Docker level

Docker supports plugins which can check any container creation before it happens.  For example the [twistlock/authz](https://github.com/twistlock/authz)
Docker plugin can audit the full request sent to the Docker daemon and approve or deny it.  This could include checking if the image is allowed.

We reject this option because:
- it requires all nodes to be able to configured with how to reach the Backend, which complicates node setup.
- it may not work with other runtimes
- propagating error messages back to the user is more difficult
- it requires plumbing additional information about requests to nodes (if we later want to consider `User` in policy).

### Policy Stored in API

We decided to store policy about what SecurityContexts a pod can have in the API, via PodSecurityPolicy.
This is because Pods are a Kubernetes object, and the Policy is very closely tied to the definition of Pods,
and grows in step as the Pods API grows.

For Image policy, the connection is not as strong.  To Kubernetes API, and Image is just a string, and it
does not know any of the image metadata, which lives outside the API.

Image policy may depend on the Dockerfile, the source code, the source repo, the source review tools,
vulnerability databases, and so on.  Kubernetes does not have these as built-in concepts or have plans to add
them anytime soon.

### Registry whitelist/blacklist

We considered a whitelist/blacklist of registries and/or repositories. Basically, a prefix match on image strings.
 The problem of approving images would be then pushed to a problem of controlling who has access to push to a
trusted registry/repository.  That approach is simple for kubernetes.  Problems with it are:

- tricky to allow users to share a repository but have different image policies per user or per namespace.
- tricky to do things after image push, such as scan image for vulnerabilities (such as Docker Nautilus), and have those results considered by policy
- tricky to block "older" versions from running, whose interaction with current system may not be well understood.
- how to allow emergency override?
- hard to change policy decision over time.

We still want to use rkt trust, docker content trust, etc for any registries used. We just need additional
image policy checks beyond what trust can provide.

### Send every Request to a Generic Admission Control Backend

Instead of just sending a subset of PodSpec to an Image Provenance backed, we could have sent every object
that is created or updated (or deleted?) to one or ore Generic Admission Control Backends.

This might be a good idea, but needs quite a bit more thought.  Some questions with that approach are:
It will not be a generic webhook. A generic webhook would need a lot more discussion:

- a generic webhook needs to touch all objects, not just pods. So it won't have a fixed schema.  How to express this in our IDL?  Harder to write clients
  that interpret unstructured data rather than a fixed schema.  Harder to version, and to detect errors.
- a generic webhook client needs to ignore kinds it does not care about, or the apiserver needs to know which backends care about which kinds.  How
  to specify which backends see which requests.  Sending all requests including high-rate requests like events and pod-status updated, might be
  too high a rate for some backends?

Additionally, just sending all the fields of just the Pod kind also has problems:
- it exposes our whole API to a webhook backend without giving us (the project) any chance to review or understand how it is being used.
- because we do not know which fields of an object are inspected by the backend, caching of decisions is not effective. Sending fewer fields allows caching.
- sending fewer fields makes it possible to rev the version of the webhook request slower than the version of our internal obejcts (e.g. pod v2 could still use imageReview v1.)
probably lots more reasons.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/image-provenance.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

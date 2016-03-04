<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# Service Accounts

## Motivation

Processes in Pods may need to call the Kubernetes API.  For example:
  - scheduler
  - replication controller
  - node controller
  - a map-reduce type framework which has a controller that then tries to make a dynamically determined number of workers and watch them
  - continuous build and push system
  - monitoring system

They also may interact with services other than the Kubernetes API, such as:
  - an image repository, such as docker -- both when the images are pulled to start the containers, and for writing
    images in the case of pods that generate images.
  - accessing other cloud services, such as blob storage, in the context of a large, integrated, cloud offering (hosted
    or private).
  - accessing files in an NFS volume attached to the pod

## Design Overview

A service account binds together several things:
  - a *name*, understood by users, and perhaps by peripheral systems, for an identity
  - a *principal* that can be authenticated and [authorized](../admin/authorization.md)
  - a [security context](security_context.md), which defines the Linux Capabilities, User IDs, Groups IDs, and other
    capabilities and controls on interaction with the file system and OS.
  - a set of [secrets](secrets.md), which a container may use to
    access various networked resources.

## Design Discussion

A new object Kind is added:

```go
type ServiceAccount struct {
    TypeMeta   `json:",inline" yaml:",inline"`
    ObjectMeta `json:"metadata,omitempty" yaml:"metadata,omitempty"`

    username string
    securityContext ObjectReference // (reference to a securityContext object)
    secrets []ObjectReference // (references to secret objects
}
```

The name ServiceAccount is chosen because it is widely used already (e.g. by Kerberos and LDAP)
to refer to this type of account.  Note that it has no relation to Kubernetes Service objects.

The ServiceAccount object does not include any information that could not be defined separately:
  - username can be defined however users are defined.
  - securityContext and secrets are only referenced and are created using the REST API.

The purpose of the serviceAccount object is twofold:
  - to bind usernames to securityContexts and secrets, so that the username can be used to refer succinctly
    in contexts where explicitly naming securityContexts and secrets would be inconvenient
  - to provide an interface to simplify allocation of new securityContexts and secrets.
These features are explained later.

### Names

From the standpoint of the Kubernetes API, a `user` is any principal which can authenticate to Kubernetes API.
This includes a human running `kubectl` on her desktop and a container in a Pod on a Node making API calls.

There is already a notion of a username in Kubernetes, which is populated into a request context after authentication.
However, there is no API object representing a user.  While this may evolve, it is expected that in mature installations,
the canonical storage of user identifiers will be handled by a system external to Kubernetes.

Kubernetes does not dictate how to divide up the space of user identifier strings.  User names can be
simple Unix-style short usernames, (e.g. `alice`), or may be qualified to allow for federated identity (
`alice@example.com` vs `alice@example.org`.)  Naming convention may distinguish service accounts from user
accounts (e.g. `alice@example.com` vs `build-service-account-a3b7f0@foo-namespace.service-accounts.example.com`),
but Kubernetes does not require this.

Kubernetes also does not require that there be a distinction between human and Pod users.  It will be possible
to setup a cluster where Alice the human talks to the Kubernetes API as username `alice` and starts pods that
also talk to the API as user `alice` and write files to NFS as user `alice`.  But, this is not recommended.

Instead, it is recommended that Pods and Humans have distinct identities, and reference implementations will
make this distinction.

The distinction is useful for a number of reasons:
  - the requirements for humans and automated processes are different:
    - Humans need a wide range of capabilities to do their daily activities. Automated processes often have more narrowly-defined activities.
    - Humans may better tolerate the exceptional conditions created by expiration of a token.  Remembering to handle
      this in a program is more annoying.  So, either long-lasting credentials or automated rotation of credentials is
      needed.
    - A Human typically keeps credentials on a machine that is not part of the cluster and so not subject to automatic
      management.  A VM with a role/service-account can have its credentials automatically managed.
  - the identity of a Pod cannot in general be mapped to a single human.
    - If policy allows, it may be created by one human, and then updated by another, and another, until its behavior cannot be attributed to a single human.

**TODO**: consider getting rid of separate serviceAccount object and just rolling its parts into the SecurityContext or
Pod Object.

The `secrets` field is a list of references to /secret objects that an process started as that service account should
have access to be able to assert that role.

The secrets are not inline with the serviceAccount object.  This way, most or all users can have permission to `GET /serviceAccounts` so they can remind themselves
what serviceAccounts are available for use.

Nothing will prevent creation of a serviceAccount with two secrets of type `SecretTypeKubernetesAuth`, or secrets of two
different types.  Kubelet and client libraries will have some behavior, TBD, to handle the case of multiple secrets of a
given type (pick first or provide all and try each in order, etc).

When a serviceAccount and a matching secret exist, then a `User.Info` for the serviceAccount and a `BearerToken` from the secret
are added to the map of tokens used by the authentication process in the apiserver, and similarly for other types.  (We
might have some types that do not do anything on apiserver but just get pushed to the kubelet.)

### Pods

The `PodSpec` is extended to have a `Pods.Spec.ServiceAccountUsername` field.  If this is unset, then a
default value is chosen.  If it is set, then the corresponding value of `Pods.Spec.SecurityContext` is set by the
Service Account Finalizer (see below).

TBD: how policy limits which users can make pods with which service accounts.

### Authorization

Kubernetes API Authorization Policies refer to users.  Pods created with a `Pods.Spec.ServiceAccountUsername` typically
get a `Secret` which allows them to authenticate to the Kubernetes APIserver as a particular user.  So any
policy that is desired can be applied to them.

A higher level workflow is needed to coordinate creation of serviceAccounts, secrets and relevant policy objects.
Users are free to extend Kubernetes to put this business logic wherever is convenient for them, though the
Service Account Finalizer is one place where this can happen (see below).

### Kubelet

The kubelet will treat as "not ready to run" (needing a finalizer to act on it) any Pod which has an empty
SecurityContext.

The kubelet will set a default, restrictive, security context for any pods created from non-Apiserver config
sources (http, file).

Kubelet watches apiserver for secrets which are needed by pods bound to it.

**TODO**: how to only let kubelet see secrets it needs to know.

### The service account finalizer

There are several ways to use Pods with SecurityContexts and Secrets.

One way is to explicitly specify the securityContext and all secrets of a Pod when the pod is initially created,
like this:

**TODO**: example of pod with explicit refs.

Another way is with the *Service Account Finalizer*, a plugin process which is optional, and which handles
business logic around service accounts.

The Service Account Finalizer watches Pods, Namespaces, and ServiceAccount definitions.

First, if it finds pods which have a `Pod.Spec.ServiceAccountUsername` but no `Pod.Spec.SecurityContext` set,
then it copies in the referenced securityContext and secrets references for the corresponding `serviceAccount`.

Second, if ServiceAccount definitions change, it may take some actions.
**TODO**: decide what actions it takes when a serviceAccount definition changes.  Does it stop pods, or just
allow someone to list ones that are out of spec?  In general, people may want to customize this?

Third, if a new namespace is created, it may create a new serviceAccount for that namespace.  This may include
a new username (e.g. `NAMESPACE-default-service-account@serviceaccounts.$CLUSTERID.kubernetes.io`), a new
securityContext, a newly generated secret to authenticate that serviceAccount to the Kubernetes API, and default
policies for that service account.
**TODO**: more concrete example.  What are typical default permissions for default service account (e.g. readonly access
to services in the same namespace and read-write access to events in that namespace?)

Finally, it may provide an interface to automate creation of new serviceAccounts.  In that case, the user may want
to GET serviceAccounts to see what has been created.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/service_accounts.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

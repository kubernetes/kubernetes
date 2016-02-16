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
The latest release of this document can be found
[here](http://releases.k8s.io/release-1.1/docs/proposals/service-catalog.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

# Abstract

Kubernetes has Services but they are scoped per namespace, and (assuming eventual support for RBAC)
you may not want everyone to be able to see and use all the services in your namespace.

This proposal is divided into multiple phases, as some items are potentially immediately actionable,
while others are dependent on features that don't yet exist in Kubernetes.

For Phase 1, this proposal is about:

- associating additional configuration/secret data with Services
- adding a new Service Catalog resource into which users can publish entries that others can easily
  find and consume
- making it easier to link Services (and their associated configuration data) to deployable
  resources (Deployments, ReplicationControllers, Pods, etc.)

Once Phase 1 has been implemented, Phase 2 will address:

- adding support for dynamic provisioning of resources from the catalog

# Phase 1

## Use cases

### Shared development database

A development team is working on an application that uses a database. The IT department manages the
database (i.e., it lives off-cluster). All developers share the same credentials to access the
database, but these credentials are managed by IT. Rather than having each developer create his or
her own `Service` and `Secret` to connect to the database, IT creates a "db-app-xyz" `Service` and a
"db-app-xyz" `Secret` in the "info-tech" namespace. IT also has lots of other `Service` resources in
their namespace and they don't want to expose all of them to the development team. Therefore, they
publish "db-app-xyz" to a service catalog. To use this service, a developer searches for it in the
service catalog and adds it to their namespace.

### Easy linking of a Service to a Deployable resource

A user has developed an application that uses a database. The user doesn't want to hard-code
the URL to the database, because that would be brittle and require rebuilding the application if the
database coordinates change. Instead, the user wants to be able to create the application and link
it to a database service. A typical sequence for this use case in a Platform as a Service (PaaS) might
look like:

1. User asks PaaS to create a new application A
2. User asks PaaS to add a database to application A
3. User pushes code to application A
4. PaaS builds and deploys application A
5. Application A starts up, connecting to the database by looking up a well-known, predefined
   environment variable for the database's URL which was injected and set by the PaaS

## Service configuration data

The only data currently associated with a service is a list of zero or more endpoints. Many services
also have configuration data required to use them. These include things such as logins, passwords,
and additional connection parameters. To tie configuration data to a service, we propose adding
information to the `Service` type to be able to express a relationship between the `Service` and its
relevant configuration data. This could be accomplished by adding new field(s) or by using
annotations; these would contain references to the relevant associated resources.  Potential
configuration data target reference types include `Secrets` and `ConfigMap` (but to avoid limiting
to just those types, we can use `LocalObjectReferences`).

Note: some of the associated data such as protocol and path (a.k.a. context root) are probably
better suited as annotations or fields on the `Service` itself, and not as references. These items
are most likely used by the cluster itself in some way, or potentialy by a UI; e.g., to show a link
to the service if the UI knows it supports HTTP.

Specifying references to `Secrets`, `ConfigMaps`, etc. by itself does not accomplish much; to be
useful, that data needs to be available to the processes executing in containers. More on this
process is below in the sections on service claims and linking.

## Service catalogs

A service catalog is a listing of published service entries. For Phase 1, the only valid type of
entry is a `Service`. This allows us to support a service that points to an external database that
is deployed outside of the cluster, as well as services that act as load balancers to a selected set
of backend pods via the kube-proxy.

The catalog is not meant to include every service in the cluster. Instead, it should contain those
services that users wish to highlight and make available to other users. For example, your namespace
might contain "etcd", "etcd-discovery", and "postgresql" services, and the only one you want to
share with others is the postgresql service.

One way of implementing a service catalog could be to create a namespace to represent a shared
catalog. This could be feasible, but it has some limitations:

- If we eventually add support for additional types of entries in the catalog (templates, service
  brokers), users/UIs would have to query multiple resource types to retrieve the entire catalog
- Assuming we add functionality to associate `Secrets` with a `Service`, a `Pod` running in
  namespace "foo" wouldn't be able to access the secret in namespace "servicecatalog"
- Users have to know the name of the namespace(s) that contain shared services

In light of this, we believe an actual `ServiceCatalog` resource, combined with the claiming and
provisioning mechanisms described below, offers richer functionality to end users than a shared
namespace.

Because the service catalog is meant to span namespaces, it should not be a namespaced resource. We
should support multiple service catalogs, as different groups using the cluster might want to offer
their own catalogs. We may also want to consider a "default" catalog that is displayed in the
absence of multiple catalogs, to simplify how users interact with service catalogs.

### Publishing to a catalog

Users should be able to publish entries to a service catalog. We have considered two options for
this.

#### Option 1: annotate the `Service`

If you want to have a `Service` included in a service catalog, add the following annotations:

- kubernetes.io/catalog.destination = "default"
- kubernetes.io/catalog.entry.name = "awesome-etcd"
- kubernetes.io/catalog.entry.description = "an etcd service"

This example publishes a `Service` into the "default" `ServiceCatalog` with the entry name
"awesome-etcd".

In order to appear in the listing of a service catalog's entries, we most likely will want to use a
reflector/cache of some sort. Any time a `Service` is modified, the reflector inspects the updated
resource and adds/updates/removes it from the cache. The implementation of this cache is potentially
not trivial, as it needs to take into account security policy decisions such as "is UserX allowed to
publish to this catalog?".

**Pros**

- does not require an API change to the core `Service` type
- information about the catalog entry is stored on the `Service` itself, making it easy to see if
  the service has been published or not

**Cons**

- requires a cache for efficiency
- checking security policy decisions could be difficult
- users won't receive any immediate indication that publishing an entry to a catalog was denied
  (but they could potentially see the denial in an annotation)

#### Option 2: add `ServiceCatalogEntry` resource

If you want to have a `Service` included in a service catalog, create a new `ServiceCatalogEntry`
resource, such as:

```yaml
apiVersion: catalog/v1beta1
kind: ServiceCatalogEntry
metadata:
  name: awesome-etcd
catalog: default
description: an etcd service
targetObjectReference:
  apiVersion: v1
  kind: Service
  namespace: foo
  name: etcd
```

This example publishes a `Service` called "etcd" from the namespace "foo" into the "default"
`ServiceCatalog` with the entry name "awesome-etcd".

**Pros**

- is a strongly-typed resource, with specific fields
- users can get immediate feedback that publishing an entry was accepted or denied

**Cons**

- checking security policy decisions could be difficult

### Viewing a catalog

Users should be able to list the entries in a service catalog. Users should be able to select an
entry and "consume" it. We call this consumption "claiming" a service from a service catalog.

### Claiming a service

When you want to use an entry from the service catalog, you create a `ServiceClaim` that references
the desired entry. A controller processes new claims for admission. This determines if the user who
created the claim is allowed to consume the entry from the catalog. This decision can be flexible:
it could be automated based on policy, or it could support manual intervention and workflow.

Once the claim has been admitted, a controller performs the provisioning process. The controller
creates a new service in the user's namespace that "points" to the original service.  Additionally,
the controller clones each of the original service's referenced resources to the user's namespace.

#### Communicating with the claimed service

##### Option 1: CNAME

Given an original service "foo.bob.svc.cluster.local", and a service created via claim "bar" in
namespace "alice", DNS requests for "bar.alice.svc.cluster.local" resolve via CNAME to
"foo.bob.svc.cluster.local".

**Cons**:

- Doesn't work for TLS hostname verification. A request to "bar.alice.svc.cluster.local" will
  receive a certificate for "foo.bob.svc.cluster.local", which will cause hostname verification to
  fail.

##### Option 2: ????

(I need some assistance from the community on this topic)

## Linking services

Claiming a service catalog entry only creates resources in the user's namespace. If all you need is
a service and its DNS entry, this may be sufficient for your pods to function. But if you need the
configuration data injected into your pod, it would be nice to make that easier to do.

We want to add the ability to link a service to a deployable resource such as a Deployment. This
could look something like:

    kubectl link svc/postgresql deployment/web
    service "postgresql" linked to deployemnt "web"
    configdata "postgresql-options" linked to deployment "web" as a volume
    secret "postgresql-credentials" linked to deployment "web" as a volume

This command would automatically inject the ConfigData and Secret objects as volumes into the
Deployment. This could be flexible as well, allowing you instead to expose these items as
environment variables.

In the example above, the volumes could potentially be mounted as:

/var/run/kubernetes.io/links/configdata/postgresql-options
/var/run/kubernetes.io/links/secrets/postgresql-credentials

Note: linking is orthogonal to a service catalog and service claims.

There are several outstanding questions in this area:

- How do we best represent the intent that a resource such as a `Deployment` is linked to a `Service`?
	- 1 suggestion is to add a `ServiceLinks []ObjectReference` to `PodSpec`
- When do we process the link to create the environment variables and/or volumes?
- Is it acceptable to define naming conventions for these volumes?
	- How can an application/container author develop for these conventions most easily?
- How can we easily support a use case where we need to:
	1. Mount a file
	2. Create an environment variable with a specific name that points to that file (e.g., to access the GCP APIs via standard client libraries; see https://developers.google.com/identity/protocols/application-default-credentials)


## Cross-namespace networking

If the cluster has multi-tenant network isolation enabled, then a pod in namespace A won't be able
to talk to a service in namespace B. We should look into ways to automatically manipulate the
isolation rules to "poke holes" when claims are provisioned, to make the desired connectivity work,
and to "unpoke" when the connectivity is no longer needed.

## Proposed API changes

- Add annotation/field to `Service` to specify associated resources
- Add `ServiceCatalog` types in a separate `catalog` API group

```go
type ServiceCatalog struct {
  unversioned.TypeMeta
  ObjectMeta

  // Not sure exactly what we want/need here
  // Could include who is allowed to publish
}

type ServiceCatalogList struct {
  unversioned.TypeMeta
  ListMeta

  Items []ServiceCatalog
}
```

- Add `ServiceCatalogEntry` types

```go
type ServiceCatalogEntry struct {
  unversioned.TypeMeta
  ObjectMeta

  // The ServiceCatalog to which this entry belongs
  Catalog string
  // This entry's description
  Description string
  // The resource to which this entry refers
  TargetObjectReference ObjectReference
  // The entry's type for provisioning. Different types may be handled by different provisioners to support distinct functionality. Defaults to "reference" as that is all that phase 1 supports, but needed to support phase 2.
  Type string
}

type ServiceCatalogEntryList struct {
  unversioned.TypeMeta
  ListMeta

  Items []ServiceCatalogEntry
}
```

- Add `ServiceClaim` types

```go
type ServiceClaim struct {
  unversioned.TypeMeta
  ObjectMeta

  Spec ServiceClaimSpec
  Status ServiceClaimStatus
}

type ServiceClaimSpec struct {
  // Specifies the desired service catalog
  ServiceCatalogName string
  // Specifies the entry to claim
  Entry ObjectReference
}

type ServiceClaimStatus struct {
  State ServiceClaimState
  // An array of the items created when this claim was provisioned
  ProvisionedItems []LocalObjectReference
}

type ServiceClaimState string

const (
  ServiceClaimState ServiceClaimStateNew = "New"
  ServiceClaimState ServiceClaimStateAdmitted = "Admitted"
  ServiceClaimState ServiceClaimStateRejected = "Rejected"
  ServiceClaimState ServiceClaimStateProvisioned = "Provisioned"
)

type ServiceClaimList struct {
  unversioned.TypeMeta
  ListMeta

  Items []ServiceClaim
}
```

# Phase 2

## Use cases

### Template-based provisioning

A user creates a "template" (e.g. if something similar to [OpenShift
Templates](https://docs.openshift.org/latest/dev_guide/templates.html) exists) that makes it easy to
create everything needed to spin up a new PostgreSQL database (customizable username/password,
`Service`, `Deployment`, etc.). The user wants to share only this template in a service catalog so
others can find it and use it, while keeping other templates in the namespace private.

### Custom provisioning

The IT department manages an off-cluster database. Each developer wanting to access the database is
required to use a unique username and password to access the database. Additionally, each developer
accesses a unique tablespace that no other team members can access. The IT department used to create
database accounts and tablespaces by hand in response to individual developer requests.

Moving forward, IT wants to automate the process. They create a web service that implements a
"service broker" API (something similar to [Cloud
Foundry's](http://docs.cloudfoundry.org/services/api.html#api-overview)). They add an entry to the
service catalog for their database service, pointing at their web service. When a developer consumes
the entry from the catalog, IT's web service is contacted, resulting in a new username, password,
and tablespace.

## Types of service catalog entries

An entry in the service catalog has a "type", which indicates the behavior that occurs when it is
consumed by a user. We have thought of the following potential types:

- Reference: "I want to use this entry as-is, including its configuration data"
	- This is what Phase 1 implements; namely, provisioning a `Service` and its related resources in the destination namespace
- Template: "I want to create items from the specified template"
- ServiceBroker: "I want the creation to be goverened by some other entity that implements the 'service broker' HTTP interface"

We see the type as an arbitrary `string`; one or more controllers could run to process each type, performing whatever logic is appropriate to fulfill the specific type in question.

Additional fulfillment types are possible as long as there is a controller that is handling them.

# TODOs

- Figure out security
	- How do we determine who can publish a service to a given catalog?
	- How do we determine who can see/use a specific service from a given catalog?
- How do you keep the claimed services in sync with the sources?
	- e.g. if the source service references a secret, and the secret's content changes
- What does it mean to "unlink"?
- What happens if you delete a claim - does it cascade?


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-catalog.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

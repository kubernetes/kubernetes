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

## Abstract

A new catalog concept is proposed for sharing reusable recipes for services,
the containers that back them, and configuration data associated with them.
Users will be able to publish recipes to the catalog and browse the catalog
for recipes to use.

## Motivation

Users don’t like reinventing the wheel. As an example, most users would prefer
to be able to search for and use something like a database template to run their
own database over doing the work necessary to create a custom solution.  There
are a number of pieces attendant to such an effort: a Service to provide a
stable network endpoint for the database, a Deployment to back that endpoint
with database containers, and possibly a ConfigMap and or Secret to store
configuration information and credentials about the database.

If you assume that most or all namespaces and the resources in them are
private, we need a way for users to be able to share resources with others.
Role-based access control (RBAC) could help, as it allows users to control who
can access resources in a namespace, but that pertains to existing resources.
If you want create a pre-canned way of running something like a database and
share that with others, that’s not RBAC against existing resources; that’s
something more akin to publishing a "recipe" to a searchable catalog.

This document describes a new “Catalog” concept for sharing reusable
resources.

## Use cases

1.  Advertising and discovering services and recipes:
    1.  As a service operator, I want to be able to publish service offerings
        and recipes, so that users can search for my services and recipes they
        can use
    2.  As a service operator, I want to be able to label my service offerings
        and recipes, so that users can search for my services and recipes
        according to labels, without knowing about my service to begin with
    3.  As a user, I want to be able to search for services that are shared that
        I can consume in order to locate the right service to use in my
        application
2.  Recipes for running software systems in Kubernetes:
    1.  As someone who created a recipe for running a software system in
        Kubernetes, I want to share this recipe with others so that they can
        easily stand up their own copy
    2.  As someone who wants to run a particular software system in Kubernetes,
        I want to be able to search for and use recipes that others may have
        already created, so I can avoid spending time getting it to run myself
3.  Sharing resources for a service:
    1.  As an operator of a software system, I want to share the resources that
        are required to use the system so that my users can easily consume
        them in their own namespaces
    2.  As a user of a software system running in Kubernetes, I want to consume
        the shared resources associated with that system in my own namespace so
        that I can use the system in my application
4.  Sharing unique resources for a service:
    1.  As an operator of a software system, I want to be able to generate a
        unique resource for each user that wants to use the system so that I can
        manage permissions granularly
    2.  As a user of a software system, I want to get a unique set of resources
        for system in my namespace so that I can use the system in my
        application
5.  Policy for viewing and using services
    1.  As a service operator, I want to be able to describe a policy for who is
        able to view my service, so that I can ensure that only the right users
        have access to see that my service exists
    2.  As a service operator, I want to be able to describe a policy for who is
        able to use my service, so that I can ensure that users have the right
        degree of autonomy when using my service
6.  Consuming services - visibility
    1.  As a service operator, I want to be able to see the users consuming my
        service and track their usage of my service, so that I can be aware of
        the consumers of the service and charge them according to their usage
    2.  As an developer consuming services, I want to be able to see the
        services are being consumed, so that I can ensure that I am consuming
        only services that I need
7.  Resource provisioning
    1.  As a service operator, I want to be able to provision new resources for
        a user when they begin to consume my service so that the user will have
        their own resources to use for my service

### Advertising services and recipes

Within and without a Kubernetes cluster, there are services that users wish to
highlight and make available to other users.  Users might also wish to publish
recipes that allow other users to run their own services.  Some examples:

1.  A user's namespace contains `etcd`, `etcd- discovery`, and `postgresql`
    services, and the only one the user wants to share with others is the
    `postgresql` service
2.  A SaaS product like a externally hosted database for which a Kubernetes
    Service exists to provide a stable network endpoint
3.  A user makes a database run in Kubernetes and wants to share their recipe

In order to share these services, there has to be a central place where they can
be registered and advertised.  In this proposal, we'll call this place the
'catalog'.

### Labeling catalog entries

A user making an advertisement in the catalog should be able to label their
offerings.  Labeling an entry in the catalog lets users search by category or
attribute.

### Searching for services and recipes

Users should be able to browse the catalog and search by name and label.  Being
able to search by a set of labels makes it easier for users to discover catalog
entries of interest.

### Sharing a single set of resources for a service

The simplest way to share resources for an existing service is to share the same
resources for each consumer.  As an example: a development team is working on an
application that uses a database. The IT department manages the database (i.e.,
it lives off-cluster). All developers share the same credentials to access the
database, but these credentials are managed by IT. Rather than having each
developer create his or her own `Service` and `Secret` to connect to the
database, IT creates a "db-app-xyz" `Service` and a "db-app-xyz" `Secret` in the
"info-tech" namespace. 

### Consuming a set of shared resources for a service

Continuing our shared database example from a developer perspective: to use the
shared database service, a developer searches for it in the service catalog and
adds it to their namespace.  When the developer adds the service from the
catalog into their own namespace, they receive a copy of each of the resources
(Secrets, ConfigMaps, etc) that the service publisher has associated with that
service in their namespace.

### Sharing unique resources per consumer of a service

In a variation of the previous shared database example, IT has decided that
they do not want all developers to use the same credentials to access the
database.  Instead IT wants to issue unique credentials to each application
accessing the database.

### Consuming a unique set of resources for a service

To use the shared database, the developer searches for the it in the catalog
and creates a claim for that service.  The database service gets a notification
that a developer wants access to the database and provisions resources
(credentials, tablespace, etc) for the developer.  The resources (Secrets,
ConfigMaps, etc) required for accessing the database are created as part of the
claim binding in the developers namespace.

### Sharing recipes

Users also want the ability to share recipes for running services in addition to
sharing access to services that are already running.  As a completely fictitious
example, say the a user creates some kind of recipe that makes it easy to create
everything needed to spin up a new PostgreSQL database (customizable
username/password, `Service`, `Deployment`, etc.). The user wants to share this
recipe in a service catalog so others can find it and use it.

### Consuming recipes

When a user consumes a recipe, the pieces of the recipe are fully realized in
that user's namespace.  For example, if the recipe is to run an instance of
PostgreSQL, the user's namespace would probably have several new resources
created in it:

1.  A `Deployment` for the actual PostgreSQL containers
2.  A `Service` to provide a stable network endpoint
3.  A `Secret` with credentials to use the database

### Policy: who can view a service

When a user publishes an entry to the catalog, it is natural to want to be able
to express a policy for who should be able to see that service.  As an example,
take the case of an IT department that maintains several services inside and
outside of Kubernetes for use by many other departments in their organization.
The IT department wants to ensure that users in other departments can only see
the services that are meant for them to consume.  It is likely that there will
be some globally visible services and others that are just for certain
departments to use.  This implies that the publisher of a catalog entry should
be able to indicate whether a service is globally visible, or whether visibility
should be restricted to certain groups of users or service accounts.

### Policy: who can consume a service, and when

Once a user has located a catalog entry they want to consume, another problem
comes into play: what degree of access to that entry should they have?  Should
the user have carte-blanche access to that entry, should they have a quota
limiting their use of the entry somehow, or does a human being need to approve
their use of the entry?  There should be a policy to describe this.  Let's
revisit external database example.  Say that there are 3 other departments with
users that want to use the catalog entry for this database: marketing, shipping,
and sales.  The IT department wants the marketing department to be able to make
as many unique usernames and passwords as it wants, the shipping department to
be able to have up to 25 users, and the sales department to have 5 users.  The
IT department should be able to write a policy for the database service that
expresses these requirements.  So, when a user from the shipping department
tries to consume the service for the 26th time, they receive an error and have
to contact IT to have the policy changed so that they can get the additional
user they need.

### Metrics for operators

Operators naturally want to be able to see usage information for their catalog
entries.  Understanding how many users are consuming the entry helps service
operators and recipe creators understand the business impact of their catalog
entries.  Usage data is also necessary to construct market places where users
are charged for their use of a particular catalog entry.

### Metrics for service consumers

Consumers of entries from the catalog want to be able to see exactly which
entries their projects consume.  For example, if there is a cost associated with
using a catalog entry, a user will want to be able to ensure that their project
consumes only the catalog entries that are necessary for their project to
function.

### Provisioning resources

A number of use-cases that have already been discussed involve provisioning new
resources in Kubernetes.  It should also be possible to provision new resources
outside the cluster when an entry in the catalog is consumed.  For example, in
the case of an external database, the service operator may want to create a new
tablespace in the database for each user when the user adds the entry for the
database to their project.  It should be possible, therefore, for the author of
a service or recipe to write a custom workflow that is invoked when a user
consumes their catalog entry.

## Prior Art

### Helm

[Helm](https://github.com/kubernetes/helm) uses “charts” (e.g. resource
templates) that can instantiated as “releases”. Charts are stored in a HTTP
repository as compressed tarballs.  The tarball contains, among other files:

- `chart.yaml` - metadata about the chart
- `values.yaml` -  default variable values to use when processing the resource
   template files
- `templates/` - Kubernetes resource templates processed at `helm install` time

Workflow:

- `helm init` -  starts Tiller in the cluster and initializes CLI environment.
   Tiller runs in the cluster and manages the running Releases.
- `helm install <chartname>` - downloads the chart, processes the resource
   templates, and instantiates the resources in a release

Helm addresses use case 2.1: it provides a way to both create templates and
search for/consume them.  Charts can also incorporate secrets, however, secrets
created as part of a Helm release are not updated when the secret in the chart
is modified.

It is also possible to address use case 4.2; a user could take advantage of
Helm’s support for pre hooks to e.g. run a job that requests that a new
username/password be provisioned. The job, however, would need to have
sufficient privileges to create a username/password, which means the
administrative credentials necessary to provision the new account would need to
be included in the chart itself, which is a security risk and presumably not a
viable solution. Alternatively, the job would need to make an unprivileged
request to some other service that does have sufficient privileges.

TODO
Also: 7.1, use pre-hooks to provision resources

### Docker Compose

Docker Compose is most like a pod spec file.  It can contain information about
container(s) to be deployed together (similar to a pod) and the volumes,
environment variables, and ports that the combined composition will use.

Compose is similar to Helm in that it can bring up and take down a templated set
of resources as a unit.  Obviously, Compose does not create Kubernetes resources
as Helm does, including Services which make the service offered by the
composition discoverable.

### OpenShift Templates

Openshift templates offer a feature set similar to Helm.  Templates are
resources created as yaml files and imported as a resource of type "Template"
into Openshift.  Templates are processed with the "oc process" command, filling
in fields from parameter list and can even dynamically generate parameters like
passwords, for example. Once processed, the template becomes a single yaml file
output that can be fed to `oc create -f` to create the resources in Openshift.

The use case application is the same as that of Helm.  Once resources created by
the template, they are not tracked any further as a member of the template.  It
is up to the users to apply labels such that all resources created by a template
can be selected in the cluster.

### Cloud Foundry Service Broker

The CF Service Broker is more full featured than the previous examples, in that
it implements a catalog of services (CCDB) provided by any number of service
brokers and claimable by cluster users.  The cloud controller can discover
services from service brokers that implement it. the Service Broker API
using the "catalog" call.  Other calls that happen in response to a user
claiming the service from the catalog are "provision", "bind", "unbind", and
"deprovision".  The separation of the provisioning stage from the binding stage
allows for asynchronous dynamic provisioning of a service for a particular user.

This satisfies all the use cases except for the syncing of changed credentials.
There is no mechanism for the Service Broker to initiate a deprovision or
resync.

## Prototype

We have implemented a prototype to demonstrate a possible workflow for Catalogs
in a Kubernetes cluster.  The implementation is
[here](https://github.com/sjenning/kubernetes/tree/catalog-apiserver).

This prototype allows for the following type of workflow:

1. A user (the “provider” or “publisher”) creates one or more resources that
   he/she wants to share    with others. For this example, let’s imagine that I
   want to share a secret I created with other    users.
1. The publisher creates a catalog posting that references the secret.
1. Another user (the “consumer”) searches for and locates the entry in the
   catalog for the secret.
1. The consumer instantiates the catalog entry (we’re currently calling this a
   claim, but the name is open for discussion)
1. The end result is a new secret in the consumer’s namespace.
1. If the publisher changes the contents of the secret, all consumers’ secrets
   are automatically updated with the latest data.

We added a new executable that runs an apiserver and controllers related to
catalogs. This process handles requests to a new API group,
servicecatalog/v1alpha1. It also starts the controllers that monitor the
resources in the new API group and performs actions such as catalog entry
generation, catalog entry claim binding, and synchronization of resources
created by catalog entry claims.

The types in the new API group are:

- CatalogPosting
  - Namespace-scoped references to a collection of resources to be made
    available for claiming
- CatalogEntry
    - Cluster-scoped reference to a CatalogPosting with information relevant to
      consumers; displays sufficient level of details to consumers without
      exposing sensitive information
- CatalogClaim
  - A resource that expresses intent to consume (i.e. import into a destination
    namespace) the resources offered by a CatalogPosting via a CatalogEntry

Here is a more detailed description of the workflow above:

1. The publisher creates one or more resources in his/her namespace that are to
   be shared with others
1. The publisher creates a CatalogPosting that references the resources from
   step 1
1. A controller creates a CatalogEntry that corresponds to the new
   CatalogPosting that contains information about the shared resources relevant
   to consumers
1. The consumer lists the available entries in the catalog
1. The consumer creates a CatalogClaim for a specific CatalogEntry expressing
   intent to consume that entry
1. A controller processes the CatalogClaim and provisions the appropriate
   resources in the consumer’s namespace
1. A controller monitors resources associated with CatalogPostings for change
   and keeps all resources provisioned from CatalogClaims in sync

The controller to sync all possible resources types that a catalog posting could
reference would be a large task. For demonstration purposes, the claim sync
controller in this prototype only syncs secrets, since those are a resource a
publisher could likely change and would need to sync down to the claimed secrets
in order for pods to continue functioning.

For example, if a publisher shares a secret and a consumer claims it, when the
publisher changes the secret, the copy in the consumer’s namespace is updated.
The consumer may need to restart/redeploy any pods referencing that secret, if
the secret’s contents are referenced as environment variables, or if the
application isn’t able to react to changes to secrets mounted as files.

A video demo of the prototype is
[here](https://www.youtube.com/watch?v=Jbi19qk79bo).

<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-catalog.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

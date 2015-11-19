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

# Service Catalog

## Abstract

Kubernetes has Services but they are scoped per namespace, and (assuming eventual support for RBAC) you may not want everyone to be able to see and use all the services in your namespace.


This proposal is about:

- associating additional configuration/secret data with Services
- adding a new Service Catalog resource into which users can publish services that others can easily find and consume
- adding support for dynamic provisioning of resources from the catalog
- making it easier to link Services (and their configuration data) to Pods

## Service configuration data

The only data currently associated with a service is a list of zero or more endpoints. Many services also have configuration data associated with them. These include things such as logins, passwords, and additional connection parameters. To tie configuration data to a service, we propose adding information to the Service type to be able to express a relationship between the Service and its relevant configuration data. This could be accomplished by adding new field(s) or by using annotations. Potential configuration data target reference types include Secrets and ConfigData.

Just creating the reference association by itself does not accomplish much; to be useful, that data needs to be available to the processes executing in containers. More on this process is below in the sections on service claims and linking.

## Service catalogs

A service catalog is a listing of "services". Examples might include an external database that is deployed outside of the cluster, an actual Kubernetes Service, a template of cluster primitives to be created, and custom provisioners that can perform tasks such as creating users in external systems.

The catalog is not meant to include every service in the cluster. Instead, it should contain those services that users with to highlight and make available to other users. For example, your namespace might contain "etcd", "etcd-discovery", and "postgresql" services, and the only one you want to share with others is the postgresql service.

### Publishing to a catalog

Users should be able to publish entries to a service catalog. One means of accomplishing could be by adding annotations to the resources that can go into a catalog (e.g, Services). Another option would be to create `ServiceCatalogEntry` resources that reference the appropriate resources.

An entry in the service catalog has a "type", which indicates the behavior that occurs when it is claimed by a user. We have thought of the following potential types:

- reference: "I want to use this entry as-is, including its configuration data"
- template: "I want to create items from the specified template"
- provision: "I want the creation to be goverened by some other entity that implements the 'service broker' HTTP interface"

We see the type as an arbitrary `string`; one or more controllers could run to process each claim type, performing whatever logic is appropriate to fulfill the specific type in question.

### Viewing a catalog

Users should be able to list the entries in a service catalog. Users should be able to select an entry and "consume" it. We call this consumption "claiming" a service from a service catalog.

### Claiming a service

When you want to use an entry from the service catalog, you create a `ServiceClaim` that references the desired entry. A controller processes new claims for admission. This determines if the user who created the claim is allowed to consume the entry from the catalog. This decision can be flexible: it could be automated based on policy, or it could support manual intervention and workflow.

Once the claim has been admitted, a controller performs the provisioning process based on the service catalog entry's type.

For a reference claim, a new service is created in the user's namespace. The new service somehow needs to be a CNAME to the source. Additionally, for each configuration resource referenced by the source service, the controller creates a clone of it in the user's namespace.

For a template claim, the template is processed and its items are created in the user's namespace (see [OpenShift templates](https://docs.openshift.org/latest/dev_guide/templates.html) for more details on templates).

For a provision claim, the controller sends requests to the specified "service broker" endpoint to perform whatever provisioning actions the broker is coded to do.

Additional fulfillment types are possible as long as there is a controller that is handling them.

TODO poking holes in cluster firewall for cross-namespace connections

## Linking services

Claiming a service catalog entry only creates resources in the user's namespace. If all you need is a service and its DNS entry, this may be sufficient for your pods to function. But if you need the configuration data injected into your pod, it would be nice to make that easier to do.

We want to add the ability to link a service to a deployable resource such as a Deployment. This could look something like:

    kubectl link postgresql deployment/web
    service "postgresql" linked to deployemnt "web"
    configdata "postgresql-options" linked to deployment "web" as a volume
    secret "postgresql-credentials" linked to deployment "web" as a volume

This command would automatically inject the ConfigData and Secret objects as volumes into the Deployment. This could be flexible as well, allowing you instead to expose these items as environment variables.

## Cross-namespace networking

If the cluster has multi-tenant network isolation enabled, then a pod in namespace A won't be able to talk to a service in namespace B. We should look into ways to automatically manipulate the isolation rules to "poke holes" when claims are provisioned, to make the desired connectivity work, and to "unpoke" when the connectivity is no longer needed.

# TODOs

- One catalog or multiple?
- Figure out security
	- Who can publish a service?
	- Who can see/use a specific service?
- How do you keep the claimed services in sync with the sources?
- What does it mean to "unlink"?
- What happens if you delete a claim - does it cascade?



<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/service-catalog.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

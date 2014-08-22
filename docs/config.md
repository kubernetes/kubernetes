# Declarative Configuration

Declarative configuration of the desired state can be a powerful tool for managing continuously running services. In particular, low-impact, high-frequency, predictable (i.e., you can tell what it will do in advance), reversible (i.e., possible to rollback), idempotent, eventually consistent updates are the killer app for declarative configuration of multiple objects. 

## Decomposing the configuration process

Where possible, the Docker image should be used to configuration of Docker and the Linux environment known at build time, such as exposed ports, expected volumes, the entry point, capabilities required for correct operation of the application, and so on. This document will focus on configuration of objects managed by Kubernetes and its ecosystem.

We expect **configuration source** to be managed under a version control system, such as git. We’ll discuss the possible and recommended forms of the configuration source more below.

From the configuration source, we advocate the **generation** of the set of objects you wish to be instantiated. The resulting objects should be represented using a simple data format syntax, such as YAML or JSON. Breaking this out as a discrete step has a number of advantages, such as enabling validation and change review. This step should be 100% reproducible from a given version of the configuration source and any accompanying late-bound information, such as parameter values. 

Once the literal objects have been generated, it should be possible to perform a number of management operations on these objects in the system, such as to create them, update them, delete them, or delete them and then recreate them (which may be necessary if the objects are not 100% updatable). This will be achieved by communicating with the system’s RESTful APIs. In particular, objects will be created and/or updated via a **reconciliation process**. In order to do this in an extensible fashion, we will impose some compliance requirements upon APIs that can be targeted by this library/tool.

## Soap box: avoiding complex configuration generation

Complex configuration generation may appear attractive in order to reduce boilerplate or to reduce the number of explicit configuration variants that must be maintained, but it also imposes challenges for understandability, auditability, provenance tracking, authorization delegation, and resilience.

The need for configuration generation or transformation can be mitigated via a variety of means, such as appropriate default values, runtime automation, such as auto-scalers, or even code within user applications. To facilitate the latter, we should make labels available to lifecycle hooks and to containers (**mechanism TBD** -- see [issue 386](https://github.com/GoogleCloudPlatform/kubernetes/issues/386)).

Furthermore, with Kubernetes, we’re trying to encourage an open ecosystem of composable systems and layered APIs. Kubernetes will also support plug-ins for a variety of functionality. If you find your use case requires complex configuration transformations, that suggests a missing ecosystem component or plug-in. We encourage you to help the community build that missing mechanism as opposed to proliferating the use of fragile configuration generation logic.

For example, a common need is to specify instance-specific behavior. Rather than attempting to statically configure such behavior, one could build a runtime task-assignment service, which would be resilient to pod and host failures.

## Configuration source

One could write configuration source using a DSL, such as Aurora’s [Pystachio](http://aurora.incubator.apache.org/documentation/latest/configuration-tutorial/). However, after many years of experience doing just that, we recommend that you don’t. In addition to the usual challenges of DSLs (limited expressibility and testability, limited familiarity, inadequate documentation and tool support, etc.), configuration DSLs promote complex configuration generation, and also tend to have an infectious quality that compromises interoperability and composability. We also advise against pervasive preprocessing in a fashion akin to macro expansion, as that tends to subvert abstraction and encapsulation. 

Instead, we advocate that you write configuration source in a simple, concise data format that supports lightweight markup, such as YAML. Then, if necessary, write programs (e.g., in Go or Python) that accept this format and input and generate it as output, in order to accept objects of one type and write out objects of other types. This approach promotes encapsulation, and facilitates testing and reproducibility.

There would need to be a registry to select the generator based on object type. The generator could run locally (analogous to application invocation based on media type), or could be launched on the cluster on demand, or could be a continuously running service with an API (not unlike the built-in types).

Any shared generator programs (i.e., used by multiple users or projects) should be versioned similarly to APIs, since changing a generator’s output for a given input would compromise reproducibility.

## Object generation

One should be able to standard build infrastructure to perform object generation. It should be possible to use multiple configuration source formats and configuration generators, and compose together all of the resulting API objects.

We recommend capturing inputs to the generation process in version-controlled files.

Docker images referenced by mutable tags should be resolved to immutable image identifiers as part of this process (**mechanism TBD**). A team desiring reproducibility/reversibility should commit these identifiers to version control, also.

One common input to the object generation process will be the set of deployment labels to apply, such as to differentiate between development and production deployments, daily and weekly release tracks, users or teams, and so on. The labels both need to be applied to API objects and injected into label selectors. They likely need to prefix object names, also, in order to uniquify them. We should provide a simple generator that performs just these substitutions. We could provide it as a service in the apiserver. This service could be used, in general, to canonicalize objects for diff’ing during reconciliation.

## Management and reconciliation

Once we have generated a set of API objects, it should be possible to perform a number of management operations on them, such as creation, update, or even deletion. Creation and update are performed via a reconciliation process. A library/tool must compare the new desired state specified by the generated objects with the desired state already instantiated in the system, and then initiate operations to coerce the system’s state to match that of the generated objects. 

As discussed in [the initial configuration PR](https://github.com/GoogleCloudPlatform/kubernetes/pull/987), it should be possible to apply the management operation to a subset of the objects, by object type, by name(s), and/or by label selector.

The entire process should also return whether there are any permanent or retryable errors, in order to facilitate wrapping the process in a higher-level workflow.

## API requirements

In order to perform the above reconciliation protocol, the targeted APIs must adhere to the following requirements:

* Declarative creation via POST of the desired state
  * Idempotent creation based on object name; return AlreadyExists if the object exists; retain terminated objects for a reasonable duration (e.g., 5 minutes)
* Declarative update via PUT of the new desired state
  * Idempotence and consistency of PUTs ensured using per-object version numbers as preconditions
  * Mutations should return these version numbers to facilitate read-after-write consistency in the presence of intermediaries, such as proxies and caches
* GET should return the current desired state in a form that can be simply diff’ed with the new desired state (e.g., without observed or operational state); this could be done via a standard API endpoint or parameter
  * Default values should not be filled in and returned in the desired state; again, this could be an option (e.g., populate_default_values=false). This implies the system must remember that the user did not specify a value. We could potentially drop this requirement by requiring object canonicalization, but this approach also turns out to be useful for automation, such as auto-scalers (discussed more below).
  * Changes to default values are considered incompatible and thus require an API version change (and dynamically computed “smart defaults” are expected to provide reasonably consistent behavior within a particular API version)
* We need to standardize API error responses, both [standard HTTP errors](http://golang.org/pkg/net/http/#pkg-constants) and [API-level errors](https://github.com/GoogleCloudPlatform/kubernetes/blob/master/pkg/api/types.go], particularly which ones are permanent and which ones are retryable.  
* All objects should support labels, and it should be possible to filter list results using label selectors.

## Automated re-configuration processes

As mentioned above, automation systems, such as auto-scalers (which would set the `replicas` field in replicationControllers), require special accommodation in order to avoid conflicts with user-provided declarative configuration. Either the user’s configuration could contain annotations to inform the reconciliation library to copy the values from the system (thus preserving the values set by automation components), or the API could provide a way for automation components to set *shadow values*, which would appear as customized defaults to the user. The latter approach is preferable because it allows new API fields to be added and to be set automatically without requiring changes to user configurations.

## Deployment workflow

In general, we advocate building applications such that they are tolerant to being deployed in arbitrary orders. This leads to applications more tolerant to failures, also. However, some deployment order dependencies are inevitable. For example, we already have the issue that services must be created before their client pods. One could imagine that creation order by type could be configurable within the reconciliation library/tool. For more general dependencies, we suggest wrapping invocations of the library with a workflow tool, since often non-API operations are also required in such cases (e.g., calls to databases or storage systems).

## Updates

As mentioned at the beginning, updates are the killer app for declarative configuration.

Using the reconciliation protocol, it is reasonably straightforward to update only the objects that have changed. However, some care is required in order to not cause a client-visible outage of a running application.

For now, we will primarily discuss how to update pods, particularly those managed by replicationControllers.

First, as discussed in [issue 620](https://github.com/GoogleCloudPlatform/kubernetes/issues/620), we need a way to monitor application readiness. We then need a way to specify the desired aggregate level of service using a label selector. From this, we should be able to determine a safe rate at which to update the corresponding pods. If more complex control over the process is desired, we should enable control of the process by an external workflow tool.

For in-place updates of the pods, the ability to update a set of objects identified by a label selector from the same generated object should be sufficient -- we don’t necessarily need to make the reconciliation library/tool aware of replicationController semantics.

Other approaches are possible, too. For example, one could create a new replicationController with the new pod template and instruct an auto-scaler to scale it up while scaling the old one down. One could even keep both sets of pods running and redirect traffic by updating the corresponding service. The primitives are intentionally flexible in order to permit multiple approaches to deployment.


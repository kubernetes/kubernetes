# Kubernetes Proposal - Projects

**Related PR:** 

| Topic | Link |
| ---- | ---- |
| Access.md | https://github.com/GoogleCloudPlatform/kubernetes/pull/891 |
| Indexing | Need an issue to discuss optimiziation of label selectors |
| Id vs Name | Need an issue to discuss human-readable name versus id semantics |
| Cluster Subdivision | https://github.com/GoogleCloudPlatform/kubernetes/issues/442 |

## Background

High level goals:

* Enable an easy-to-use mechanism to scope Kubernetes resources
* Ensure it aligns with access control proposal
* Ensure it provides a pattern to easily add new project scoped resources as extensions to core
* Ensure it is efficient at desired project scale limits

## Use cases

Actors:

1. k8s admin - administers a kubernetes cluster
2. k8s user - uses a kubernetes cluster to schedule pods

Initial:

1. Ability for a k8s admin to create projects
2. Ability for a k8s admin to delete projects
3. Ability for a k8s user to list projects
4. Ability for a k8s user to get a project
5. Ability for a k8s user to list k8s resource scoped to a project
6. Project has a DNS-compatible id to support compound naming conventions
7. A project-scoped resource is not transferrable to another project after creation.

Future Improvements:

1. Ability for a k8s user to create projects
2. Abiltiy for a k8s user to delete projects
3. Ability to tie project-scoped resources to project-lifecycle (i.e. handle orphaned resources)

## Proposed Design

### Model Changes

Introduce a top-level object called **Project** that provides a scope to Kubernetes cluster resources.

```
// Project enables scoping of select Kubernetes resources
type Project struct {
	JSONBase `json:",inline" yaml:",inline"`

	// This project's set of labels
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`
}
```

Any resource that is scoped by a **Project** must suppport **labels**.

To scope a resource to a project, a new **ProjectResourceStorage** object will be defined that will scope storage
resources to a particular project constraint.

```
// ProjectResourceStorage implements the RESTStorage interface and enforces project scoping of resource types
type ProjectResourceStorage struct {
	projectStorage *project.RegistryStorage
	delegate *RESTStorage
}
```

The **ProjectResourceStorage** will wrap an existing RESTStorage object to enforce project scoping constraints as follows:

1. Create(interface{})
	* Require that a label **project=projectID** is on object, else error
	* Require that a project exists with **projectID** in **projectStorage**, else error
	* Invoke delegate.Create()
2. List(labels.Selector)
	* Require that a label **project=projectID** is in labels set, else error
	* Invoke delegate.List()
3. Get(id string)
	* Invoke delegate.Get()
4. Update(interface{})
	* Enforce label **project=projectID** is equal to existing object **project=projectID** label, else error
	* Invoke delegate.Update()
5. Delete(id string)
	* Invoke delegate.Delete()
6. New
	* Invoke delegate.New()

The following resources are project-scoped.
* pod
* controller
* service

The following resources may be project-scoped.
* policy

### Kubernetes API Server

The API server may be provided with an implicit project scope on each request to project scoped resources.

If a project scope is provided, it is validated to match the resource to an associated project.

1. Query parameters, e.g. ?project=<project>
2. HTTP header, e.g. X_KUBERNETES_PROJECT=<project>
3. Subdomain, e.g. http://myproject.kubernetes.example.com
4. HTTP Path, e.g. http://kubernetes.example.com/myproject/

### Kubernetes Storage

#### Define a project resource, ensure a project label or attribute is applied on all project-scoped resources

| Key | Description |
| ---- | ---- |
| /registry/projects/{project} | Holds information about the {project} resource |

Project scoped resources storage behavior is not changed, and is unknown by the project resource.
The project scoping is done via application of a project-label or attribute on underlying resources.
Efficient query on that attribute is required for scoping queries.

Rationale:

1. Efficient label selector query is needed for other resource types.
2. Do not require a project-scoped resource to change its storage location.
	* For example, it could be possible to persist project's outside of etcd, but keep pod defininitions in etcd.
	* Keeps resource storage options best suited to the particular resource type.	
3. Implementation strategy of wrapping storage delegates keeps scoped resource persistence agnostic of project storage.

For an analysis on alternative options discussed, see **Appendix: Alternative storage proposal**

### kubecfg client

kubecfg supports following:

1. list /projects
2. get /project/{projectID}

To filter resources by project, the client may do the following:

kubecfg -l project={projectID} list /{resource}

To avoid always specifying the label selector on list operations, the following is proposed:

* kubecfg set-project <projectId>
	* Store a local file to configure client default project to use as default label selector on list queries.
	* If no <projectId> is provided, just print current default project is none provided.
* kubecfg unset-project
	* Remove the client default project.

The kubecfg client would store default project information in the same manner it caches authentication information today
as a file on file system.

## Open Issues

### Deleting a project, when are related resources removed?

There are multiple options to pursue:

1. Notification model to tell project scoped resources that the project is being removed.
2. Scheduled clean-up of orphaned resources that are associated with a removed project.

## Appendix

### Alternative storage proposals discussed

#### Option 1: Scope API Server resources by project, Kubelet/Scheduler resources by host

| Key | Description |
| ---- | ---- |
| /registry/projects/{project} | Holds information about the {project} resource |
| /registry/projects/{project}/policies/{policy} | Holds information about the {policy} resource in {project} |
| /registry/projects/{project}/services/{service} | Holds information about the {service} resource in {project} |
| /registry/projects/{project}/controllers/{controller} | Holds information about the {controller} resource in {project} |
| /registry/projects/{project}/pods/{pod} | Holds information about the {pod} resource in {project} |
| /registry/hosts/{host} | Holds information about the pods scheduled on {host} |

Pros:

1. No need for a secondary index to filter resource type by project
2. Logical scoping of project managed resources reflected in etcd hierarchy
3. Simpler cleanup of project-scoped resources [recursive delete]
4. Plugin non-core resources (builds, deployments, etc.) have clear storage path and model to scope to project.

Cons:

1. Need secondary index to list any resource independent of project
2. Difficult to lookup project-scoped resource by immutable identifier (requires an index by id).
3. It's likely that non-core resources may have dependent clean-up on their lifecycle so co-location in etcd may not have major value.

#### Option 2: Apply a project label or attribute on all project-scoped resources

| Key | Description |
| ---- | ---- |
| /registry/projects/{project} | Holds information about the {project} resource |
| /registry/policies/{policy} | Holds information about the {policy} resource |
| /registry/services/{service} | Holds information about the {service} resource |
| /registry/controllers/{controller} | Holds information about the {controller} resource |
| /registry/pods/{pod} | Holds information about the {pod} resource |
| /registry/hosts/{host} | Holds information about the pods scheduled on {host} |

Pros:

1. Limited disruption over current model
2. Easy to query resources independent of project scope
3. Easy to address an object by immutable identifier.

Cons:

1. Requires an index to avoid an expensive post-filter when listing resources by project (e.g. list pods by project)
2. It is not as clear how resources are scoped in storage.
3. Security on write paths is less aligned with security model of Kubernetes if we wanted to control more fine-grain write paths.
4. Potentially harder to clean-up resources with a project.

#### Option 3: Project scope each resource type

| Key | Description |
| ---- | ---- |
| /registry/projects/{project} | Holds information about the {project} resource |
| /registry/services/{project}/{service} | Holds information about the {service} resource in {project} |
| /registry/controllers/{project}/{controller} | Holds information about the {controller} resource in {project} |
| /registry/pods/{project}/{pod} | Holds information about the {pod} resource in {project} |
| /registry/hosts/{host} | Holds information about the pods scheduled on {host} |

Pros:

1. No need for a secondary index to filter resource type by project
2. Do not need secondary index to list any resource independent of project
3. Simpler cleanup of project-scoped resources [recursive delete]

Cons:

1. No ability to recursively roll-up all resources for a given project in single call
2. Clean-up of project scoped resources is more complicated.
3. Unclear how to look-up a resource by immutable id vs. name (still requires an indexer)

#### Preferred Option:

Depending on how we want to optimize Kubernetes, either Option 1 or Option 2 may work.

If option 1 is preferred, 

1.  Data is scoped logically to ACL model.
2.  Does not require a secondary index to implement ACL model efficiently.
3.  Simpler to enumerate resources scoped to a project.
4.  Name uniqueness constraints are enforced by etcd key.
5.  No easy mechanism to look-up a resource by it's immutable identifier.
6.  May have an impact on how etcd watches are handled by scheduler

If we wanted to list resources independent of project, we would require an index.

If option 2 is preferred,

1. Data is scoped logically to resource structure.
2. Does require an index to implement ACL model and project scoping model efficiently.
3. Easy to look up a resource by immutable id.

At this point, option 2 is preferred.

Rationale:

1. Optimizing label selectors needs to happen across all resource types, so efficient scoped query should get supported.
2. Strong desire to keep etcd a key/value store and to not encode path semantics in keys
3. Implementation strategy of wrapping storage delegates

#### Additional comments:

In general, it seems like we have a pattern that now requires alternate listing strategies.

It appears we need a system like the following that clearly distinguishes the following behaviors on a given resource type,

1. storage opts (create, update, get, delete resource by id)
2. listing opts (list resource with optional attribute filter and sort)
3. watching opts (listen for changes on a resource type across data sources)

We may want to internally harden how we perform these operations so listing opts can always be fulfilled by some external
indexer rather than calling direct to etcd.  Introducing a project scope concept requires indexes to effectively meet resource retrieval, and the debate between option 1 and 2 is which we favor in our storage, but in the end, if we need an index to support all the required potential use cases, it may be best to specify the pattern we follow for indexing data now before we get too far down a particular path.

# Overview

Object metadata (labels, annotations) can be used to select objects
(e.g. selectors) and modify their behavior (e.g. namespace annotations
for network isolation control).

It is convenient to also use object metadata to apply certain policies
to these objects. Some examples:

- image policy can restrict pods labeled as 'env: prod' to only use
  certain trusted registries;

- service policy can restrict 'LoadBalancer' service type to services
  labeled with 'allowExternal: true'.

In addition, authorization controls can be used to enable resource
operations for certain subsets of users. These controls tend to be
coarse-grained: if a user can create a pod in a namespace, they can
label or annotate it as they see fit. It makes types of policy
mentioned above less useful, as user can arbitrarily change metadata
and thus make their objects compliant with any policy they want.

In order to constrain object attributes to certain roles, we propose
to add the notion of protected attributes to RBAC API group.

By default no attributes are protected: this matches the existing
behavior.

Protected attribute is a (kind, name, role, valuesList) tuple. Kind
and name uniquely identify the attribute: initial implementation
supports labels and annotations, but can be extended in the
future. Role is a reference to RBAC ClusterRole or Role. If valuesList
is empty, role can use any value for the attribute. Otherwise, role
can only set the attribute to the values in the valuesList.

Protected attributes can be defined on cluster or namespace level:
cluster protected attributes apply to every namespace and can only
reference cluster roles. Namespace-level protected attributes only
apply in the namespace they are defined in, and can reference both
cluster roles or roles in that namespace.

# Implementation

New types are added to RBAC group to support protected attributes:

``` go
// ProtectedAttribute allows a fine-grained control of who can set or
// remove certain attributes (e.g. labels/annotations) on resources. In
// order to set or remove the protected attribute, requester must be a
// member of a role that has access to that attribute. Applies only to
// resources in the same namespace as ProtectedAttribute itself.
type ProtectedAttribute struct {
	unversioned.TypeMeta
	api.ObjectMeta

	// AttributeKind is a kind of an attribute this restriction applies to.
	// Can be "Label" or "Annotation".
	AttributeKind string

	// AttributeName is the name of an attribute this restriction
	// applies to.
	AttributeName string

	// RoleRef references a Role or a ClusterRole that can set or
	// remove the attribute.
	RoleRef api.ObjectReference

	// ProtectedValues is an optional list of values protected by the
	// role. By default every value is protected, ProtectedValues
	// allows narrowing it down to a fixed list.
	ProtectedValues []string
}

// ClusterProtectedAttribute allows a fine-grained control of who can
// set or remove certain attributes (e.g. labels/annotations) on
// resources. In order to set or remove the protected attribute,
// requester must be a member of a role that has access to that
// attribute. Applies to all namespaces in the cluster.
type ClusterProtectedAttribute struct {
	unversioned.TypeMeta
	api.ObjectMeta

	// AttributeKind is a kind of an attribute this restriction
	// applies to. Can be "Label" or "Annotation".
	AttributeKind string

	// AttributeName is the name of an attribute this restriction
	// applies to.
	AttributeName string

	// RoleRef references a Role or a ClusterRole that can set or
	// remove the attribute.
	RoleRef api.ObjectReference

	// ProtectedValues is an optional list of values protected by the
	// role. By default every value is protected, ProtectedValues
	// allows narrowing it down to a fixed list.
	ProtectedValues []string
}
```

## Admission controller

New admission controller is added to enforce protected attributes. It
ensures that resource is only admitted if a requester has sufficient
role memberships by inspecting all labels and annotations and
cross-referencing protected attributes with requester role bindings:

https://github.com/olegshaldybin/kubernetes/blob/protected-attributes/plugin/pkg/admission/protectedattributes/admission.go

## Alternatives considered and their shortcomings

### Policies directly referencing roles

- RBAC is not only possible authorization mode, and making all
  policies depend on RBAC types limits their flexibility;

- roles are derived from user's identity, while policies are usually
  more naturally described by workload types (dev, prod, test).

### Hook-based attribute restrictions

- require writing and configuring plugins for different types of
  hooks: while generally useful, we think that first-class API objects
  for protected attributes provide a more user-friendly solution for
  labels and annotations. It's possible to extend them with other
  types, and even add callouts to external protected attribute
  handlers in the future.

### Non-RBAC based fine-grained authorization

- RBAC seems to be a natural prerequisite, since we need to be able to
  map object metadata to the capabilities of a requester. While it's
  possible to introduce role hierarchy that is specific to protected
  attributes, that would require duplicating a good chunk of RBAC
  functionality, and introduce unnecessary confusion.

---
---

## Protected Attributes

In Kubernetes, any label and/or annotation can be added to any
resource metadata.  But sometimes it makes sense to constrain
sensitive attributes so that they can only be modified by users in a
particular role. For example, certain pods/services can be a subject
to different policies based on labels they are tagged with.

Protected attributes allow users to put restrictions on
labels/annotations. This functionality works alongside RBAC
(Role-Based Access Control) authorization and considered experimental.

Just like roles and role bindings, protected attributes can be defined
on a cluster or a namespace level. Cluster protected attributes are
enforced in all namespaces, namespace-level protected attributes are
only enforced in the namespace they are defined in.

Protected attributes can reference either the role in the same
namespace as the attribute itself, or a cluster role. If no specific
attribute values are specified, role can set any value. If the list of
values is provided, role can only set the attribute value to one of
the predefined values.

Example 1: namespace admins can set label "env" in "default" namespace.

```yml
apiVersion: rbac.authorization.k8s.io/v1alpha1
kind: ProtectedAttribute
metadata:
  namespace: default
  name: envLabel

attributeKind: Label
attributeName: env

roleRef:
  kind: Role
  name: admin
```

Example 2: cluster network admins can control network isolation.

```yml
apiVersion: rbac.authorization.k8s.io/v1alpha1
kind: ClusterProtectedAttribute
metadata:
  name: netIsolation

attributeKind: Annotation
attributeName: net.alpha.kubernetes.io/network-isolation

protectedValues:
  - "on"
  - "off"

roleRef:
  kind: ClusterRole
  name: admin
```

Protected attribute lifecycle is itself a subject to RBAC
authorization decisions: user must have enough permissions to
manage protected attribute resources.

### Attribute Admission Rules

For every label and annotation on the resource being created, updated
and deleted (both current and new version for updates), Kubernetes
checks if that attribute is protected either on cluster or namespace
level. If attribute it's not protected, it's allowed. Otherwise, user
performing the action must be a member of at least one Role or
ClusterRole that can set the attribute to its current value.

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
[here](http://releases.k8s.io/release-1.3/docs/proposals/security-context-constraints.md).

Documentation for other releases can be found at
[releases.k8s.io](http://releases.k8s.io).
</strong>
--

<!-- END STRIP_FOR_RELEASE -->

<!-- END MUNGE: UNVERSIONED_WARNING -->

## Abstract

PodSecurityPolicy allows cluster administrators to control the creation and validation of a security
context for a pod and containers.

## Motivation

Administration of a multi-tenant cluster requires the ability to provide varying sets of permissions
among the tenants, the infrastructure components, and end users of the system who may themselves be
administrators within their own isolated namespace.

Actors in a cluster may include infrastructure that is managed by administrators, infrastructure
that is exposed to end users (builds, deployments), the isolated end user namespaces in the cluster, and
the individual users inside those namespaces.  Infrastructure components that operate on behalf of a
user (builds, deployments) should be allowed to run at an elevated level of permissions without
granting the user themselves an elevated set of permissions.

## Goals

1.  Associate [service accounts](../design/service_accounts.md), groups, and users with
a set of constraints that dictate how a security context is established for a pod and the pod's containers.
1.  Provide the ability for users and infrastructure components to run pods with elevated privileges
on behalf of another user or within a namespace where privileges are more restrictive.
1.  Secure the ability to reference elevated permissions or to change the constraints under which
a user runs.

## Use Cases

Use case 1:
As an administrator, I can create a namespace for a person that can't create privileged containers
AND enforce that the UID of the containers is set to a certain value

Use case 2:
As a cluster operator, an infrastructure component should be able to create a pod with elevated
privileges in a namespace where regular users cannot create pods with these privileges or execute
commands in that pod.

Use case 3:
As a cluster administrator, I can allow a given namespace (or service account) to create privileged
pods or to run root pods

Use case 4:
As a cluster administrator, I can allow a project administrator to control the security contexts of
pods and service accounts within a project


## Requirements

1.  Provide a set of restrictions that controls how a security context is created for pods and containers
as a new cluster-scoped object called `PodSecurityPolicy`.
1.  User information in `user.Info` must be available to admission controllers. (Completed in
https://github.com/GoogleCloudPlatform/kubernetes/pull/8203)
1.  Some authorizers may restrict a user’s ability to reference a service account.  Systems requiring
the ability to secure service accounts on a user level must be able to add a policy that enables
referencing specific service accounts themselves.
1.  Admission control must validate the creation of Pods against the allowed set of constraints.

## Design

### Model

PodSecurityPolicy objects exist in the root scope, outside of a namespace.  The
PodSecurityPolicy will reference users and groups that are allowed
to operate under the constraints.  In order to support this, `ServiceAccounts` must be mapped
to a user name or group list by the authentication/authorization layers.  This allows the security
context to treat users, groups, and service accounts uniformly.

Below is a list of PodSecurityPolicies which will likely serve most use cases:

1.  A default policy object.  This object is permissioned to something which covers all actors, such
as a `system:authenticated` group, and will likely be the most restrictive set of constraints.
1.  A default constraints object for service accounts.  This object can be identified as serving
a group identified by `system:service-accounts`, which can be imposed by the service account authenticator / token generator.
1.  Cluster admin constraints identified by `system:cluster-admins` group - a set of constraints with elevated privileges that can be used
by an administrative user or group.
1.  Infrastructure components constraints which can be identified either by a specific service
account or by a group containing all service accounts.

```go
// PodSecurityPolicy governs the ability to make requests that affect the SecurityContext
// that will be applied to a pod and container.
type PodSecurityPolicy struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty"`

	// Spec defines the policy enforced.
	Spec PodSecurityPolicySpec `json:"spec,omitempty"`
}

// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
	// Privileged determines if a pod can request to be run as privileged.
	Privileged bool `json:"privileged,omitempty"`
	// Capabilities is a list of capabilities that can be added.
	Capabilities []api.Capability `json:"capabilities,omitempty"`
	// Volumes allows and disallows the use of different types of volume plugins.
	Volumes VolumeSecurityPolicy `json:"volumes,omitempty"`
	// HostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	HostNetwork bool `json:"hostNetwork,omitempty"`
	// HostPorts determines which host port ranges are allowed to be exposed.
	HostPorts []HostPortRange `json:"hostPorts,omitempty"`
	// HostPID determines if the policy allows the use of HostPID in the pod spec.
	HostPID bool `json:"hostPID,omitempty"`
	// HostIPC determines if the policy allows the use of HostIPC in the pod spec.
	HostIPC bool `json:"hostIPC,omitempty"`
	// SELinuxContext is the strategy that will dictate the allowable labels that may be set.
	SELinuxContext SELinuxContextStrategyOptions `json:"seLinuxContext,omitempty"`
	// RunAsUser is the strategy that will dictate the allowable RunAsUser values that may be set.
	RunAsUser RunAsUserStrategyOptions `json:"runAsUser,omitempty"`

	// The users who have permissions to use this policy
	Users []string `json:"users,omitempty"`
	// The groups that have permission to use this policy
	Groups []string `json:"groups,omitempty"`
}

// HostPortRange defines a range of host ports that will be enabled by a policy
// for pods to use.  It requires both the start and end to be defined.
type HostPortRange struct {
	// Start is the beginning of the port range which will be allowed.
	Start int `json:"start"`
	// End is the end of the port range which will be allowed.
	End int `json:"end"`
}

// VolumeSecurityPolicy allows and disallows the use of different types of volume plugins.
type VolumeSecurityPolicy struct {
	// HostPath allows or disallows the use of the HostPath volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#hostpath
	HostPath bool `json:"hostPath,omitempty"`
	// EmptyDir allows or disallows the use of the EmptyDir volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#emptydir
	EmptyDir bool `json:"emptyDir,omitempty"`
	// GCEPersistentDisk allows or disallows the use of the GCEPersistentDisk volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#gcepersistentdisk
	GCEPersistentDisk bool `json:"gcePersistentDisk,omitempty"`
	// AWSElasticBlockStore allows or disallows the use of the AWSElasticBlockStore volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#awselasticblockstore
	AWSElasticBlockStore bool `json:"awsElasticBlockStore,omitempty"`
	// GitRepo allows or disallows the use of the GitRepo volume plugin.
	GitRepo bool `json:"gitRepo,omitempty"`
	// Secret allows or disallows the use of the Secret volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#secrets
	Secret bool `json:"secret,omitempty"`
	// NFS allows or disallows the use of the NFS volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/volumes.md#nfs
	NFS bool `json:"nfs,omitempty"`
	// ISCSI allows or disallows the use of the ISCSI volume plugin.
	// More info: http://releases.k8s.io/HEAD/examples/volumes/iscsi/README.md
	ISCSI bool `json:"iscsi,omitempty"`
	// Glusterfs allows or disallows the use of the Glusterfs volume plugin.
	// More info: http://releases.k8s.io/HEAD/examples/volumes/glusterfs/README.md
	Glusterfs bool `json:"glusterfs,omitempty"`
	// PersistentVolumeClaim allows or disallows the use of the PersistentVolumeClaim volume plugin.
	// More info: http://releases.k8s.io/HEAD/docs/user-guide/persistent-volumes.md#persistentvolumeclaims
	PersistentVolumeClaim bool `json:"persistentVolumeClaim,omitempty"`
	// RBD allows or disallows the use of the RBD volume plugin.
	// More info: http://releases.k8s.io/HEAD/examples/volumes/rbd/README.md
	RBD bool `json:"rbd,omitempty"`
	// Cinder allows or disallows the use of the Cinder volume plugin.
	// More info: http://releases.k8s.io/HEAD/examples/mysql-cinder-pd/README.md
	Cinder bool `json:"cinder,omitempty"`
	// CephFS allows or disallows the use of the CephFS volume plugin.
	CephFS bool `json:"cephfs,omitempty"`
	// DownwardAPI allows or disallows the use of the DownwardAPI volume plugin.
	DownwardAPI bool `json:"downwardAPI,omitempty"`
	// FC allows or disallows the use of the FC volume plugin.
	FC bool `json:"fc,omitempty"`
}

// SELinuxContextStrategyOptions defines the strategy type and any options used to create the strategy.
type SELinuxContextStrategyOptions struct {
	// Type is the strategy that will dictate the allowable labels that may be set.
	Type SELinuxContextStrategy `json:"type"`
	// seLinuxOptions required to run as; required for MustRunAs
	// More info: http://releases.k8s.io/HEAD/docs/design/security_context.md#security-context
	SELinuxOptions *api.SELinuxOptions `json:"seLinuxOptions,omitempty"`
}

// SELinuxContextStrategyType denotes strategy types for generating SELinux options for a
// SecurityContext.
type SELinuxContextStrategy string

const (
	// container must have SELinux labels of X applied.
	SELinuxStrategyMustRunAs SELinuxContextStrategy = "MustRunAs"
	// container may make requests for any SELinux context labels.
	SELinuxStrategyRunAsAny SELinuxContextStrategy = "RunAsAny"
)

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
type RunAsUserStrategyOptions struct {
	// Type is the strategy that will dictate the allowable RunAsUser values that may be set.
	Type RunAsUserStrategy `json:"type"`
	// UID is the user id that containers must run as.  Required for the MustRunAs strategy if not using
	// a strategy that supports pre-allocated uids.
	UID *int64 `json:"uid,omitempty"`
	// UIDRangeMin defines the min value for a strategy that allocates by a range based strategy.
	UIDRangeMin *int64 `json:"uidRangeMin,omitempty"`
	// UIDRangeMax defines the max value for a strategy that allocates by a range based strategy.
	UIDRangeMax *int64 `json:"uidRangeMax,omitempty"`
}

// RunAsUserStrategyType denotes strategy types for generating RunAsUser values for a
// SecurityContext.
type RunAsUserStrategy string

const (
	// container must run as a particular uid.
	RunAsUserStrategyMustRunAs RunAsUserStrategy = "MustRunAs"
	// container must run as a particular uid.
	RunAsUserStrategyMustRunAsRange RunAsUserStrategy = "MustRunAsRange"
	// container must run as a non-root uid
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategy = "MustRunAsNonRoot"
	// container may make requests for any uid.
	RunAsUserStrategyRunAsAny RunAsUserStrategy = "RunAsAny"
)
```

### PodSecurityPolicy Lifecycle

As reusable objects in the root scope, PodSecurityPolicy follows the lifecycle of the
cluster itself.  Maintenance of constraints such as adding, assigning, or changing them is the
responsibility of the cluster administrator.

Creating a new user within a namespace should not require the cluster administrator to
define the user's PodSecurityPolicy.  They should receive the default set of policies
that the administrator has defined for the groups they are assigned.


## Default PodSecurityPolicy And Overrides

In order to establish policy for service accounts and users, there must be a way
to identify the default set of constraints that is to be used.  This is best accomplished by using
groups.  As mentioned above, groups may be used by the authentication/authorization layer to ensure
that every user maps to at least one group (with a default example of `system:authenticated`) and it
is up to the cluster administrator to ensure that a `PodSecurityPolicy` object exists that
references the group.

If an administrator would like to provide a user with a changed set of security context permissions,
they may do the following:

1.  Create a new `PodSecurityPolicy` object and add a reference to the user or a group
that the user belongs to.
1.  Add the user (or group) to an existing `PodSecurityPolicy` object with the proper
elevated privileges.

## Admission

Admission control using an authorizer provides the ability to control the creation of resources
based on capabilities granted to a user.  In terms of the `PodSecurityPolicy`, it means
that an admission controller may inspect the user info made available in the context to retrieve
an appropriate set of policies for validation.

The appropriate set of PodSecurityPolicies is defined as all of the policies
available that have reference to the user or groups that the user belongs to.

Admission will use the PodSecurityPolicy to ensure that any requests for a
specific security context setting are valid and to generate settings using the following approach:

1.  Determine all the available `PodSecurityPolicy` objects that are allowed to be used
1.  Sort the `PodSecurityPolicy` objects in a most restrictive to least restrictive order.
1.  For each `PodSecurityPolicy`, generate a `SecurityContext` for each container.  The generation phase will not override
any user requested settings in the `SecurityContext`, and will rely on the validation phase to ensure that
the user requests are valid.
1.  Validate the generated `SecurityContext` to ensure it falls within the boundaries of the `PodSecurityPolicy`
1.  If all containers validate under a single `PodSecurityPolicy` then the pod will be admitted
1.  If all containers DO NOT validate under the `PodSecurityPolicy` then try the next `PodSecurityPolicy`
1.  If no `PodSecurityPolicy` validates for the pod then the pod will not be admitted


## Creation of a SecurityContext Based on PodSecurityPolicy

The creation of a `SecurityContext` based on a `PodSecurityPolicy` is based upon the configured
settings of the `PodSecurityPolicy`.

There are three scenarios under which a `PodSecurityPolicy` field may fall:

1.  Governed by a boolean: fields of this type will be defaulted to the most restrictive value.
For instance, `AllowPrivileged` will always be set to false if unspecified.

1.  Governed by an allowable set: fields of this type will be checked against the set to ensure
their value is allowed.  For example, `AllowCapabilities` will ensure that only capabilities
that are allowed to be requested are considered valid.  `HostNetworkSources` will ensure that
only pods created from source X are allowed to request access to the host network.
1.  Governed by a strategy: Items that have a strategy to generate a value will provide a
mechanism to generate the value as well as a mechanism to ensure that a specified value falls into
the set of allowable values.  See the Types section for the description of the interfaces that
strategies must implement.

Strategies have the ability to become dynamic.  In order to support a dynamic strategy it should be
possible to make a strategy that has the ability to either be pre-populated with dynamic data by
another component (such as an admission controller) or has the ability to retrieve the information
itself based on the data in the pod.  An example of this would be a pre-allocated UID for the namespace.
A dynamic `RunAsUser` strategy could inspect the namespace of the pod in order to find the required pre-allocated
UID and generate or validate requests based on that information.


```go
// SELinuxStrategy defines the interface for all SELinux constraint strategies.
type SELinuxStrategy interface {
	// Generate creates the SELinuxOptions based on constraint rules.
	Generate(pod *api.Pod, container *api.Container) (*api.SELinuxOptions, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(pod *api.Pod, container *api.Container) fielderrors.ValidationErrorList
}

// RunAsUserStrategy defines the interface for all uid constraint strategies.
type RunAsUserStrategy interface {
	// Generate creates the uid based on policy rules.
	Generate(pod *api.Pod, container *api.Container) (*int64, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(pod *api.Pod, container *api.Container) fielderrors.ValidationErrorList
}
```

## Escalating Privileges by an Administrator

An administrator may wish to create a resource in a namespace that runs with
escalated privileges.   By allowing security context
constraints to operate on both the requesting user and the pod's service account, administrators are able to
create pods in namespaces with elevated privileges based on the administrator's security context
constraints.

This also allows the system to guard commands being executed in the non-conforming container.  For
instance, an `exec` command can first check the security context of the pod against the security
context constraints of the user or the user's ability to reference a service account.
If it does not validate then it can block users from executing the command.  Since the validation
will be user aware, administrators would still be able to run the commands that are restricted to normal users.

## Interaction with the Kubelet

In certain cases, the Kubelet may need provide information about
the image in order to validate the security context.  An example of this is a cluster
that is configured to run with a UID strategy of `MustRunAsNonRoot`.

In this case the admission controller can set the existing `MustRunAsNonRoot` flag on the `SecurityContext`
based on the UID strategy of the `SecurityPolicy`.  It should still validate any requests on the pod
for a specific UID and fail early if possible.  However, if the `RunAsUser` is not set on the pod
it should still admit the pod and allow the Kubelet to ensure that the image does not run as
`root` with the existing non-root checks.




<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/proposals/security-context-constraints.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->

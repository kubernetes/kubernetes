/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*
This file (together with pkg/apis/extensions/v1beta1/types.go) contain the experimental
types in kubernetes. These API objects are experimental, meaning that the
APIs may be broken at any time by the kubernetes team.

DISCLAIMER: The implementation of the experimental API group itself is
a temporary one meant as a stopgap solution until kubernetes has proper
support for multiple API groups. The transition may require changes
beyond registration differences. In other words, experimental API group
support is experimental.
*/

package extensions

import (
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/pkg/api"
)

const (
	// SysctlsPodSecurityPolicyAnnotationKey represents the key of a whitelist of
	// allowed safe and unsafe sysctls in a pod spec. It's a comma-separated list of plain sysctl
	// names or sysctl patterns (which end in *). The string "*" matches all sysctls.
	SysctlsPodSecurityPolicyAnnotationKey string = "security.alpha.kubernetes.io/sysctls"
)

// describes the attributes of a scale subresource
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	// +optional
	Replicas int32
}

// represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int32

	// label query over pods that should match the replicas count.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector
}

// +genclient=true
// +noMethods=true

// represents a scaling request for a resource.
type Scale struct {
	metav1.TypeMeta
	// Standard object metadata; More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata.
	// +optional
	metav1.ObjectMeta

	// defines the behavior of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status.
	// +optional
	Spec ScaleSpec

	// current status of the scale. More info: http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#spec-and-status. Read-only.
	// +optional
	Status ScaleStatus
}

// Dummy definition
type ReplicationControllerDummy struct {
	metav1.TypeMeta
}

// Alpha-level support for Custom Metrics in HPA (as annotations).
type CustomMetricTarget struct {
	// Custom Metric name.
	Name string
	// Custom Metric value (average).
	TargetValue resource.Quantity
}

type CustomMetricTargetList struct {
	Items []CustomMetricTarget
}

type CustomMetricCurrentStatus struct {
	// Custom Metric name.
	Name string
	// Custom Metric value (average).
	CurrentValue resource.Quantity
}

type CustomMetricCurrentStatusList struct {
	Items []CustomMetricCurrentStatus
}

// +genclient=true
// +nonNamespaced=true

// A ThirdPartyResource is a generic representation of a resource, it is used by add-ons and plugins to add new resource
// types to the API.  It consists of one or more Versions of the api.
type ThirdPartyResource struct {
	metav1.TypeMeta

	// Standard object metadata
	// +optional
	metav1.ObjectMeta

	// Description is the description of this object.
	// +optional
	Description string

	// Versions are versions for this third party object
	Versions []APIVersion
}

type ThirdPartyResourceList struct {
	metav1.TypeMeta

	// Standard list metadata.
	// +optional
	metav1.ListMeta

	// Items is the list of horizontal pod autoscalers.
	Items []ThirdPartyResource
}

// An APIVersion represents a single concrete version of an object model.
// TODO: we should consider merge this struct with GroupVersion in metav1.go
type APIVersion struct {
	// Name of this version (e.g. 'v1').
	Name string
}

// An internal object, used for versioned storage in etcd.  Not exposed to the end user.
type ThirdPartyResourceData struct {
	metav1.TypeMeta
	// Standard object metadata.
	// +optional
	metav1.ObjectMeta

	// Data is the raw JSON data for this data.
	// +optional
	Data []byte
}

// +genclient=true

type Deployment struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the desired behavior of the Deployment.
	// +optional
	Spec DeploymentSpec

	// Most recently observed status of the Deployment.
	// +optional
	Status DeploymentStatus
}

type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	// +optional
	Replicas int32

	// Label selector for pods. Existing ReplicaSets whose pods are
	// selected by this will be the ones affected by this deployment.
	// +optional
	Selector *metav1.LabelSelector

	// Template describes the pods that will be created.
	Template api.PodTemplateSpec

	// The deployment strategy to use to replace existing pods with new ones.
	// +optional
	Strategy DeploymentStrategy

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	// +optional
	MinReadySeconds int32

	// The number of old ReplicaSets to retain to allow rollback.
	// This is a pointer to distinguish between explicit zero and not specified.
	// +optional
	RevisionHistoryLimit *int32

	// Indicates that the deployment is paused and will not be processed by the
	// deployment controller.
	// +optional
	Paused bool

	// The config this deployment is rolling back to. Will be cleared after rollback is done.
	// +optional
	RollbackTo *RollbackConfig

	// The maximum time in seconds for a deployment to make progress before it
	// is considered to be failed. The deployment controller will continue to
	// process failed deployments and a condition with a ProgressDeadlineExceeded
	// reason will be surfaced in the deployment status. Once autoRollback is
	// implemented, the deployment controller will automatically rollback failed
	// deployments. Note that progress will not be estimated during the time a
	// deployment is paused. This is not set by default.
	ProgressDeadlineSeconds *int32
}

// DeploymentRollback stores the information required to rollback a deployment.
type DeploymentRollback struct {
	metav1.TypeMeta
	// Required: This must match the Name of a deployment.
	Name string
	// The annotations to be updated to a deployment
	// +optional
	UpdatedAnnotations map[string]string
	// The config of this deployment rollback.
	RollbackTo RollbackConfig
}

type RollbackConfig struct {
	// The revision to rollback to. If set to 0, rollbck to the last revision.
	// +optional
	Revision int64
}

const (
	// DefaultDeploymentUniqueLabelKey is the default key of the selector that is added
	// to existing RCs (and label key that is added to its pods) to prevent the existing RCs
	// to select new pods (and old pods being select by new RC).
	DefaultDeploymentUniqueLabelKey string = "pod-template-hash"
)

type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	// +optional
	Type DeploymentStrategyType

	// Rolling update config params. Present only if DeploymentStrategyType =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	// +optional
	RollingUpdate *RollingUpdateDeployment
}

type DeploymentStrategyType string

const (
	// Kill all existing pods before creating new ones.
	RecreateDeploymentStrategyType DeploymentStrategyType = "Recreate"

	// Replace the old RCs by new one using rolling update i.e gradually scale down the old RCs and scale up the new one.
	RollingUpdateDeploymentStrategyType DeploymentStrategyType = "RollingUpdate"
)

// Spec to control the desired behavior of rolling update.
type RollingUpdateDeployment struct {
	// The maximum number of pods that can be unavailable during the update.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at the start of update (ex: 10%).
	// Absolute number is calculated from percentage by rounding down.
	// This can not be 0 if MaxSurge is 0.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, the old RC can be scaled down by 30%
	// immediately when the rolling update starts. Once new pods are ready, old RC
	// can be scaled down further, followed by scaling up the new RC, ensuring
	// that at least 70% of original number of pods are available at all times
	// during the update.
	// +optional
	MaxUnavailable intstr.IntOrString

	// The maximum number of pods that can be scheduled above the original number of
	// pods.
	// Value can be an absolute number (ex: 5) or a percentage of total pods at
	// the start of the update (ex: 10%). This can not be 0 if MaxUnavailable is 0.
	// Absolute number is calculated from percentage by rounding up.
	// By default, a value of 1 is used.
	// Example: when this is set to 30%, the new RC can be scaled up by 30%
	// immediately when the rolling update starts. Once old pods have been killed,
	// new RC can be scaled up further, ensuring that total number of pods running
	// at any time during the update is atmost 130% of original pods.
	// +optional
	MaxSurge intstr.IntOrString
}

type DeploymentStatus struct {
	// The generation observed by the deployment controller.
	// +optional
	ObservedGeneration int64

	// Total number of non-terminated pods targeted by this deployment (their labels match the selector).
	// +optional
	Replicas int32

	// Total number of non-terminated pods targeted by this deployment that have the desired template spec.
	// +optional
	UpdatedReplicas int32

	// Total number of ready pods targeted by this deployment.
	// +optional
	ReadyReplicas int32

	// Total number of available pods (ready for at least minReadySeconds) targeted by this deployment.
	// +optional
	AvailableReplicas int32

	// Total number of unavailable pods targeted by this deployment.
	// +optional
	UnavailableReplicas int32

	// Represents the latest available observations of a deployment's current state.
	Conditions []DeploymentCondition
}

type DeploymentConditionType string

// These are valid conditions of a deployment.
const (
	// Available means the deployment is available, ie. at least the minimum available
	// replicas required are up and running for at least minReadySeconds.
	DeploymentAvailable DeploymentConditionType = "Available"
	// Progressing means the deployment is progressing. Progress for a deployment is
	// considered when a new replica set is created or adopted, and when new pods scale
	// up or old pods scale down. Progress is not estimated for paused deployments or
	// when progressDeadlineSeconds is not specified.
	DeploymentProgressing DeploymentConditionType = "Progressing"
	// ReplicaFailure is added in a deployment when one of its pods fails to be created
	// or deleted.
	DeploymentReplicaFailure DeploymentConditionType = "ReplicaFailure"
)

// DeploymentCondition describes the state of a deployment at a certain point.
type DeploymentCondition struct {
	// Type of deployment condition.
	Type DeploymentConditionType
	// Status of the condition, one of True, False, Unknown.
	Status api.ConditionStatus
	// The last time this condition was updated.
	LastUpdateTime metav1.Time
	// Last time the condition transitioned from one status to another.
	LastTransitionTime metav1.Time
	// The reason for the condition's last transition.
	Reason string
	// A human readable message indicating details about the transition.
	Message string
}

type DeploymentList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	// Items is the list of deployments.
	Items []Deployment
}

type DaemonSetUpdateStrategy struct {
	// Type of daemon set update. Can be "RollingUpdate" or "OnDelete".
	// Default is OnDelete.
	// +optional
	Type DaemonSetUpdateStrategyType

	// Rolling update config params. Present only if type = "RollingUpdate".
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be. Same as DeploymentStrategy.RollingUpdate.
	// See https://github.com/kubernetes/kubernetes/issues/35345
	// +optional
	RollingUpdate *RollingUpdateDaemonSet
}

type DaemonSetUpdateStrategyType string

const (
	// Replace the old daemons by new ones using rolling update i.e replace them on each node one after the other.
	RollingUpdateDaemonSetStrategyType DaemonSetUpdateStrategyType = "RollingUpdate"

	// Replace the old daemons only when it's killed
	OnDeleteDaemonSetStrategyType DaemonSetUpdateStrategyType = "OnDelete"
)

// Spec to control the desired behavior of daemon set rolling update.
type RollingUpdateDaemonSet struct {
	// The maximum number of DaemonSet pods that can be unavailable during the
	// update. Value can be an absolute number (ex: 5) or a percentage of total
	// number of DaemonSet pods at the start of the update (ex: 10%). Absolute
	// number is calculated from percentage by rounding up.
	// This cannot be 0.
	// Default value is 1.
	// Example: when this is set to 30%, at most 30% of the total number of nodes
	// that should be running the daemon pod (i.e. status.desiredNumberScheduled)
	// can have their pods stopped for an update at any given
	// time. The update starts by stopping at most 30% of those DaemonSet pods
	// and then brings up new DaemonSet pods in their place. Once the new pods
	// are available, it then proceeds onto other DaemonSet pods, thus ensuring
	// that at least 70% of original number of DaemonSet pods are available at
	// all times during the update.
	// +optional
	MaxUnavailable intstr.IntOrString
}

// DaemonSetSpec is the specification of a daemon set.
type DaemonSetSpec struct {
	// A label query over pods that are managed by the daemon set.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector

	// An object that describes the pod that will be created.
	// The DaemonSet will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: https://kubernetes.io/docs/concepts/workloads/controllers/replicationcontroller#pod-template
	Template api.PodTemplateSpec

	// An update strategy to replace existing DaemonSet pods with new pods.
	// +optional
	UpdateStrategy DaemonSetUpdateStrategy

	// The minimum number of seconds for which a newly created DaemonSet pod should
	// be ready without any of its container crashing, for it to be considered
	// available. Defaults to 0 (pod will be considered available as soon as it
	// is ready).
	// +optional
	MinReadySeconds int32

	// A sequence number representing a specific generation of the template.
	// Populated by the system. It can be set only during the creation.
	// +optional
	TemplateGeneration int64
}

// DaemonSetStatus represents the current status of a daemon set.
type DaemonSetStatus struct {
	// The number of nodes that are running at least 1
	// daemon pod and are supposed to run the daemon pod.
	CurrentNumberScheduled int32

	// The number of nodes that are running the daemon pod, but are
	// not supposed to run the daemon pod.
	NumberMisscheduled int32

	// The total number of nodes that should be running the daemon
	// pod (including nodes correctly running the daemon pod).
	DesiredNumberScheduled int32

	// The number of nodes that should be running the daemon pod and have one
	// or more of the daemon pod running and ready.
	NumberReady int32

	// The most recent generation observed by the daemon set controller.
	// +optional
	ObservedGeneration int64

	// The total number of nodes that are running updated daemon pod
	// +optional
	UpdatedNumberScheduled int32

	// The number of nodes that should be running the
	// daemon pod and have one or more of the daemon pod running and
	// available (ready for at least spec.minReadySeconds)
	// +optional
	NumberAvailable int32

	// The number of nodes that should be running the
	// daemon pod and have none of the daemon pod running and available
	// (ready for at least spec.minReadySeconds)
	// +optional
	NumberUnavailable int32
}

// +genclient=true

// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// The desired behavior of this daemon set.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Spec DaemonSetSpec

	// The current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Status DaemonSetStatus
}

const (
	// DaemonSetTemplateGenerationKey is the key of the labels that is added
	// to daemon set pods to distinguish between old and new pod templates
	// during DaemonSet template update.
	DaemonSetTemplateGenerationKey string = "pod-template-generation"
)

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	metav1.TypeMeta
	// Standard list metadata.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// A list of daemon sets.
	Items []DaemonSet
}

type ThirdPartyResourceDataList struct {
	metav1.TypeMeta
	// Standard list metadata
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta
	// Items is a list of third party objects
	Items []ThirdPartyResourceData
}

// +genclient=true

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
type Ingress struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta

	// Spec is the desired state of the Ingress.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Spec IngressSpec

	// Status is the current state of the Ingress.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#spec-and-status
	// +optional
	Status IngressStatus
}

// IngressList is a collection of Ingress.
type IngressList struct {
	metav1.TypeMeta
	// Standard object's metadata.
	// More info: https://github.com/kubernetes/community/blob/master/contributors/devel/api-conventions.md#metadata
	// +optional
	metav1.ListMeta

	// Items is the list of Ingress.
	Items []Ingress
}

// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
	// A default backend capable of servicing requests that don't match any
	// rule. At least one of 'backend' or 'rules' must be specified. This field
	// is optional to allow the loadbalancer controller or defaulting logic to
	// specify a global default.
	// +optional
	Backend *IngressBackend

	// TLS configuration. Currently the Ingress only supports a single TLS
	// port, 443. If multiple members of this list specify different hosts, they
	// will be multiplexed on the same port according to the hostname specified
	// through the SNI TLS extension, if the ingress controller fulfilling the
	// ingress supports SNI.
	// +optional
	TLS []IngressTLS

	// A list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	// +optional
	Rules []IngressRule
	// TODO: Add the ability to specify load-balancer IP through claims
}

// IngressTLS describes the transport layer security associated with an Ingress.
type IngressTLS struct {
	// Hosts are a list of hosts included in the TLS certificate. The values in
	// this list must match the name/s used in the tlsSecret. Defaults to the
	// wildcard host setting for the loadbalancer controller fulfilling this
	// Ingress, if left unspecified.
	// +optional
	Hosts []string
	// SecretName is the name of the secret used to terminate SSL traffic on 443.
	// Field is left optional to allow SSL routing based on SNI hostname alone.
	// If the SNI host in a listener conflicts with the "Host" header field used
	// by an IngressRule, the SNI host is used for termination and value of the
	// Host header is used for routing.
	// +optional
	SecretName string
	// TODO: Consider specifying different modes of termination, protocols etc.
}

// IngressStatus describe the current state of the Ingress.
type IngressStatus struct {
	// LoadBalancer contains the current status of the load-balancer.
	// +optional
	LoadBalancer api.LoadBalancerStatus
}

// IngressRule represents the rules mapping the paths under a specified host to
// the related backend services. Incoming requests are first evaluated for a host
// match, then routed to the backend associated with the matching IngressRuleValue.
type IngressRule struct {
	// Host is the fully qualified domain name of a network host, as defined
	// by RFC 3986. Note the following deviations from the "host" part of the
	// URI as defined in the RFC:
	// 1. IPs are not allowed. Currently an IngressRuleValue can only apply to the
	//	  IP in the Spec of the parent Ingress.
	// 2. The `:` delimiter is not respected because ports are not allowed.
	//	  Currently the port of an Ingress is implicitly :80 for http and
	//	  :443 for https.
	// Both these may change in the future.
	// Incoming requests are matched against the host before the IngressRuleValue.
	// If the host is unspecified, the Ingress routes all traffic based on the
	// specified IngressRuleValue.
	// +optional
	Host string
	// IngressRuleValue represents a rule to route requests for this IngressRule.
	// If unspecified, the rule defaults to a http catch-all. Whether that sends
	// just traffic matching the host to the default backend or all traffic to the
	// default backend, is left to the controller fulfilling the Ingress. Http is
	// currently the only supported IngressRuleValue.
	// +optional
	IngressRuleValue
}

// IngressRuleValue represents a rule to apply against incoming requests. If the
// rule is satisfied, the request is routed to the specified backend. Currently
// mixing different types of rules in a single Ingress is disallowed, so exactly
// one of the following must be set.
type IngressRuleValue struct {
	//TODO:
	// 1. Consider renaming this resource and the associated rules so they
	// aren't tied to Ingress. They can be used to route intra-cluster traffic.
	// 2. Consider adding fields for ingress-type specific global options
	// usable by a loadbalancer, like http keep-alive.

	// +optional
	HTTP *HTTPIngressRuleValue
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// A collection of paths that map requests to backends.
	Paths []HTTPIngressPath
	// TODO: Consider adding fields for ingress-type specific global
	// options usable by a loadbalancer, like http keep-alive.
}

// HTTPIngressPath associates a path regex with a backend. Incoming urls matching
// the path are forwarded to the backend.
type HTTPIngressPath struct {
	// Path is an extended POSIX regex as defined by IEEE Std 1003.1,
	// (i.e this follows the egrep/unix syntax, not the perl syntax)
	// matched against the path of an incoming request. Currently it can
	// contain characters disallowed from the conventional "path"
	// part of a URL as defined by RFC 3986. Paths must begin with
	// a '/'. If unspecified, the path defaults to a catch all sending
	// traffic to the backend.
	// +optional
	Path string

	// Backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// Specifies the name of the referenced service.
	ServiceName string

	// Specifies the port of the referenced service.
	ServicePort intstr.IntOrString
}

// +genclient=true

// ReplicaSet represents the configuration of a replica set.
type ReplicaSet struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the desired behavior of this ReplicaSet.
	// +optional
	Spec ReplicaSetSpec

	// Status is the current status of this ReplicaSet. This data may be
	// out of date by some window of time.
	// +optional
	Status ReplicaSetStatus
}

// ReplicaSetList is a collection of ReplicaSets.
type ReplicaSetList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []ReplicaSet
}

// ReplicaSetSpec is the specification of a ReplicaSet.
// As the internal representation of a ReplicaSet, it must have
// a Template set.
type ReplicaSetSpec struct {
	// Replicas is the number of desired replicas.
	Replicas int32

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	// +optional
	MinReadySeconds int32

	// Selector is a label query over pods that should match the replica count.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on pod template.
	// More info: https://kubernetes.io/docs/concepts/overview/working-with-objects/labels/#label-selectors
	// +optional
	Selector *metav1.LabelSelector

	// Template is the object that describes the pod that will be created if
	// insufficient replicas are detected.
	// +optional
	Template api.PodTemplateSpec
}

// ReplicaSetStatus represents the current status of a ReplicaSet.
type ReplicaSetStatus struct {
	// Replicas is the number of actual replicas.
	Replicas int32

	// The number of pods that have labels matching the labels of the pod template of the replicaset.
	// +optional
	FullyLabeledReplicas int32

	// The number of ready replicas for this replica set.
	// +optional
	ReadyReplicas int32

	// The number of available replicas (ready for at least minReadySeconds) for this replica set.
	// +optional
	AvailableReplicas int32

	// ObservedGeneration is the most recent generation observed by the controller.
	// +optional
	ObservedGeneration int64

	// Represents the latest available observations of a replica set's current state.
	// +optional
	Conditions []ReplicaSetCondition
}

type ReplicaSetConditionType string

// These are valid conditions of a replica set.
const (
	// ReplicaSetReplicaFailure is added in a replica set when one of its pods fails to be created
	// due to insufficient quota, limit ranges, pod security policy, node selectors, etc. or deleted
	// due to kubelet being down or finalizers are failing.
	ReplicaSetReplicaFailure ReplicaSetConditionType = "ReplicaFailure"
)

// ReplicaSetCondition describes the state of a replica set at a certain point.
type ReplicaSetCondition struct {
	// Type of replica set condition.
	Type ReplicaSetConditionType
	// Status of the condition, one of True, False, Unknown.
	Status api.ConditionStatus
	// The last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time
	// The reason for the condition's last transition.
	// +optional
	Reason string
	// A human readable message indicating details about the transition.
	// +optional
	Message string
}

// +genclient=true
// +nonNamespaced=true

// PodSecurityPolicy governs the ability to make requests that affect the SecurityContext
// that will be applied to a pod and container.
type PodSecurityPolicy struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the policy enforced.
	// +optional
	Spec PodSecurityPolicySpec
}

// PodSecurityPolicySpec defines the policy enforced.
type PodSecurityPolicySpec struct {
	// Privileged determines if a pod can request to be run as privileged.
	// +optional
	Privileged bool
	// DefaultAddCapabilities is the default set of capabilities that will be added to the container
	// unless the pod spec specifically drops the capability.  You may not list a capability in both
	// DefaultAddCapabilities and RequiredDropCapabilities.
	// +optional
	DefaultAddCapabilities []api.Capability
	// RequiredDropCapabilities are the capabilities that will be dropped from the container.  These
	// are required to be dropped and cannot be added.
	// +optional
	RequiredDropCapabilities []api.Capability
	// AllowedCapabilities is a list of capabilities that can be requested to add to the container.
	// Capabilities in this field may be added at the pod author's discretion.
	// You must not list a capability in both AllowedCapabilities and RequiredDropCapabilities.
	// +optional
	AllowedCapabilities []api.Capability
	// Volumes is a white list of allowed volume plugins.  Empty indicates that all plugins
	// may be used.
	// +optional
	Volumes []FSType
	// HostNetwork determines if the policy allows the use of HostNetwork in the pod spec.
	// +optional
	HostNetwork bool
	// HostPorts determines which host port ranges are allowed to be exposed.
	// +optional
	HostPorts []HostPortRange
	// HostPID determines if the policy allows the use of HostPID in the pod spec.
	// +optional
	HostPID bool
	// HostIPC determines if the policy allows the use of HostIPC in the pod spec.
	// +optional
	HostIPC bool
	// SELinux is the strategy that will dictate the allowable labels that may be set.
	SELinux SELinuxStrategyOptions
	// RunAsUser is the strategy that will dictate the allowable RunAsUser values that may be set.
	RunAsUser RunAsUserStrategyOptions
	// SupplementalGroups is the strategy that will dictate what supplemental groups are used by the SecurityContext.
	SupplementalGroups SupplementalGroupsStrategyOptions
	// FSGroup is the strategy that will dictate what fs group is used by the SecurityContext.
	FSGroup FSGroupStrategyOptions
	// ReadOnlyRootFilesystem when set to true will force containers to run with a read only root file
	// system.  If the container specifically requests to run with a non-read only root file system
	// the PSP should deny the pod.
	// If set to false the container may run with a read only root file system if it wishes but it
	// will not be forced to.
	// +optional
	ReadOnlyRootFilesystem bool
}

// HostPortRange defines a range of host ports that will be enabled by a policy
// for pods to use.  It requires both the start and end to be defined.
type HostPortRange struct {
	// Min is the start of the range, inclusive.
	Min int
	// Max is the end of the range, inclusive.
	Max int
}

// FSType gives strong typing to different file systems that are used by volumes.
type FSType string

var (
	AzureFile             FSType = "azureFile"
	Flocker               FSType = "flocker"
	FlexVolume            FSType = "flexVolume"
	HostPath              FSType = "hostPath"
	EmptyDir              FSType = "emptyDir"
	GCEPersistentDisk     FSType = "gcePersistentDisk"
	AWSElasticBlockStore  FSType = "awsElasticBlockStore"
	GitRepo               FSType = "gitRepo"
	Secret                FSType = "secret"
	NFS                   FSType = "nfs"
	ISCSI                 FSType = "iscsi"
	Glusterfs             FSType = "glusterfs"
	PersistentVolumeClaim FSType = "persistentVolumeClaim"
	RBD                   FSType = "rbd"
	Cinder                FSType = "cinder"
	CephFS                FSType = "cephFS"
	DownwardAPI           FSType = "downwardAPI"
	FC                    FSType = "fc"
	ConfigMap             FSType = "configMap"
	VsphereVolume         FSType = "vsphereVolume"
	Quobyte               FSType = "quobyte"
	AzureDisk             FSType = "azureDisk"
	PhotonPersistentDisk  FSType = "photonPersistentDisk"
	Projected             FSType = "projected"
	PortworxVolume        FSType = "portworxVolume"
	ScaleIO               FSType = "scaleIO"
	All                   FSType = "*"
)

// SELinuxStrategyOptions defines the strategy type and any options used to create the strategy.
type SELinuxStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable labels that may be set.
	Rule SELinuxStrategy
	// seLinuxOptions required to run as; required for MustRunAs
	// More info: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/security_context.md
	// +optional
	SELinuxOptions *api.SELinuxOptions
}

// SELinuxStrategy denotes strategy types for generating SELinux options for a
// Security.
type SELinuxStrategy string

const (
	// container must have SELinux labels of X applied.
	SELinuxStrategyMustRunAs SELinuxStrategy = "MustRunAs"
	// container may make requests for any SELinux context labels.
	SELinuxStrategyRunAsAny SELinuxStrategy = "RunAsAny"
)

// RunAsUserStrategyOptions defines the strategy type and any options used to create the strategy.
type RunAsUserStrategyOptions struct {
	// Rule is the strategy that will dictate the allowable RunAsUser values that may be set.
	Rule RunAsUserStrategy
	// Ranges are the allowed ranges of uids that may be used.
	// +optional
	Ranges []UserIDRange
}

// UserIDRange provides a min/max of an allowed range of UserIDs.
type UserIDRange struct {
	// Min is the start of the range, inclusive.
	Min types.UnixUserID
	// Max is the end of the range, inclusive.
	Max types.UnixUserID
}

// GroupIDRange provides a min/max of an allowed range of GroupIDs.
type GroupIDRange struct {
	// Min is the start of the range, inclusive.
	Min types.UnixGroupID
	// Max is the end of the range, inclusive.
	Max types.UnixGroupID
}

// RunAsUserStrategy denotes strategy types for generating RunAsUser values for a
// SecurityContext.
type RunAsUserStrategy string

const (
	// container must run as a particular uid.
	RunAsUserStrategyMustRunAs RunAsUserStrategy = "MustRunAs"
	// container must run as a non-root uid
	RunAsUserStrategyMustRunAsNonRoot RunAsUserStrategy = "MustRunAsNonRoot"
	// container may make requests for any uid.
	RunAsUserStrategyRunAsAny RunAsUserStrategy = "RunAsAny"
)

// FSGroupStrategyOptions defines the strategy type and options used to create the strategy.
type FSGroupStrategyOptions struct {
	// Rule is the strategy that will dictate what FSGroup is used in the SecurityContext.
	// +optional
	Rule FSGroupStrategyType
	// Ranges are the allowed ranges of fs groups.  If you would like to force a single
	// fs group then supply a single range with the same start and end.
	// +optional
	Ranges []GroupIDRange
}

// FSGroupStrategyType denotes strategy types for generating FSGroup values for a
// SecurityContext
type FSGroupStrategyType string

const (
	// container must have FSGroup of X applied.
	FSGroupStrategyMustRunAs FSGroupStrategyType = "MustRunAs"
	// container may make requests for any FSGroup labels.
	FSGroupStrategyRunAsAny FSGroupStrategyType = "RunAsAny"
)

// SupplementalGroupsStrategyOptions defines the strategy type and options used to create the strategy.
type SupplementalGroupsStrategyOptions struct {
	// Rule is the strategy that will dictate what supplemental groups is used in the SecurityContext.
	// +optional
	Rule SupplementalGroupsStrategyType
	// Ranges are the allowed ranges of supplemental groups.  If you would like to force a single
	// supplemental group then supply a single range with the same start and end.
	// +optional
	Ranges []GroupIDRange
}

// SupplementalGroupsStrategyType denotes strategy types for determining valid supplemental
// groups for a SecurityContext.
type SupplementalGroupsStrategyType string

const (
	// container must run as a particular gid.
	SupplementalGroupsStrategyMustRunAs SupplementalGroupsStrategyType = "MustRunAs"
	// container may make requests for any gid.
	SupplementalGroupsStrategyRunAsAny SupplementalGroupsStrategyType = "RunAsAny"
)

// PodSecurityPolicyList is a list of PodSecurityPolicy objects.
type PodSecurityPolicyList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []PodSecurityPolicy
}

// +genclient=true

type NetworkPolicy struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Specification of the desired behavior for this NetworkPolicy.
	// +optional
	Spec NetworkPolicySpec
}

type NetworkPolicySpec struct {
	// Selects the pods to which this NetworkPolicy object applies.  The array of ingress rules
	// is applied to any pods selected by this field. Multiple network policies can select the
	// same set of pods.  In this case, the ingress rules for each are combined additively.
	// This field is NOT optional and follows standard label selector semantics.
	// An empty podSelector matches all pods in this namespace.
	PodSelector metav1.LabelSelector

	// List of ingress rules to be applied to the selected pods.
	// Traffic is allowed to a pod if namespace.networkPolicy.ingress.isolation is undefined and cluster policy allows it,
	// OR if the traffic source is the pod's local node,
	// OR if the traffic matches at least one ingress rule across all of the NetworkPolicy
	// objects whose podSelector matches the pod.
	// If this field is empty then this NetworkPolicy does not affect ingress isolation.
	// If this field is present and contains at least one rule, this policy allows any traffic
	// which matches at least one of the ingress rules in this list.
	// +optional
	Ingress []NetworkPolicyIngressRule
}

// This NetworkPolicyIngressRule matches traffic if and only if the traffic matches both ports AND from.
type NetworkPolicyIngressRule struct {
	// List of ports which should be made accessible on the pods selected for this rule.
	// Each item in this list is combined using a logical OR.
	// If this field is empty or missing, this rule matches all ports (traffic not restricted by port).
	// If this field is present and contains at least one item, then this rule allows traffic
	// only if the traffic matches at least one port in the list.
	// +optional
	Ports []NetworkPolicyPort

	// List of sources which should be able to access the pods selected for this rule.
	// Items in this list are combined using a logical OR operation.
	// If this field is empty or missing, this rule matches all sources (traffic not restricted by source).
	// If this field is present and contains at least on item, this rule allows traffic only if the
	// traffic matches at least one item in the from list.
	// +optional
	From []NetworkPolicyPeer
}

type NetworkPolicyPort struct {
	// Optional.  The protocol (TCP or UDP) which traffic must match.
	// If not specified, this field defaults to TCP.
	// +optional
	Protocol *api.Protocol

	// If specified, the port on the given protocol.  This can
	// either be a numerical or named port on a pod.  If this field is not provided,
	// this matches all port names and numbers.
	// If present, only traffic on the specified protocol AND port
	// will be matched.
	// +optional
	Port *intstr.IntOrString
}

type NetworkPolicyPeer struct {
	// Exactly one of the following must be specified.

	// This is a label selector which selects Pods in this namespace.
	// This field follows standard label selector semantics.
	// If present but empty, this selector selects all pods in this namespace.
	// +optional
	PodSelector *metav1.LabelSelector

	// Selects Namespaces using cluster scoped-labels.  This
	// matches all pods in all namespaces selected by this label selector.
	// This field follows standard label selector semantics.
	// If present but empty, this selector selects all namespaces.
	// +optional
	NamespaceSelector *metav1.LabelSelector
}

// NetworkPolicyList is a list of NetworkPolicy objects.
type NetworkPolicyList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []NetworkPolicy
}

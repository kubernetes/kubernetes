/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package v1beta1

import (
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/util"
)

// describes the attributes of a scale subresource
type ScaleSpec struct {
	// desired number of instances for the scaled object.
	Replicas int `json:"replicas,omitempty"`
}

// represents the current status of a scale subresource.
type ScaleStatus struct {
	// actual number of observed instances of the scaled object.
	Replicas int `json:"replicas"`

	// label query over pods that should match the replicas count. More info: http://releases.k8s.io/release-1.1/docs/user-guide/labels.md#label-selectors
	Selector map[string]string `json:"selector,omitempty"`
}

// represents a scaling request for a resource.
type Scale struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata; More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata.
	v1.ObjectMeta `json:"metadata,omitempty"`

	// defines the behavior of the scale. More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status.
	Spec ScaleSpec `json:"spec,omitempty"`

	// current status of the scale. More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status. Read-only.
	Status ScaleStatus `json:"status,omitempty"`
}

// Dummy definition
type ReplicationControllerDummy struct {
	unversioned.TypeMeta `json:",inline"`
}

// SubresourceReference contains enough information to let you inspect or modify the referred subresource.
type SubresourceReference struct {
	// Kind of the referent; More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#types-kinds"
	Kind string `json:"kind,omitempty"`
	// Name of the referent; More info: http://releases.k8s.io/release-1.1/docs/user-guide/identifiers.md#names
	Name string `json:"name,omitempty"`
	// API version of the referent
	APIVersion string `json:"apiVersion,omitempty"`
	// Subresource name of the referent
	Subresource string `json:"subresource,omitempty"`
}

type CPUTargetUtilization struct {
	// fraction of the requested CPU that should be utilized/used,
	// e.g. 70 means that 70% of the requested CPU should be in use.
	TargetPercentage int `json:"targetPercentage"`
}

// specification of a horizontal pod autoscaler.
type HorizontalPodAutoscalerSpec struct {
	// reference to Scale subresource; horizontal pod autoscaler will learn the current resource consumption from its status,
	// and will set the desired number of pods by modifying its spec.
	ScaleRef SubresourceReference `json:"scaleRef"`
	// lower limit for the number of pods that can be set by the autoscaler, default 1.
	MinReplicas *int `json:"minReplicas,omitempty"`
	// upper limit for the number of pods that can be set by the autoscaler; cannot be smaller than MinReplicas.
	MaxReplicas int `json:"maxReplicas"`
	// target average CPU utilization (represented as a percentage of requested CPU) over all the pods;
	// if not specified it defaults to the target CPU utilization at 80% of the requested resources.
	CPUUtilization *CPUTargetUtilization `json:"cpuUtilization,omitempty"`
}

// current status of a horizontal pod autoscaler
type HorizontalPodAutoscalerStatus struct {
	// most recent generation observed by this autoscaler.
	ObservedGeneration *int64 `json:"observedGeneration,omitempty"`

	// last time the HorizontalPodAutoscaler scaled the number of pods;
	// used by the autoscaler to control how often the number of pods is changed.
	LastScaleTime *unversioned.Time `json:"lastScaleTime,omitempty"`

	// current number of replicas of pods managed by this autoscaler.
	CurrentReplicas int `json:"currentReplicas"`

	// desired number of replicas of pods managed by this autoscaler.
	DesiredReplicas int `json:"desiredReplicas"`

	// current average CPU utilization over all pods, represented as a percentage of requested CPU,
	// e.g. 70 means that an average pod is using now 70% of its requested CPU.
	CurrentCPUUtilizationPercentage *int `json:"currentCPUUtilizationPercentage,omitempty"`
}

// configuration of a horizontal pod autoscaler.
type HorizontalPodAutoscaler struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata. More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// behaviour of autoscaler. More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status.
	Spec HorizontalPodAutoscalerSpec `json:"spec,omitempty"`

	// current information about the autoscaler.
	Status HorizontalPodAutoscalerStatus `json:"status,omitempty"`
}

// list of horizontal pod autoscaler objects.
type HorizontalPodAutoscalerList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	unversioned.ListMeta `json:"metadata,omitempty"`

	// list of horizontal pod autoscaler objects.
	Items []HorizontalPodAutoscaler `json:"items"`
}

// A ThirdPartyResource is a generic representation of a resource, it is used by add-ons and plugins to add new resource
// types to the API.  It consists of one or more Versions of the api.
type ThirdPartyResource struct {
	unversioned.TypeMeta `json:",inline"`

	// Standard object metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Description is the description of this object.
	Description string `json:"description,omitempty"`

	// Versions are versions for this third party object
	Versions []APIVersion `json:"versions,omitempty"`
}

// ThirdPartyResourceList is a list of ThirdPartyResources.
type ThirdPartyResourceList struct {
	unversioned.TypeMeta `json:",inline"`

	// Standard list metadata.
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of ThirdPartyResources.
	Items []ThirdPartyResource `json:"items"`
}

// An APIVersion represents a single concrete version of an object model.
type APIVersion struct {
	// Name of this version (e.g. 'v1').
	Name string `json:"name,omitempty"`

	// The API group to add this object into, default 'experimental'.
	APIGroup string `json:"apiGroup,omitempty"`
}

// An internal object, used for versioned storage in etcd.  Not exposed to the end user.
type ThirdPartyResourceData struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata.
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Data is the raw JSON data for this data.
	Data []byte `json:"name,omitempty"`
}

// Deployment enables declarative updates for Pods and ReplicationControllers.
type Deployment struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object metadata.
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Specification of the desired behavior of the Deployment.
	Spec DeploymentSpec `json:"spec,omitempty"`

	// Most recently observed status of the Deployment.
	Status DeploymentStatus `json:"status,omitempty"`
}

// DeploymentSpec is the specification of the desired behavior of the Deployment.
type DeploymentSpec struct {
	// Number of desired pods. This is a pointer to distinguish between explicit
	// zero and not specified. Defaults to 1.
	Replicas *int `json:"replicas,omitempty"`

	// Label selector for pods. Existing ReplicationControllers whose pods are
	// selected by this will be the ones affected by this deployment.
	Selector map[string]string `json:"selector,omitempty"`

	// Template describes the pods that will be created.
	Template *v1.PodTemplateSpec `json:"template,omitempty"`

	// The deployment strategy to use to replace existing pods with new ones.
	Strategy DeploymentStrategy `json:"strategy,omitempty"`

	// Key of the selector that is added to existing RCs (and label key that is
	// added to its pods) to prevent the existing RCs to select new pods (and old
	// pods being selected by new RC).
	// Users can set this to an empty string to indicate that the system should
	// not add any selector and label. If unspecified, system uses
	// "deployment.kubernetes.io/podTemplateHash".
	// Value of this key is hash of DeploymentSpec.PodTemplateSpec.
	// No label is added if this is set to empty string.
	UniqueLabelKey *string `json:"uniqueLabelKey,omitempty"`
}

// DeploymentStrategy describes how to replace existing pods with new ones.
type DeploymentStrategy struct {
	// Type of deployment. Can be "Recreate" or "RollingUpdate". Default is RollingUpdate.
	Type DeploymentStrategyType `json:"type,omitempty"`

	// Rolling update config params. Present only if DeploymentStrategyType =
	// RollingUpdate.
	//---
	// TODO: Update this to follow our convention for oneOf, whatever we decide it
	// to be.
	RollingUpdate *RollingUpdateDeployment `json:"rollingUpdate,omitempty"`
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
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// Absolute number is calculated from percentage by rounding up.
	// This can not be 0 if MaxSurge is 0.
	// By default, a fixed value of 1 is used.
	// Example: when this is set to 30%, the old RC can be scaled down to 70% of desired pods
	// immediately when the rolling update starts. Once new pods are ready, old RC
	// can be scaled down further, followed by scaling up the new RC, ensuring
	// that the total number of pods available at all times during the update is at
	// least 70% of desired pods.
	MaxUnavailable *util.IntOrString `json:"maxUnavailable,omitempty"`

	// The maximum number of pods that can be scheduled above the desired number of
	// pods.
	// Value can be an absolute number (ex: 5) or a percentage of desired pods (ex: 10%).
	// This can not be 0 if MaxUnavailable is 0.
	// Absolute number is calculated from percentage by rounding up.
	// By default, a value of 1 is used.
	// Example: when this is set to 30%, the new RC can be scaled up immediately when
	// the rolling update starts, such that the total number of old and new pods do not exceed
	// 130% of desired pods. Once old pods have been killed,
	// new RC can be scaled up further, ensuring that total number of pods running
	// at any time during the update is atmost 130% of desired pods.
	MaxSurge *util.IntOrString `json:"maxSurge,omitempty"`

	// Minimum number of seconds for which a newly created pod should be ready
	// without any of its container crashing, for it to be considered available.
	// Defaults to 0 (pod will be considered available as soon as it is ready)
	MinReadySeconds int `json:"minReadySeconds,omitempty"`
}

// DeploymentStatus is the most recently observed status of the Deployment.
type DeploymentStatus struct {
	// Total number of non-terminated pods targeted by this deployment (their labels match the selector).
	Replicas int `json:"replicas,omitempty"`

	// Total number of non-terminated pods targeted by this deployment that have the desired template spec.
	UpdatedReplicas int `json:"updatedReplicas,omitempty"`
}

// DeploymentList is a list of Deployments.
type DeploymentList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of Deployments.
	Items []Deployment `json:"items"`
}

// DaemonSetSpec is the specification of a daemon set.
type DaemonSetSpec struct {
	// Selector is a label query over pods that are managed by the daemon set.
	// Must match in order to be controlled.
	// If empty, defaulted to labels on Pod template.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/labels.md#label-selectors
	Selector map[string]string `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created.
	// The DaemonSet will create exactly one copy of this pod on every node
	// that matches the template's node selector (or on every node if no node
	// selector is specified).
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/replication-controller.md#pod-template
	Template *v1.PodTemplateSpec `json:"template,omitempty"`
}

// DaemonSetStatus represents the current status of a daemon set.
type DaemonSetStatus struct {
	// CurrentNumberScheduled is the number of nodes that are running at least 1
	// daemon pod and are supposed to run the daemon pod.
	// More info: http://releases.k8s.io/release-1.1/docs/admin/daemon.md
	CurrentNumberScheduled int `json:"currentNumberScheduled"`

	// NumberMisscheduled is the number of nodes that are running the daemon pod, but are
	// not supposed to run the daemon pod.
	// More info: http://releases.k8s.io/release-1.1/docs/admin/daemon.md
	NumberMisscheduled int `json:"numberMisscheduled"`

	// DesiredNumberScheduled is the total number of nodes that should be running the daemon
	// pod (including nodes correctly running the daemon pod).
	// More info: http://releases.k8s.io/release-1.1/docs/admin/daemon.md
	DesiredNumberScheduled int `json:"desiredNumberScheduled"`
}

// DaemonSet represents the configuration of a daemon set.
type DaemonSet struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Spec defines the desired behavior of this daemon set.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Spec DaemonSetSpec `json:"spec,omitempty"`

	// Status is the current status of this daemon set. This data may be
	// out of date by some window of time.
	// Populated by the system.
	// Read-only.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Status DaemonSetStatus `json:"status,omitempty"`
}

// DaemonSetList is a collection of daemon sets.
type DaemonSetList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is a list of daemon sets.
	Items []DaemonSet `json:"items"`
}

// ThirdPartyResrouceDataList is a list of ThirdPartyResourceData.
type ThirdPartyResourceDataList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of ThirdpartyResourceData.
	Items []ThirdPartyResourceData `json:"items"`
}

// Job represents the configuration of a single job.
type Job struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Spec is a structure defining the expected behavior of a job.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Spec JobSpec `json:"spec,omitempty"`

	// Status is a structure describing current status of a job.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Status JobStatus `json:"status,omitempty"`
}

// JobList is a collection of jobs.
type JobList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard list metadata
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of Job.
	Items []Job `json:"items"`
}

// JobSpec describes how the job execution will look like.
type JobSpec struct {

	// Parallelism specifies the maximum desired number of pods the job should
	// run at any given time. The actual number of pods running in steady state will
	// be less than this number when ((.spec.completions - .status.successful) < .spec.parallelism),
	// i.e. when the work left to do is less than max parallelism.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/jobs.md
	Parallelism *int `json:"parallelism,omitempty"`

	// Completions specifies the desired number of successfully finished pods the
	// job should be run with. Defaults to 1.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/jobs.md
	Completions *int `json:"completions,omitempty"`

	// Selector is a label query over pods that should match the pod count.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/labels.md#label-selectors
	Selector *PodSelector `json:"selector,omitempty"`

	// Template is the object that describes the pod that will be created when
	// executing a job.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/jobs.md
	Template v1.PodTemplateSpec `json:"template"`
}

// JobStatus represents the current state of a Job.
type JobStatus struct {

	// Conditions represent the latest available observations of an object's current state.
	// More info: http://releases.k8s.io/release-1.1/docs/user-guide/jobs.md
	Conditions []JobCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`

	// StartTime represents time when the job was acknowledged by the Job Manager.
	// It is not guaranteed to be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	StartTime *unversioned.Time `json:"startTime,omitempty"`

	// CompletionTime represents time when the job was completed. It is not guaranteed to
	// be set in happens-before order across separate operations.
	// It is represented in RFC3339 form and is in UTC.
	CompletionTime *unversioned.Time `json:"completionTime,omitempty"`

	// Active is the number of actively running pods.
	Active int `json:"active,omitempty"`

	// Succeeded is the number of pods which reached Phase Succeeded.
	Succeeded int `json:"succeeded,omitempty"`

	// Failed is the number of pods which reached Phase Failed.
	Failed int `json:"failed,omitempty"`
}

type JobConditionType string

// These are valid conditions of a job.
const (
	// JobComplete means the job has completed its execution.
	JobComplete JobConditionType = "Complete"
)

// JobCondition describes current state of a job.
type JobCondition struct {
	// Type of job condition, currently only Complete.
	Type JobConditionType `json:"type"`
	// Status of the condition, one of True, False, Unknown.
	Status v1.ConditionStatus `json:"status"`
	// Last time the condition was checked.
	LastProbeTime unversioned.Time `json:"lastProbeTime,omitempty"`
	// Last time the condition transit from one status to another.
	LastTransitionTime unversioned.Time `json:"lastTransitionTime,omitempty"`
	// (brief) reason for the condition's last transition.
	Reason string `json:"reason,omitempty"`
	// Human readable message indicating details about last transition.
	Message string `json:"message,omitempty"`
}

// Ingress is a collection of rules that allow inbound connections to reach the
// endpoints defined by a backend. An Ingress can be configured to give services
// externally-reachable urls, load balance traffic, terminate SSL, offer name
// based virtual hosting etc.
type Ingress struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	v1.ObjectMeta `json:"metadata,omitempty"`

	// Spec is the desired state of the Ingress.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Spec IngressSpec `json:"spec,omitempty"`

	// Status is the current state of the Ingress.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#spec-and-status
	Status IngressStatus `json:"status,omitempty"`
}

// IngressList is a collection of Ingress.
type IngressList struct {
	unversioned.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: http://releases.k8s.io/release-1.1/docs/devel/api-conventions.md#metadata
	unversioned.ListMeta `json:"metadata,omitempty"`

	// Items is the list of Ingress.
	Items []Ingress `json:"items"`
}

// IngressSpec describes the Ingress the user wishes to exist.
type IngressSpec struct {
	// A default backend capable of servicing requests that don't match any
	// rule. At least one of 'backend' or 'rules' must be specified. This field
	// is optional to allow the loadbalancer controller or defaulting logic to
	// specify a global default.
	Backend *IngressBackend `json:"backend,omitempty"`
	// A list of host rules used to configure the Ingress. If unspecified, or
	// no rule matches, all traffic is sent to the default backend.
	Rules []IngressRule `json:"rules,omitempty"`
	// TODO: Add the ability to specify load-balancer IP through claims
}

// IngressStatus describe the current state of the Ingress.
type IngressStatus struct {
	// LoadBalancer contains the current status of the load-balancer.
	LoadBalancer v1.LoadBalancerStatus `json:"loadBalancer,omitempty"`
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
	Host string `json:"host,omitempty"`
	// IngressRuleValue represents a rule to route requests for this IngressRule.
	// If unspecified, the rule defaults to a http catch-all. Whether that sends
	// just traffic matching the host to the default backend or all traffic to the
	// default backend, is left to the controller fulfilling the Ingress. Http is
	// currently the only supported IngressRuleValue.
	IngressRuleValue `json:",inline,omitempty"`
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

	HTTP *HTTPIngressRuleValue `json:"http,omitempty"`
}

// HTTPIngressRuleValue is a list of http selectors pointing to backends.
// In the example: http://<host>/<path>?<searchpart> -> backend where
// where parts of the url correspond to RFC 3986, this resource will be used
// to match against everything after the last '/' and before the first '?'
// or '#'.
type HTTPIngressRuleValue struct {
	// A collection of paths that map requests to backends.
	Paths []HTTPIngressPath `json:"paths"`
	// TODO: Consider adding fields for ingress-type specific global
	// options usable by a loadbalancer, like http keep-alive.
}

// HTTPIngressPath associates a path regex with a backend. Incoming urls matching
// the path are forwarded to the backend.
type HTTPIngressPath struct {
	// Path is a extended POSIX regex as defined by IEEE Std 1003.1,
	// (i.e this follows the egrep/unix syntax, not the perl syntax)
	// matched against the path of an incoming request. Currently it can
	// contain characters disallowed from the conventional "path"
	// part of a URL as defined by RFC 3986. Paths must begin with
	// a '/'. If unspecified, the path defaults to a catch all sending
	// traffic to the backend.
	Path string `json:"path,omitempty"`

	// Backend defines the referenced service endpoint to which the traffic
	// will be forwarded to.
	Backend IngressBackend `json:"backend"`
}

// IngressBackend describes all endpoints for a given service and port.
type IngressBackend struct {
	// Specifies the name of the referenced service.
	ServiceName string `json:"serviceName"`

	// Specifies the port of the referenced service.
	ServicePort util.IntOrString `json:"servicePort"`
}

// A pod selector is a label query over a set of pods. The result of matchLabels and
// matchExpressions are ANDed. An empty pod selector matches all objects. A null
// pod selector matches no objects.
type PodSelector struct {
	// matchLabels is a map of {key,value} pairs. A single {key,value} in the matchLabels
	// map is equivalent to an element of matchExpressions, whose key field is "key", the
	// operator is "In", and the values array contains only "value". The requirements are ANDed.
	MatchLabels map[string]string `json:"matchLabels,omitempty"`
	// matchExpressions is a list of pod selector requirements. The requirements are ANDed.
	MatchExpressions []PodSelectorRequirement `json:"matchExpressions,omitempty"`
}

// A pod selector requirement is a selector that contains values, a key, and an operator that
// relates the key and values.
type PodSelectorRequirement struct {
	// key is the label key that the selector applies to.
	Key string `json:"key" patchStrategy:"merge" patchMergeKey:"key"`
	// operator represents a key's relationship to a set of values.
	// Valid operators ard In, NotIn, Exists and DoesNotExist.
	Operator PodSelectorOperator `json:"operator"`
	// values is an array of string values. If the operator is In or NotIn,
	// the values array must be non-empty. If the operator is Exists or DoesNotExist,
	// the values array must be empty. This array is replaced during a strategic
	// merge patch.
	Values []string `json:"values,omitempty"`
}

// A pod selector operator is the set of operators that can be used in a selector requirement.
type PodSelectorOperator string

const (
	PodSelectorOpIn           PodSelectorOperator = "In"
	PodSelectorOpNotIn        PodSelectorOperator = "NotIn"
	PodSelectorOpExists       PodSelectorOperator = "Exists"
	PodSelectorOpDoesNotExist PodSelectorOperator = "DoesNotExist"
)

/*
Copyright 2017 The Kubernetes Authors.

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

package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type (
	ConditionStatus   string
	CarpConditionType string
	CarpPhase         string
	RestartPolicy     string
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.1

// Carp is a collection of containers, used as either input (create, update) or as output (list, get).
type Carp struct {
	metav1.TypeMeta `json:",inline"`
	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Specification of the desired behavior of the carp.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Spec CarpSpec `json:"spec,omitempty" protobuf:"bytes,2,opt,name=spec"`

	// Most recently observed status of the carp.
	// This data may not be up to date.
	// Populated by the system.
	// Read-only.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status CarpStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// CarpStatus represents information about the status of a carp. Status may trail the actual
// state of a system.
type CarpStatus struct {
	// Current condition of the carp.
	// More info: http://kubernetes.io/docs/user-guide/carp-states#carp-phase
	// +optional
	Phase CarpPhase `json:"phase,omitempty" protobuf:"bytes,1,opt,name=phase,casttype=CarpPhase"`
	// Current service state of carp.
	// More info: http://kubernetes.io/docs/user-guide/carp-states#carp-conditions
	// +patchStrategy=merge
	// +patchMergeKey=type
	// +listType=map
	// +listMapKey=type
	// +optional
	Conditions []CarpCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,2,opt,name=conditions"`
	// A human readable message indicating details about why the carp is in this condition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,3,opt,name=message"`
	// A brief CamelCase message indicating details about why the carp is in this state.
	// e.g. 'DiskPressure'
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,4,opt,name=reason"`

	// IP address of the host to which the carp is assigned. Empty if not yet scheduled.
	// +optional
	HostIP string `json:"hostIP,omitempty" protobuf:"bytes,5,opt,name=hostIP"`
	// IP address allocated to the carp. Routable at least within the cluster.
	// Empty if not yet allocated.
	// +optional
	CarpIP string `json:"carpIP,omitempty" protobuf:"bytes,6,opt,name=carpIP"`

	// RFC 3339 date and time at which the object was acknowledged by the Kubelet.
	// This is before the Kubelet pulled the container image(s) for the carp.
	// +optional
	StartTime *metav1.Time `json:"startTime,omitempty" protobuf:"bytes,7,opt,name=startTime"`

	// Carp infos are provided by different clients, hence the map type.
	//
	// +listType=map
	// +listMapKey=a
	// +listMapKey=b
	// +listMapKey=c
	Infos []CarpInfo `json:"infos,omitempty" protobuf:"bytes,8,rep,name=infos"`
}

type CarpCondition struct {
	// Type is the type of the condition.
	// Currently only Ready.
	// More info: http://kubernetes.io/docs/user-guide/carp-states#carp-conditions
	Type CarpConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=CarpConditionType"`
	// Status is the status of the condition.
	// Can be True, False, Unknown.
	// More info: http://kubernetes.io/docs/user-guide/carp-states#carp-conditions
	Status ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status,casttype=ConditionStatus"`
	// Last time we probed the condition.
	// +optional
	LastProbeTime metav1.Time `json:"lastProbeTime,omitempty" protobuf:"bytes,3,opt,name=lastProbeTime"`
	// Last time the condition transitioned from one status to another.
	// +optional
	LastTransitionTime metav1.Time `json:"lastTransitionTime,omitempty" protobuf:"bytes,4,opt,name=lastTransitionTime"`
	// Unique, one-word, CamelCase reason for the condition's last transition.
	// +optional
	Reason string `json:"reason,omitempty" protobuf:"bytes,5,opt,name=reason"`
	// Human-readable message indicating details about last transition.
	// +optional
	Message string `json:"message,omitempty" protobuf:"bytes,6,opt,name=message"`
}

type CarpInfo struct {
	// A is the first map key.
	// +required
	A int64 `json:"a" protobuf:"bytes,1,name=a"`
	// B is the second map key.
	// +required
	B string `json:"b" protobuf:"bytes,2,name=b"`
	// C is the third, optional map key
	// +optional
	C *string `json:"c,omitempty" protobuf:"bytes,4,opt,name=c"`

	// Some data for each pair of A and B.
	Data string `json:"data" protobuf:"bytes,3,name=data"`
}

// CarpSpec is a description of a carp
type CarpSpec struct {
	// Restart policy for all containers within the carp.
	// One of Always, OnFailure, Never.
	// Default to Always.
	// More info: http://kubernetes.io/docs/user-guide/carp-states#restartpolicy
	// +optional
	RestartPolicy RestartPolicy `json:"restartPolicy,omitempty" protobuf:"bytes,3,opt,name=restartPolicy,casttype=RestartPolicy"`
	// Optional duration in seconds the carp needs to terminate gracefully. May be decreased in delete request.
	// Value must be non-negative integer. The value zero indicates delete immediately.
	// If this value is nil, the default grace period will be used instead.
	// The grace period is the duration in seconds after the processes running in the carp are sent
	// a termination signal and the time when the processes are forcibly halted with a kill signal.
	// Set this value longer than the expected cleanup time for your process.
	// Defaults to 30 seconds.
	// +optional
	TerminationGracePeriodSeconds *int64 `json:"terminationGracePeriodSeconds,omitempty" protobuf:"varint,4,opt,name=terminationGracePeriodSeconds"`
	// Optional duration in seconds the carp may be active on the node relative to
	// StartTime before the system will actively try to mark it failed and kill associated containers.
	// Value must be a positive integer.
	// +optional
	ActiveDeadlineSeconds *int64 `json:"activeDeadlineSeconds,omitempty" protobuf:"varint,5,opt,name=activeDeadlineSeconds"`
	// NodeSelector is a selector which must be true for the carp to fit on a node.
	// Selector which must match a node's labels for the carp to be scheduled on that node.
	// More info: http://kubernetes.io/docs/user-guide/node-selection/README
	// +optional
	NodeSelector map[string]string `json:"nodeSelector,omitempty" protobuf:"bytes,7,rep,name=nodeSelector"`

	// ServiceAccountName is the name of the ServiceAccount to use to run this carp.
	// More info: https://kubernetes.io/docs/concepts/security/service-accounts/
	// +optional
	ServiceAccountName string `json:"serviceAccountName,omitempty" protobuf:"bytes,8,opt,name=serviceAccountName"`
	// DeprecatedServiceAccount is a deprecated alias for ServiceAccountName.
	// Deprecated: Use serviceAccountName instead.
	// +k8s:conversion-gen=false
	// +optional
	DeprecatedServiceAccount string `json:"deprecatedServiceAccount,omitempty" protobuf:"bytes,9,opt,name=deprecatedServiceAccount"`

	// NodeName is a request to schedule this carp onto a specific node. If it is non-empty,
	// the scheduler simply schedules this carp onto that node, assuming that it fits resource
	// requirements.
	// +optional
	NodeName string `json:"nodeName,omitempty" protobuf:"bytes,10,opt,name=nodeName"`
	// Host networking requested for this carp. Use the host's network namespace.
	// Default to false.
	// +k8s:conversion-gen=false
	// +optional
	HostNetwork bool `json:"hostNetwork,omitempty" protobuf:"varint,11,opt,name=hostNetwork"`
	// Use the host's pid namespace.
	// Optional: Default to false.
	// +k8s:conversion-gen=false
	// +optional
	HostPID bool `json:"hostPID,omitempty" protobuf:"varint,12,opt,name=hostPID"`
	// Use the host's ipc namespace.
	// Optional: Default to false.
	// +k8s:conversion-gen=false
	// +optional
	HostIPC bool `json:"hostIPC,omitempty" protobuf:"varint,13,opt,name=hostIPC"`
	// Specifies the hostname of the Carp
	// If not specified, the carp's hostname will be set to a system-defined value.
	// +optional
	Hostname string `json:"hostname,omitempty" protobuf:"bytes,16,opt,name=hostname"`
	// If specified, the fully qualified Carp hostname will be "<hostname>.<subdomain>.<carp namespace>.svc.<cluster domain>".
	// If not specified, the carp will not have a domainname at all.
	// +optional
	Subdomain string `json:"subdomain,omitempty" protobuf:"bytes,17,opt,name=subdomain"`
	// If specified, the carp will be dispatched by specified scheduler.
	// If not specified, the carp will be dispatched by default scheduler.
	// +optional
	SchedulerName string `json:"schedulerName,omitempty" protobuf:"bytes,19,opt,name=schedulerName"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +k8s:prerelease-lifecycle-gen:introduced=1.1

// CarpList is a list of Carps.
type CarpList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#types-kinds
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// List of carps.
	// More info: http://kubernetes.io/docs/user-guide/carps
	Items []Carp `json:"items" protobuf:"bytes,2,rep,name=items"`
}

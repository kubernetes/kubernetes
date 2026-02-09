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

package testapigroup

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

// Carp is a collection of containers, used as either input (create, update) or as output (list, get).
type Carp struct {
	metav1.TypeMeta
	// +optional
	metav1.ObjectMeta

	// Spec defines the behavior of a carp.
	// +optional
	Spec CarpSpec

	// Status represents the current information about a carp. This data may not be up
	// to date.
	// +optional
	Status CarpStatus
}

// CarpStatus represents information about the status of a carp. Status may trail the actual
// state of a system.
type CarpStatus struct {
	// +optional
	Phase CarpPhase
	// +optional
	Conditions []CarpCondition
	// A human readable message indicating details about why the carp is in this state.
	// +optional
	Message string
	// A brief CamelCase message indicating details about why the carp is in this state. e.g. 'DiskPressure'
	// +optional
	Reason string

	// +optional
	HostIP string
	// +optional
	CarpIP string

	// Date and time at which the object was acknowledged by the Kubelet.
	// This is before the Kubelet pulled the container image(s) for the carp.
	// +optional
	StartTime *metav1.Time

	// Carp infos are provided by different clients, hence the map type.
	//
	// +listType=map
	// +listKey=a
	// +listKey=b
	// +listKey=c
	Infos []CarpInfo
}

type CarpCondition struct {
	Type   CarpConditionType
	Status ConditionStatus
	// +optional
	LastProbeTime metav1.Time
	// +optional
	LastTransitionTime metav1.Time
	// +optional
	Reason string
	// +optional
	Message string
}

type CarpInfo struct {
	// A is the first map key.
	// +required
	A int64
	// B is the second map key.
	// +required
	B string
	// C is the third, optional map key
	// +optional
	C *string

	// Some data for each pair of A and B.
	Data string
}

// CarpSpec is a description of a carp
type CarpSpec struct {
	// +optional
	RestartPolicy RestartPolicy
	// Optional duration in seconds the carp needs to terminate gracefully. May be decreased in delete request.
	// Value must be non-negative integer. The value zero indicates delete immediately.
	// If this value is nil, the default grace period will be used instead.
	// The grace period is the duration in seconds after the processes running in the carp are sent
	// a termination signal and the time when the processes are forcibly halted with a kill signal.
	// Set this value longer than the expected cleanup time for your process.
	// +optional
	TerminationGracePeriodSeconds *int64
	// Optional duration in seconds relative to the StartTime that the carp may be active on a node
	// before the system actively tries to terminate the carp; value must be positive integer
	// +optional
	ActiveDeadlineSeconds *int64
	// NodeSelector is a selector which must be true for the carp to fit on a node
	// +optional
	NodeSelector map[string]string

	// ServiceAccountName is the name of the ServiceAccount to use to run this carp
	// The carp will be allowed to use secrets referenced by the ServiceAccount
	ServiceAccountName string

	// NodeName is a request to schedule this carp onto a specific node.  If it is non-empty,
	// the scheduler simply schedules this carp onto that node, assuming that it fits resource
	// requirements.
	// +optional
	NodeName string
	// Specifies the hostname of the Carp.
	// If not specified, the carp's hostname will be set to a system-defined value.
	// +optional
	Hostname string
	// If specified, the fully qualified Carp hostname will be "<hostname>.<subdomain>.<carp namespace>.svc.<cluster domain>".
	// If not specified, the carp will not have a domainname at all.
	// +optional
	Subdomain string
	// If specified, the carp will be dispatched by specified scheduler.
	// If not specified, the carp will be dispatched by default scheduler.
	// +optional
	SchedulerName string
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// CarpList is a list of Carps.
type CarpList struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []Carp
}

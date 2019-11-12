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

package v1beta1

import (
	corev1 "sigs.k8s.io/kustomize/pseudo/k8s/api/core/v1"
	metav1 "sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime.Object

// Event is a report of an event somewhere in the cluster. It generally denotes some state change in the system.
type Event struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Required. Time when this Event was first observed.
	EventTime metav1.MicroTime `json:"eventTime" protobuf:"bytes,2,opt,name=eventTime"`

	// Data about the Event series this event represents or nil if it's a singleton Event.
	// +optional
	Series *EventSeries `json:"series,omitempty" protobuf:"bytes,3,opt,name=series"`

	// Name of the controller that emitted this Event, e.g. `kubernetes.io/kubelet`.
	// +optional
	ReportingController string `json:"reportingController,omitempty" protobuf:"bytes,4,opt,name=reportingController"`

	// ID of the controller instance, e.g. `kubelet-xyzf`.
	// +optional
	ReportingInstance string `json:"reportingInstance,omitempty" protobuf:"bytes,5,opt,name=reportingInstance"`

	// What action was taken/failed regarding to the regarding object.
	// +optional
	Action string `json:"action,omitempty" protobuf:"bytes,6,name=action"`

	// Why the action was taken.
	Reason string `json:"reason,omitempty" protobuf:"bytes,7,name=reason"`

	// The object this Event is about. In most cases it's an Object reporting controller implements.
	// E.g. ReplicaSetController implements ReplicaSets and this event is emitted because
	// it acts on some changes in a ReplicaSet object.
	// +optional
	Regarding corev1.ObjectReference `json:"regarding,omitempty" protobuf:"bytes,8,opt,name=regarding"`

	// Optional secondary object for more complex actions. E.g. when regarding object triggers
	// a creation or deletion of related object.
	// +optional
	Related *corev1.ObjectReference `json:"related,omitempty" protobuf:"bytes,9,opt,name=related"`

	// Optional. A human-readable description of the status of this operation.
	// Maximal length of the note is 1kB, but libraries should be prepared to
	// handle values up to 64kB.
	// +optional
	Note string `json:"note,omitempty" protobuf:"bytes,10,opt,name=note"`

	// Type of this event (Normal, Warning), new types could be added in the
	// future.
	// +optional
	Type string `json:"type,omitempty" protobuf:"bytes,11,opt,name=type"`

	// Deprecated field assuring backward compatibility with core.v1 Event type
	// +optional
	DeprecatedSource corev1.EventSource `json:"deprecatedSource,omitempty" protobuf:"bytes,12,opt,name=deprecatedSource"`
	// Deprecated field assuring backward compatibility with core.v1 Event type
	// +optional
	DeprecatedFirstTimestamp metav1.Time `json:"deprecatedFirstTimestamp,omitempty" protobuf:"bytes,13,opt,name=deprecatedFirstTimestamp"`
	// Deprecated field assuring backward compatibility with core.v1 Event type
	// +optional
	DeprecatedLastTimestamp metav1.Time `json:"deprecatedLastTimestamp,omitempty" protobuf:"bytes,14,opt,name=deprecatedLastTimestamp"`
	// Deprecated field assuring backward compatibility with core.v1 Event type
	// +optional
	DeprecatedCount int32 `json:"deprecatedCount,omitempty" protobuf:"varint,15,opt,name=deprecatedCount"`
}

// EventSeries contain information on series of events, i.e. thing that was/is happening
// continuously for some time.
type EventSeries struct {
	// Number of occurrences in this series up to the last heartbeat time
	Count int32 `json:"count" protobuf:"varint,1,opt,name=count"`
	// Time when last Event from the series was seen before last heartbeat.
	LastObservedTime metav1.MicroTime `json:"lastObservedTime" protobuf:"bytes,2,opt,name=lastObservedTime"`
	// Information whether this series is ongoing or finished.
	// Deprecated. Planned removal for 1.18
	State EventSeriesState `json:"state" protobuf:"bytes,3,opt,name=state"`
}

type EventSeriesState string

const (
	EventSeriesStateOngoing  EventSeriesState = "Ongoing"
	EventSeriesStateFinished EventSeriesState = "Finished"
	EventSeriesStateUnknown  EventSeriesState = "Unknown"
)

// +k8s:deepcopy-gen:interfaces=sigs.k8s.io/kustomize/pseudo/k8s/apimachinery/pkg/runtime.Object

// EventList is a list of Event objects.
type EventList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of schema objects.
	Items []Event `json:"items" protobuf:"bytes,2,rep,name=items"`
}

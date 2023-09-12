/*
Copyright 2020 The Kubernetes Authors.

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
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Event is a report of an event somewhere in the cluster. It generally denotes some state change in the system.
// Events have a limited retention time and triggers and messages may evolve
// with time.  Event consumers should not rely on the timing of an event
// with a given Reason reflecting a consistent underlying trigger, or the
// continued existence of events with that Reason.  Events should be
// treated as informative, best-effort, supplemental data.
type Event struct {
	metav1.TypeMeta `json:",inline"`

	// Standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ObjectMeta `json:"metadata" protobuf:"bytes,1,opt,name=metadata"`

	// eventTime is the time when this Event was first observed. It is required.
	EventTime metav1.MicroTime `json:"eventTime" protobuf:"bytes,2,opt,name=eventTime"`

	// series is data about the Event series this event represents or nil if it's a singleton Event.
	// +optional
	Series *EventSeries `json:"series,omitempty" protobuf:"bytes,3,opt,name=series"`

	// reportingController is the name of the controller that emitted this Event, e.g. `kubernetes.io/kubelet`.
	// This field cannot be empty for new Events.
	ReportingController string `json:"reportingController,omitempty" protobuf:"bytes,4,opt,name=reportingController"`

	// reportingInstance is the ID of the controller instance, e.g. `kubelet-xyzf`.
	// This field cannot be empty for new Events and it can have at most 128 characters.
	ReportingInstance string `json:"reportingInstance,omitempty" protobuf:"bytes,5,opt,name=reportingInstance"`

	// action is what action was taken/failed regarding to the regarding object. It is machine-readable.
	// This field cannot be empty for new Events and it can have at most 128 characters.
	Action string `json:"action,omitempty" protobuf:"bytes,6,name=action"`

	// reason is why the action was taken. It is human-readable.
	// This field cannot be empty for new Events and it can have at most 128 characters.
	Reason string `json:"reason,omitempty" protobuf:"bytes,7,name=reason"`

	// regarding contains the object this Event is about. In most cases it's an Object reporting controller
	// implements, e.g. ReplicaSetController implements ReplicaSets and this event is emitted because
	// it acts on some changes in a ReplicaSet object.
	// +optional
	Regarding corev1.ObjectReference `json:"regarding,omitempty" protobuf:"bytes,8,opt,name=regarding"`

	// related is the optional secondary object for more complex actions. E.g. when regarding object triggers
	// a creation or deletion of related object.
	// +optional
	Related *corev1.ObjectReference `json:"related,omitempty" protobuf:"bytes,9,opt,name=related"`

	// note is a human-readable description of the status of this operation.
	// Maximal length of the note is 1kB, but libraries should be prepared to
	// handle values up to 64kB.
	// +optional
	Note string `json:"note,omitempty" protobuf:"bytes,10,opt,name=note"`

	// type is the type of this event (Normal, Warning), new types could be added in the future.
	// It is machine-readable.
	// This field cannot be empty for new Events.
	Type string `json:"type,omitempty" protobuf:"bytes,11,opt,name=type"`

	// deprecatedSource is the deprecated field assuring backward compatibility with core.v1 Event type.
	// +optional
	DeprecatedSource corev1.EventSource `json:"deprecatedSource,omitempty" protobuf:"bytes,12,opt,name=deprecatedSource"`
	// deprecatedFirstTimestamp is the deprecated field assuring backward compatibility with core.v1 Event type.
	// +optional
	DeprecatedFirstTimestamp metav1.Time `json:"deprecatedFirstTimestamp,omitempty" protobuf:"bytes,13,opt,name=deprecatedFirstTimestamp"`
	// deprecatedLastTimestamp is the deprecated field assuring backward compatibility with core.v1 Event type.
	// +optional
	DeprecatedLastTimestamp metav1.Time `json:"deprecatedLastTimestamp,omitempty" protobuf:"bytes,14,opt,name=deprecatedLastTimestamp"`
	// deprecatedCount is the deprecated field assuring backward compatibility with core.v1 Event type.
	// +optional
	DeprecatedCount int32 `json:"deprecatedCount,omitempty" protobuf:"varint,15,opt,name=deprecatedCount"`
}

// EventSeries contain information on series of events, i.e. thing that was/is happening
// continuously for some time. How often to update the EventSeries is up to the event reporters.
// The default event reporter in "k8s.io/client-go/tools/events/event_broadcaster.go" shows
// how this struct is updated on heartbeats and can guide customized reporter implementations.
type EventSeries struct {
	// count is the number of occurrences in this series up to the last heartbeat time.
	Count int32 `json:"count" protobuf:"varint,1,opt,name=count"`
	// lastObservedTime is the time when last Event from the series was seen before last heartbeat.
	LastObservedTime metav1.MicroTime `json:"lastObservedTime" protobuf:"bytes,2,opt,name=lastObservedTime"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// EventList is a list of Event objects.
type EventList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of schema objects.
	Items []Event `json:"items" protobuf:"bytes,2,rep,name=items"`
}

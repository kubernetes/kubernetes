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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +nonNamespaced=true

// BulkRequest contains a single request to bulk api.
type BulkRequest struct {
	metav1.TypeMeta `json:",inline"`

	// Opaque identifier used to match requests and responses.
	// +optional
	RequestID string `json:"requestId,omitempty" protobuf:"bytes,1,opt,name=requestId"`

	// Starts new watch.
	// +optional
	Watch *WatchOperation `json:"watch,omitempty" protobuf:"bytes,1,opt,name=watch"`

	// Stops existing watch.
	// +optional
	StopWatch *StopWatchOperation `json:"stopWatch,omitempty" protobuf:"bytes,2,opt,name=stopWatch"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +nonNamespaced=true

// BulkResponse contains a single response from bulk api.
type BulkResponse struct {
	metav1.TypeMeta `json:",inline"`

	// Opaque identifier propagated from the request.
	// Null for messages issued by server (WatchEvent, WatchStopped triggered by server)
	// +optional
	RequestID *string `json:"requestId,omitempty" protobuf:"bytes,1,opt,name=requestId"`

	// Contains error details when request was failed.
	// +optional
	Failure *metav1.Status `json:"status,omitempty" protobuf:"bytes,2,opt,name=status"`

	// Sent when watch was manually stopped.
	// +optional
	WatchStopped *WatchStopped `json:"watchStopped,omitempty" protobuf:"bytes,3,opt,name=watchStopped"`

	// Sent when new watch was started.
	// +optional
	WatchStarted *WatchStarted `json:"watchStarted,omitempty" protobuf:"bytes,4,opt,name=watchStarted"`

	// Contains single watch event.
	// +optional
	WatchEvent *BulkWatchEvent `json:"watchEvent,omitempty" protobuf:"bytes,5,opt,name=watchEvent"`
}

// Selector
type ResourceSelector struct {

	// Resource identifier.  This is not the kind.  For example: pods
	Resource string `json:"resource" protobuf:"bytes,3,opt,name=resource"`

	// Name of APIGroup, defaults to core API group.
	// +optional
	Group string `json:"group,omitempty" protobuf:"bytes,1,opt,name=group"`

	// Version of APIGroup, defaults to preferred version.
	// +optional
	Version string `json:"version,omitempty" protobuf:"bytes,2,opt,name=version"`

	// Namespace
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,4,opt,name=namespace"`

	// Name of the single resource object
	// +optional
	Name string `json:"name,omitempty" protobuf:"bytes,5,opt,name=namespace"`

	// Query options used to filter watched resources.
	// Field 'Watch' is ignored by bulk api.
	// +optional
	Options *metav1.ListOptions `json:"options" protobuf:"bytes,3,name=options"`
}

// Event represents a single event to a watched resource.
type BulkWatchEvent struct {

	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`

	// Watch event.
	Event metav1.WatchEvent `json:",inline" protobuf:"bytes,2,name=event"`
}

// WatchStarted is a return value when new watch was started.
type WatchStarted struct {

	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`
}

// WatchStopped is a return value when existing watch was stopped.
type WatchStopped struct {

	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`
}

// WatchOperation is a request to start new watch.
type WatchOperation struct {

	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`

	// Selector.
	Selector ResourceSelector `json:"selector" protobuf:"bytes,2,name=selector"`
}

// StopWatchOperation is a request to stop existing watch.
type StopWatchOperation struct {

	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`
}

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

package bulk

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +genclient:nonNamespaced

// BulkRequest contains a single request to bulk api.
type BulkRequest struct {
	metav1.TypeMeta

	// Opaque identifier used to match requests and responses.
	// +optional
	RequestID string

	// Starts new watch.
	// +optional
	Watch *WatchOperation

	// Starts new watch.
	// +optional
	WatchList *WatchListOperation

	// Stops existing watch.
	// +optional
	StopWatch *StopWatchOperation

	// Gets list of items.
	// +optional
	List *ListOperation

	// Gets single item.
	// +optional
	Get *GetOperation
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +genclient:nonNamespaced

// BulkResponse contains a single response from bulk api.
type BulkResponse struct {
	metav1.TypeMeta

	// Opaque identifier propagated from the request.
	// Is null for messages issued by server (events, watch aborts).
	// +optional
	RequestID *string

	// Contains error details when request (watch, watchStop etc) was failed.
	// +optional
	Failure *metav1.Status

	// Sent when watch was stopped (by server or client).
	// +optional
	WatchStopped *WatchStopped

	// Sent when new watch was started.
	// +optional
	WatchStarted *WatchStarted

	// Contains single watch event.
	// +optional
	WatchEvent *BulkWatchEvent

	// Contains single object.
	// +optional
	GetResult *GetResult

	// Contains single object.
	// +optional
	ListResult *ListResult
}

// GroupVersionResource identifies selected resource.
type GroupVersionResource struct {
	// Resource identifier.
	Resource string

	// Name of APIGroup, defaults to core API group.
	Group string

	// Version of APIGroup, defaults to preferred version.
	Version string
}

// ItemSelector identifies single resource object.
type ItemSelector struct {
	// Resource identifies requested resource.
	GroupVersionResource

	// Name is the object name.
	Name string

	// Namespace.
	Namespace string

	// Query options used to filter watched resources.
	// +optional
	Options *metav1.GetOptions
}

// ListSelector identifies list of resource objects.
type ListSelector struct {
	// Resource identifies requested resource.
	GroupVersionResource

	// Namespace.
	Namespace string

	// Query options used to filter watched resources.
	// Field 'Watch' is ignored by bulk api.
	// +optional
	Options *metav1.ListOptions
}

// BulkWatchEvent represents a single event to a watched resource.
type BulkWatchEvent struct {
	// Identifier of the watch.
	WatchID string

	// Watch event
	WatchEvent watch.Event
}

// WatchStarted is a return value when new watch was started.
type WatchStarted struct {
	// Identifier of the watch.
	WatchID string
}

// WatchStopped is a return value when existing watch was stopped.
type WatchStopped struct {
	// Identifier of the watch.
	WatchID string
}

// ListResult is a return value when existing watch was stopped.
type ListResult struct {
	// List of resources.
	List runtime.Object
}

// GetResult is a return value when existing watch was stopped.
type GetResult struct {
	// Single object.
	Item runtime.Object
}

// WatchOperation is a request to start new watch.
type WatchOperation struct {
	// Identifier of the watch.
	WatchID string

	// Selector.
	ItemSelector
}

// WatchListOperation is a request to start new watch.
type WatchListOperation struct {
	// Identifier of the watch.
	WatchID string

	// Selector.
	ListSelector
}

// StopWatchOperation is a request to stop existing watch.
type StopWatchOperation struct {
	// Identifier of the watch.
	WatchID string
}

// ListOperation is a request to perform bulk get/list.
type ListOperation struct {
	// Selector.
	ListSelector
}

// GetOperation is a request to perform bulk get/list.
type GetOperation struct {
	// Selector.
	ItemSelector
}

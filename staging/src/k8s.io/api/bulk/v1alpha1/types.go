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
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +genclient:nonNamespaced

// BulkRequest contains a single request to bulk api.
type BulkRequest struct {
	metav1.TypeMeta `json:",inline"`

	// Opaque identifier used to match requests and responses.
	RequestID string `json:"requestId" protobuf:"bytes,1,name=requestId"`

	// Starts new watch.
	// +optional
	Watch *WatchOperation `json:"watch,omitempty" protobuf:"bytes,2,opt,name=watch"`

	// Starts new watch.
	// +optional
	WatchList *WatchListOperation `json:"watchList,omitempty" protobuf:"bytes,3,opt,name=watchList"`

	// Stops existing watch.
	// +optional
	StopWatch *StopWatchOperation `json:"stopWatch,omitempty" protobuf:"bytes,4,opt,name=stopWatch"`

	// Gets list of items.
	// +optional
	List *ListOperation `json:"list,omitempty" protobuf:"bytes,5,opt,name=list"`

	// Gets single item.
	// +optional
	Get *GetOperation `json:"get,omitempty" protobuf:"bytes,6,opt,name=get"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object
// +genclient:nonNamespaced

// BulkResponse contains a single response from bulk api.
type BulkResponse struct {
	metav1.TypeMeta `json:",inline"`

	// Opaque identifier propagated from the request.
	// Null for messages issued by server (WatchEvent, WatchStopped triggered by server)
	// +optional
	RequestID *string `json:"requestId,omitempty" protobuf:"bytes,1,opt,name=requestId"`

	// Contains error details when request was failed.
	// +optional
	Failure *metav1.Status `json:"failure,omitempty" protobuf:"bytes,2,opt,name=failure"`

	// Sent when watch was manually stopped.
	// +optional
	WatchStopped *WatchStopped `json:"watchStopped,omitempty" protobuf:"bytes,3,opt,name=watchStopped"`

	// Sent when new watch was started.
	// +optional
	WatchStarted *WatchStarted `json:"watchStarted,omitempty" protobuf:"bytes,4,opt,name=watchStarted"`

	// Contains single watch event.
	// +optional
	WatchEvent *BulkWatchEvent `json:"watchEvent,omitempty" protobuf:"bytes,5,opt,name=watchEvent"`

	// Contains single object.
	// +optional
	GetResult *GetResult `json:"getResult,omitempty" protobuf:"bytes,6,opt,name=getResult"`

	// Contains single object.
	// +optional
	ListResult *ListResult `json:"listResult,omitempty" protobuf:"bytes,7,opt,name=listResult"`
}

// GroupVersionResource identifies resource.
type GroupVersionResource struct {
	// Name of APIGroup, defaults to core API group.
	// +optional
	Group string `json:"group,omitempty" protobuf:"bytes,1,opt,name=group"`

	// Version of APIGroup, defaults to preferred version.
	// +optional
	Version string `json:"version,omitempty" protobuf:"bytes,2,opt,name=version"`

	// Resource identifier (pods, events etc).  Note: this is not the kind.
	Resource string `json:"resource" protobuf:"bytes,3,opt,name=resource"`
}

// ItemSelector is the query options to get/watch calls.
type ItemSelector struct {
	// Resource identifies requested resource.
	GroupVersionResource `json:",inline" protobuf:"bytes,1,name=resource"`

	// Name is the object name.
	Name string `json:"name,omitempty" protobuf:"bytes,5,opt,name=name"`

	// Namespace
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,4,opt,name=namespace"`

	// Query options used to filter watched resources.
	// Field 'Watch' is ignored by bulk api.
	// +optional
	Options *metav1.GetOptions `json:"options" protobuf:"bytes,6,name=options"`
}

// ListSelector is the query options to list/watchList calls.
type ListSelector struct {
	// Resource identifies requested resource.
	GroupVersionResource `json:",inline" protobuf:"bytes,1,name=resource"`

	// Namespace
	// +optional
	Namespace string `json:"namespace,omitempty" protobuf:"bytes,4,opt,name=namespace"`

	// Query options used to filter watched resources.
	// Field 'Watch' is ignored by bulk api.
	// +optional
	Options *metav1.ListOptions `json:"options" protobuf:"bytes,2,name=options"`
}

// BulkWatchEvent represents a single event to a watched resource.
type BulkWatchEvent struct {
	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`

	// Watch event.
	metav1.WatchEvent `json:",inline" protobuf:"bytes,2,name=event"`
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

// ListResult is a return value when existing watch was stopped.
type ListResult struct {
	// List of resources.
	List runtime.RawExtension `json:"list" protobuf:"bytes,2,rep,name=item"`
}

// GetResult is a return value when existing watch was stopped.
type GetResult struct {
	// Single object.
	Item runtime.RawExtension `json:"item" protobuf:"bytes,2,rep,name=item"`
}

// WatchOperation is a request to start new watch.
type WatchOperation struct {
	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`

	// Selector.
	ItemSelector `json:",inline" protobuf:"bytes,2,name=selector"`
}

// WatchListOperation is a request to start new watch.
type WatchListOperation struct {
	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`

	// Selector.
	ListSelector `json:",inline" protobuf:"bytes,2,name=selector"`
}

// StopWatchOperation is a request to stop existing watch.
type StopWatchOperation struct {
	// Identifier of the watch.
	WatchID string `json:"watchId" protobuf:"bytes,1,name=watchId"`
}

// ListOperation is a request to perform bulk get/list.
type ListOperation struct {
	// Selector.
	ListSelector `json:",inline" protobuf:"bytes,1,name=selector"`
}

// GetOperation is a request to perform bulk get/list.
type GetOperation struct {
	// Selector.
	ItemSelector `json:",inline" protobuf:"bytes,1,name=selector"`
}

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

package internalversion

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ListOptions is the query options to a standard REST list call.
type ListOptions struct {
	metav1.TypeMeta

	// A selector based on labels
	LabelSelector labels.Selector
	// A selector based on fields
	FieldSelector fields.Selector
	// If true, watch for changes to this list
	Watch bool
	// allowWatchBookmarks requests watch events with type "BOOKMARK".
	// Servers that do not implement bookmarks may ignore this flag and
	// bookmarks are sent at the server's discretion. Clients should not
	// assume bookmarks are returned at any specific interval, nor may they
	// assume the server will send any BOOKMARK event during a session.
	// If this is not a watch, this field is ignored.
	AllowWatchBookmarks bool
	// resourceVersion sets a constraint on what resource versions a request may be served from.
	// See https://kubernetes.io/docs/reference/using-api/api-concepts/#resource-versions for
	// details.
	ResourceVersion string
	// resourceVersionMatch determines how resourceVersion is applied to list calls.
	// It is highly recommended that resourceVersionMatch be set for list calls where
	// resourceVersion is set.
	// See https://kubernetes.io/docs/reference/using-api/api-concepts/#resource-versions for
	// details.
	ResourceVersionMatch metav1.ResourceVersionMatch

	// Timeout for the list/watch call.
	TimeoutSeconds *int64
	// Limit specifies the maximum number of results to return from the server. The server may
	// not support this field on all resource types, but if it does and more results remain it
	// will set the continue field on the returned list object.
	Limit int64
	// Continue is a token returned by the server that lets a client retrieve chunks of results
	// from the server by specifying limit. The server may reject requests for continuation tokens
	// it does not recognize and will return a 410 error if the token can no longer be used because
	// it has expired.
	Continue string

	// `sendInitialEvents=true` may be set together with `watch=true`.
	// In that case, the watch stream will begin with synthetic events to
	// produce the current state of objects in the collection. Once all such
	// events have been sent, a synthetic "Bookmark" event  will be sent.
	// The bookmark will report the ResourceVersion (RV) corresponding to the
	// set of objects, and be marked with `"k8s.io/initial-events-end": "true"` annotation.
	// Afterwards, the watch stream will proceed as usual, sending watch events
	// corresponding to changes (subsequent to the RV) to objects watched.
	//
	// When `sendInitialEvents` option is set, we require `resourceVersionMatch`
	// option to also be set. The semantic of the watch request is as following:
	// - `resourceVersionMatch` = NotOlderThan
	//   is interpreted as "data at least as new as the provided `resourceVersion`"
	//   and the bookmark event is send when the state is synced
	//   to a `resourceVersion` at least as fresh as the one provided by the ListOptions.
	//   If `resourceVersion` is unset, this is interpreted as "consistent read" and the
	//   bookmark event is send when the state is synced at least to the moment
	//   when request started being processed.
	// - `resourceVersionMatch` set to any other value or unset
	//   Invalid error is returned.
	//
	// Defaults to true if `resourceVersion=""` or `resourceVersion="0"` (for backward
	// compatibility reasons) and to false otherwise.
	SendInitialEvents *bool
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// List holds a list of objects, which may not be known by the server.
type List struct {
	metav1.TypeMeta
	// +optional
	metav1.ListMeta

	Items []runtime.Object
}

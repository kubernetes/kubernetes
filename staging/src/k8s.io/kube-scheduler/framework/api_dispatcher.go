/*
Copyright 2025 The Kubernetes Authors.

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

package framework

import (
	"context"
	"errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
)

var (
	// ErrCallSkipped is returned by APIDispatcher.Add or sent to the OnFinish channel when the call is skipped and will not be executed.
	ErrCallSkipped = errors.New("call skipped")
	// ErrCallOverwritten is sent to the OnFinish channel when an enqueued call is overwritten by a newer or more relevant call.
	ErrCallOverwritten = errors.New("call overwritten")
)

// APICallType defines a call type name. Each APICall implementation must have a unique type within a given dispatcher.
type APICallType string

// APICallRelevances maps all possible APICallTypes to a relevance value.
// A more relevant API call should overwrite a less relevant one for the same object.
// Types of the same relevance should only be defined for different object types.
type APICallRelevances map[APICallType]int

// APICall defines the interface for an API call that can be processed by an APIDispatcher.
type APICall interface {
	// CallType returns the type of the API call.
	CallType() APICallType
	// UID returns the UID of the object this call refers to.
	// This is used to identify and potentially merge or skip calls for the same object.
	UID() types.UID
	// Execute performs the actual API call.
	Execute(ctx context.Context, client clientset.Interface) error
	// Merge merges the state of an older call for the same object into the current (receiver) call.
	// The receiver should incorporate all necessary information from oldCall, as oldCall will be discarded.
	// It returns true if the call still needs to be executed after the merge,
	// or false if the merge resulted in a no-op that can be skipped.
	Merge(oldCall APICall) (needsCall bool, err error)
	// Update applies the changes from this API call to the given object, potentially storing
	// information from the object needed for later execution. It returns true if the call still
	// needs to be executed after the update, along with the modified object.
	// The implementation should return a copy of the object if it is modified.
	Update(obj metav1.Object) (needsCall bool, updatedObj metav1.Object, err error)
}

// APICallOptions defines options for an API call.
type APICallOptions struct {
	// OnFinish is an optional channel to receive the final result of a call's lifecycle.
	//
	// The result is sent in a non-blocking way. If this channel is unbuffered and has no
	// ready receiver, the result will be dropped.
	//
	// Note that receiving an error does not guarantee the API call itself was executed.
	// For instance, an ErrCallOverwritten error is sent if the call was replaced by a
	// newer one before it could be executed.
	//
	// To opt out of receiving a result, leave this channel nil.
	OnFinish chan<- error
}

// APIDispatcher defines the interface for a dispatcher that queues and asynchronously executes API calls.
type APIDispatcher interface {
	// Add adds an API call to the dispatcher's queue. It returns an error if the call is not enqueued
	// (e.g., if it's skipped). The caller should handle ErrCallSkipped if returned.
	Add(incomingAPICall APICall, opts APICallOptions) error
	// UpdateObject applies pending changes from a queued API call to the given object,
	// if one exists in the dispatcher, and returns the potentially modified object.
	UpdateObject(obj metav1.Object) (metav1.Object, error)
}

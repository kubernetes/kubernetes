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

// IsUnexpectedError returns true if the given error is not nil and is not one of the expected
// dispatcher lifecycle errors (ErrCallSkipped, ErrCallOverwritten). This can be used to
// filter for errors that may require logging or special handling.
func IsUnexpectedError(err error) bool {
	return err != nil && !errors.Is(err, ErrCallSkipped) && !errors.Is(err, ErrCallOverwritten)
}

// APICallType defines a call type name that governs how the dispatcher handles multiple pending calls for the same object.
//
// The type determines if two calls for the same object are mergeable
// or if one should overwrite the other based on relevance:
//   - Calls with the same type should be merged.
//   - When calls have different types, their scores in APICallRelevances are used to determine precedence.
//
// Each APICall implementation should have a unique type within a given dispatcher.
type APICallType string

// APICallRelevances maps all possible APICallTypes to a relevance value.
// A more relevant API call should overwrite a less relevant one for the same object.
// Types of the same relevance should only be defined for different object types.
type APICallRelevances map[APICallType]int

// APICall defines the interface for an API call that can be processed by an APIDispatcher.
type APICall interface {
	// CallType returns the type of the API call.
	// See the APICallType and APICallRelevances comments on how to define the APICallType.
	CallType() APICallType
	// UID returns the UID of the object this call refers to.
	// This is used to identify and potentially merge or skip calls for the same object.
	UID() types.UID
	// Execute performs the actual API call.
	Execute(ctx context.Context, client clientset.Interface) error
	// Merge merges the state of an older call for the same object into the current (receiver) call.
	// The receiver should incorporate all necessary information from oldCall, as oldCall will be discarded.
	// After this method is called, IsNoOp() should be checked to see if the call can be skipped.
	Merge(oldCall APICall) error
	// Sync synchronizes the state of this call with the given object.
	// It may apply changes to the object or store information from the object needed for later execution.
	// The implementation should return a copy of the object if it is modified.
	// After this method is called, IsNoOp() should be checked to see if the call can be skipped.
	Sync(obj metav1.Object) (metav1.Object, error)
	// IsNoOp returns true if the call represents a no-operation and should be skipped by the dispatcher.
	// A call may be a no-op from its creation or become one after a Merge or Update.
	IsNoOp() bool
}

// APICallOptions defines options for an API call.
type APICallOptions struct {
	// OnFinish is an optional channel to receive the final result of a call's lifecycle.
	//
	// The result is sent in a non-blocking way. If this channel is unbuffered and has no
	// ready receiver, the result will be dropped.
	//
	// Note that receiving an error does not guarantee the API call itself was executed.
	// For instance, an ErrCallOverwritten or ErrCallSkipped error may be sent.
	//
	// To opt out of receiving a result, leave this channel nil.
	OnFinish chan<- error
}

// APIDispatcher defines the interface for a dispatcher that queues and asynchronously executes API calls.
type APIDispatcher interface {
	// Add adds an API call to the dispatcher's queue. It returns an error if the call is not enqueued
	// (e.g., if it's skipped). The caller should handle ErrCallSkipped if returned.
	Add(incomingAPICall APICall, opts APICallOptions) error
	// SyncObject performs a two-way synchronization between the given object
	// and a pending API call held within the dispatcher.
	// This can be called by the scheduler's event handlers on object updates
	// to enrich the cached state and the call.
	//
	// If a call for the object exists there, this method:
	// 1. Applies the call's pending changes to the object, providing an optimistic preview of its state.
	// 2. Allows the call to update its own internal state from the object,
	//    ensuring it has the most recent data before its eventual execution.
	//
	// It returns the modified object. If no call is pending for the object,
	// the original object is returned unmodified.
	SyncObject(obj metav1.Object) (metav1.Object, error)
}

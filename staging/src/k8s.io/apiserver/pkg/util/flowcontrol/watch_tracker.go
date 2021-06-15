/*
Copyright 2021 The Kubernetes Authors.

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

package flowcontrol

import (
	"sync"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/endpoints/request"
)

// readOnlyVerbs contains verbs for read-only requests.
var readOnlyVerbs = sets.NewString("get", "list", "watch", "proxy")

// watchIdentifier identifies group of watches that are similar.
// As described in the "Priority and Fairness" KEP, we consider
// watches similar if they have the same resourceType, namespace
// and name. We ignore selectors as they have to be evaluated
// when processing an even anyway.
//
// TODO: For now we only track the number of watches registered
// in our kube-apiserver. Eventually we should consider sharing
// this information with other kube-apiserver as described in the
// KEP, but this isn't part of the first version.
type watchIdentifier struct {
	apiGroup  string
	resource  string
	namespace string
	name      string
}

// ForgetWatchFunc is a function that should be called to forget
// the previously registered watch from the watch tracker.
type ForgetWatchFunc func()

// WatchTracker is an interface that allows tracking the number
// of watches in the system for the purpose of estimating the
// cost of incoming mutating requests.
type WatchTracker interface {
	// RegisterWatch reqisters a watch with the provided requestInfo
	// in the tracker. It returns the function that should be called
	// to forget the watcher once it is finished.
	RegisterWatch(requestInfo *request.RequestInfo) ForgetWatchFunc

	// GetInterestedWatchCount returns the number of watches that are
	// potentially interested in a request with a given RequestInfo
	// for the purpose of estimating cost of that request.
	GetInterestedWatchCount(requestInfo *request.RequestInfo) int
}

// watchTracker tracks the number of watches in the system for
// the purpose of estimating the cost of incoming mutating requests.
type watchTracker struct {
	lock sync.Mutex

	watchCount map[watchIdentifier]int
}

func NewWatchTracker() WatchTracker {
	return &watchTracker{
		watchCount: make(map[watchIdentifier]int),
	}
}

// RegisterWatch implements WatchTracker interface.
func (w *watchTracker) RegisterWatch(requestInfo *request.RequestInfo) ForgetWatchFunc {
	if requestInfo == nil || requestInfo.Verb != "watch" {
		return nil
	}

	identifier := &watchIdentifier{
		apiGroup:  requestInfo.APIGroup,
		resource:  requestInfo.Resource,
		namespace: requestInfo.Namespace,
		name:      requestInfo.Name,
	}

	w.lock.Lock()
	defer w.lock.Unlock()
	w.watchCount[*identifier]++
	return w.forgetWatch(identifier)
}

func (w *watchTracker) forgetWatch(identifier *watchIdentifier) ForgetWatchFunc {
	return func() {
		w.lock.Lock()
		defer w.lock.Unlock()

		w.watchCount[*identifier]--
		if w.watchCount[*identifier] == 0 {
			delete(w.watchCount, *identifier)
		}
	}
}

// GetInterestedWatchCount implements WatchTracker interface.
//
// TODO(wojtek-t): As of now, requestInfo for object creation (POST) doesn't
//  contain the Name field set. Figure out if we can somehow get it for the
//  more accurate cost estimation.
//
// TODO(wojtek-t): Figure out how to approach DELETECOLLECTION calls.
func (w *watchTracker) GetInterestedWatchCount(requestInfo *request.RequestInfo) int {
	if requestInfo == nil || readOnlyVerbs.Has(requestInfo.Verb) {
		return 0
	}

	result := 0
	// The watches that we're interested in include:
	// - watches for all objects of a resource type (no namespace and name specified)
	// - watches for all objects of a resource type in the same namespace (no name specified)
	// - watched interested in this particular object
	identifier := &watchIdentifier{
		apiGroup: requestInfo.APIGroup,
		resource: requestInfo.Resource,
	}

	w.lock.Lock()
	defer w.lock.Unlock()

	result += w.watchCount[*identifier]

	if requestInfo.Namespace != "" {
		identifier.namespace = requestInfo.Namespace
		result += w.watchCount[*identifier]
	}

	if requestInfo.Name != "" {
		identifier.name = requestInfo.Name
		result += w.watchCount[*identifier]
	}

	return result
}

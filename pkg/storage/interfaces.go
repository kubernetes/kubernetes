/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package storage

import (
	"time"

	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

// Versioner abstracts setting and retrieving metadata fields from database response
// onto the object ot list.
type Versioner interface {
	// UpdateObject sets storage metadata into an API object. Returns an error if the object
	// cannot be updated correctly. May return nil if the requested object does not need metadata
	// from database.
	UpdateObject(obj runtime.Object, expiration *time.Time, resourceVersion uint64) error
	// UpdateList sets the resource version into an API list object. Returns an error if the object
	// cannot be updated correctly. May return nil if the requested object does not need metadata
	// from database.
	UpdateList(obj runtime.Object, resourceVersion uint64) error
	// ObjectResourceVersion returns the resource version (for persistence) of the specified object.
	// Should return an error if the specified object does not have a persistable version.
	ObjectResourceVersion(obj runtime.Object) (uint64, error)
}

// ResponseMeta contains information about the database metadata that is associated with
// an object. It abstracts the actual underlying objects to prevent coupling with concrete
// database and to improve testability.
type ResponseMeta struct {
	// TTL is the time to live of the node that contained the returned object. It may be
	// zero or negative in some cases (objects may be expired after the requested
	// expiration time due to server lag).
	TTL int64
	// Expiration is the time at which the node that contained the returned object will expire and be deleted.
	// This can be nil if there is no expiration time set for the node.
	Expiration *time.Time
	// The resource version of the node that contained the returned object.
	ResourceVersion uint64
}

// FilterFunc is a predicate which takes an API object and returns true
// iff the object should remain in the set.
type FilterFunc func(obj runtime.Object) bool

// Everything is a FilterFunc which accepts all objects.
func Everything(runtime.Object) bool {
	return true
}

// Pass an UpdateFunc to Interface.GuaranteedUpdate to make an update
// that is guaranteed to succeed.
// See the comment for GuaranteedUpdate for more details.
type UpdateFunc func(input runtime.Object, res ResponseMeta) (output runtime.Object, ttl *uint64, err error)

// Interface offers a common interface for object marshaling/unmarshling operations and
// hides all the storage-related operations behind it.
type Interface interface {
	// Returns list of servers addresses of the underyling database.
	// TODO: This method is used only in a single place. Consider refactoring and getting rid
	// of this method from the interface.
	Backends() []string

	// Returns Versioner associated with this interface.
	Versioner() Versioner

	// Create adds a new object at a key unless it already exists. 'ttl' is time-to-live
	// in seconds (0 means forever). If no error is returned and out is not nil, out will be
	// set to the read value from database.
	Create(key string, obj, out runtime.Object, ttl uint64) error

	// Set marshals obj via json and stores in database under key. Will do an atomic update
	// if obj's ResourceVersion field is set. 'ttl' is time-to-live in seconds (0 means forever).
	// If no error is returned and out is not nil, out will be set to the read value from database.
	Set(key string, obj, out runtime.Object, ttl uint64) error

	// Delete removes the specified key and returns the value that existed at that spot.
	Delete(key string, out runtime.Object) error

	// Watch begins watching the specified key. Events are decoded into API objects,
	// and any items passing 'filter' are sent down to returned watch.Interface.
	// resourceVersion may be used to specify what version to begin watching
	// (e.g. reconnecting without missing any updates).
	Watch(key string, resourceVersion uint64, filter FilterFunc) (watch.Interface, error)

	// WatchList begins watching the specified key's items. Items are decoded into API
	// objects and any item passing 'filter' are sent down to returned watch.Interface.
	// resourceVersion may be used to specify what version to begin watching
	// (e.g. reconnecting without missing any updates).
	WatchList(key string, resourceVersion uint64, filter FilterFunc) (watch.Interface, error)

	// Get unmarshals json found at key into objPtr. On a not found error, will either
	// return a zero object of the requested type, or an error, depending on ignoreNotFound.
	// Treats empty responses and nil response nodes exactly like a not found error.
	Get(key string, objPtr runtime.Object, ignoreNotFound bool) error

	// GetToList unmarshals json found at key and opaque it into *List api object
	// (an object that satisfies the runtime.IsList definition).
	GetToList(key string, listObj runtime.Object) error

	// List unmarshalls jsons found at directory defined by key and opaque them
	// into *List api object (an object that satisfies runtime.IsList definition).
	List(key string, listObj runtime.Object) error

	// GuaranteedUpdate keeps calling 'tryUpdate()' to update key 'key' (of type 'ptrToType')
	// retrying the update until success if there is index conflict.
	// Note that object passed to tryUpdate may change acress incovations of tryUpdate() if
	// other writers are simultaneously updateing it, to tryUpdate() needs to take into account
	// the current contents of the object when deciding how the update object should look.
	//
	// Example:
	//
	// s := /* implementation of Interface */
	// err := s.GuaranteedUpdate(
	//     "myKey", &MyType{}, true,
	//     func(input runtime.Object, res ResponseMeta) (runtime.Object, *uint64, error) {
	//       // Before each incovation of the user defined function, "input" is reset to
	//       // current contents for "myKey" in database.
	//       curr := input.(*MyType)  // Guaranteed to succeed.
	//
	//       // Make the modification
	//       curr.Counter++
	//
	//       // Return the modified object - return an error to stop iterating. Return
	//       // a uint64 to alter the TTL on the object, or nil to keep it the same value.
	//       return cur, nil, nil
	//    }
	// })
	GuaranteedUpdate(key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate UpdateFunc) error

	// Codec provides access to the underlying codec being used by the implementation.
	Codec() runtime.Codec
}

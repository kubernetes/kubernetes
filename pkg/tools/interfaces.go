/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package tools

import (
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"

	"github.com/coreos/go-etcd/etcd"
)

const (
	EtcdErrorCodeNotFound      = 100
	EtcdErrorCodeTestFailed    = 101
	EtcdErrorCodeNodeExist     = 105
	EtcdErrorCodeValueRequired = 200
)

var (
	EtcdErrorNotFound      = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNotFound}
	EtcdErrorTestFailed    = &etcd.EtcdError{ErrorCode: EtcdErrorCodeTestFailed}
	EtcdErrorNodeExist     = &etcd.EtcdError{ErrorCode: EtcdErrorCodeNodeExist}
	EtcdErrorValueRequired = &etcd.EtcdError{ErrorCode: EtcdErrorCodeValueRequired}
)

// EtcdClient is an injectable interface for testing.
type EtcdClient interface {
	GetCluster() []string
	Get(key string, sort, recursive bool) (*etcd.Response, error)
	Set(key, value string, ttl uint64) (*etcd.Response, error)
	Create(key, value string, ttl uint64) (*etcd.Response, error)
	CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error)
	Delete(key string, recursive bool) (*etcd.Response, error)
	// I'd like to use directional channels here (e.g. <-chan) but this interface mimics
	// the etcd client interface which doesn't, and it doesn't seem worth it to wrap the api.
	Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error)
}

// StorageVersioner abstracts setting and retrieving metadata fields from the etcd response onto the object
// or list.
type StorageVersioner interface {
	// UpdateObject sets etcd storage metadata into an API object. Returns an error if the object
	// cannot be updated correctly. May return nil if the requested object does not need metadata
	// from etcd.
	UpdateObject(obj runtime.Object, expiration *time.Time, resourceVersion uint64) error
	// UpdateList sets the resource version into an API list object. Returns an error if the object
	// cannot be updated correctly. May return nil if the requested object does not need metadata
	// from etcd.
	UpdateList(obj runtime.Object, resourceVersion uint64) error
	// ObjectResourceVersion returns the resource version (for persistence) of the specified object.
	// Should return an error if the specified object does not have a persistable version.
	ObjectResourceVersion(obj runtime.Object) (uint64, error)
}

// ResponseMeta contains information about the etcd metadata that is associated with
// an object. It abstracts the actual underlying objects to prevent coupling with etcd
// and to improve testability.
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

// Pass an StorageUpdateFunc to StorageInterface.GuaranteedUpdate to make an update
// that is guaranteed to succeed.
// See the comment for GuaranteedUpdate for more details.
type StorageUpdateFunc func(input runtime.Object, res ResponseMeta) (output runtime.Object, ttl *uint64, err error)

// StorageInterface offers a common interface for object marshaling/unmarshling operations and
// hids all the storage-related operations behind it.
type StorageInterface interface {
	// Returns list of servers addresses of the underyling database.
	// TODO: This method is used only in a single place. Consider refactoring and getting rid
	// of this method from the interface.
	Backends() []string

	// Returns StorageVersioner associated with this interface.
	Versioner() StorageVersioner

	// Create adds a new object at a key unless it already exists. 'ttl' is time-to-live
	// in seconds (0 means forever). If no error is returned and out is not nil, out will be
	// set to the read value from etcd.
	Create(key string, obj, out runtime.Object, ttl uint64) error

	// Set marshals obj via json and stores in etcd under key. Will do an atomic update
	// if obj's ResourceVersion field is set. 'ttl' is time-to-live in seconds (0 means forever).
	// If no error is returned and out is not nil, out will be set to the read value from etcd.
	Set(key string, obj, out runtime.Object, ttl uint64) error

	// Delete removes the specified key and returns the value that existed at that spot.
	Delete(key string, out runtime.Object) error

	// RecursiveDelete removes the specified key.
	// TODO: Get rid of this method and use Delete() instead.
	RecursiveDelete(key string, recursive bool) error

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
	// retrying the update until success if there is etcd index conflict.
	// Note that object passed to tryUpdate may change acress incovations of tryUpdate() if
	// other writers are simultanously updateing it, to tryUpdate() needs to take into account
	// the current contents of the object when deciding how the update object should look.
	//
	// Exmaple:
	//
	// s := /* implementation of StorageInterface */
	// err := s.GuaranteedUpdate(
	//     "myKey", &MyType{}, true,
	//     func(input runtime.Object, res ResponseMeta) (runtime.Object, *uint64, error) {
	//       // Before each incovation of the user defined function, "input" is reset to
	//       // etcd's current contents for "myKey".
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
	GuaranteedUpdate(key string, ptrToType runtime.Object, ignoreNotFound bool, tryUpdate StorageUpdateFunc) error
}

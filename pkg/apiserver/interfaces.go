/*
Copyright 2014 Google Inc. All rights reserved.

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

package apiserver

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

// RESTStorage is a generic interface for RESTful storage services.
// Resources which are exported to the RESTful API of apiserver need to implement this interface.
type RESTStorage interface {
	// New returns an empty object that can be used with Create and Update after request data has been put into it.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	New() runtime.Object

	// List selects resources in the storage which match to the selector.
	List(ctx api.Context, label, field labels.Selector) (runtime.Object, error)

	// Get finds a resource in the storage by id and returns it.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	Get(ctx api.Context, id string) (runtime.Object, error)

	// Delete finds a resource in the storage and deletes it.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	Delete(ctx api.Context, id string) (<-chan RESTResult, error)

	// Create creates a new version of a resource.
	Create(ctx api.Context, obj runtime.Object) (<-chan RESTResult, error)

	// Update finds a resource in the storage and updates it. Some implementations
	// may allow updates creates the object - they should set the Created flag of
	// the returned RESTResultto true. In the event of an asynchronous error returned
	// via an api.Status object, the Created flag is ignored.
	Update(ctx api.Context, obj runtime.Object) (<-chan RESTResult, error)
}

// RESTResult indicates the result of a REST transformation.
type RESTResult struct {
	// The result of this operation. May be nil if the operation has no meaningful
	// result (like Delete)
	runtime.Object

	// May be set true to indicate that the Update operation resulted in the object
	// being created.
	Created bool
}

// ResourceWatcher should be implemented by all RESTStorage objects that
// want to offer the ability to watch for changes through the watch api.
type ResourceWatcher interface {
	// 'label' selects on labels; 'field' selects on the object's fields. Not all fields
	// are supported; an error should be returned if 'field' tries to select on a field that
	// isn't supported. 'resourceVersion' allows for continuing/starting a watch at a
	// particular version.
	Watch(ctx api.Context, label, field labels.Selector, resourceVersion string) (watch.Interface, error)
}

// Redirector know how to return a remote resource's location.
type Redirector interface {
	// ResourceLocation should return the remote location of the given resource, or an error.
	ResourceLocation(ctx api.Context, id string) (remoteLocation string, err error)
}

/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"io"
	"net/http"
	"net/url"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

//TODO:
// Storage interfaces need to be separated into two groups; those that operate
// on collections and those that operate on individually named items.
// Collection interfaces:
// (Method: Current -> Proposed)
//    GET: Lister -> CollectionGetter
//    WATCH: Watcher -> CollectionWatcher
//    CREATE: Creater -> CollectionCreater
//    DELETE: (n/a) -> CollectionDeleter
//    UPDATE: (n/a) -> CollectionUpdater
//
// Single item interfaces:
// (Method: Current -> Proposed)
//    GET: Getter -> NamedGetter
//    WATCH: (n/a) -> NamedWatcher
//    CREATE: (n/a) -> NamedCreater
//    DELETE: Deleter -> NamedDeleter
//    UPDATE: Update -> NamedUpdater

// Storage is a generic interface for RESTful storage services.
// Resources which are exported to the RESTful API of apiserver need to implement this interface. It is expected
// that objects may implement any of the below interfaces.
type Storage interface {
	// New returns an empty object that can be used with Create and Update after request data has been put into it.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	New() runtime.Object
}

// KindProvider specifies a different kind for its API than for its internal storage.  This is necessary for external
// objects that are not compiled into the api server.  For such objects, there is no in-memory representation for
// the object, so they must be represented as generic objects (e.g. runtime.Unknown), but when we present the object as part of
// API discovery we want to present the specific kind, not the generic internal representation.
type KindProvider interface {
	Kind() string
}

// Lister is an object that can retrieve resources that match the provided field and label criteria.
type Lister interface {
	// NewList returns an empty object that can be used with the List call.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	NewList() runtime.Object
	// List selects resources in the storage which match to the selector. 'options' can be nil.
	List(ctx api.Context, options *api.ListOptions) (runtime.Object, error)
}

// Exporter is an object that knows how to strip a RESTful resource for export
type Exporter interface {
	// Export an object.  Fields that are not user specified (e.g. Status, ObjectMeta.ResourceVersion) are stripped out
	// Returns the stripped object.  If 'exact' is true, fields that are specific to the cluster (e.g. namespace) are
	// retained, otherwise they are stripped also.
	Export(ctx api.Context, name string, opts unversioned.ExportOptions) (runtime.Object, error)
}

// Getter is an object that can retrieve a named RESTful resource.
type Getter interface {
	// Get finds a resource in the storage by name and returns it.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	Get(ctx api.Context, name string) (runtime.Object, error)
}

// GetterWithOptions is an object that retrieve a named RESTful resource and takes
// additional options on the get request. It allows a caller to also receive the
// subpath of the GET request.
type GetterWithOptions interface {
	// Get finds a resource in the storage by name and returns it.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	// The options object passed to it is of the same type returned by the NewGetOptions
	// method.
	Get(ctx api.Context, name string, options runtime.Object) (runtime.Object, error)

	// NewGetOptions returns an empty options object that will be used to pass
	// options to the Get method. It may return a bool and a string, if true, the
	// value of the request path below the object will be included as the named
	// string in the serialization of the runtime object. E.g., returning "path"
	// will convert the trailing request scheme value to "path" in the map[string][]string
	// passed to the converter.
	NewGetOptions() (runtime.Object, bool, string)
}

// Deleter is an object that can delete a named RESTful resource.
type Deleter interface {
	// Delete finds a resource in the storage and deletes it.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	// Delete *may* return the object that was deleted, or a status object indicating additional
	// information about deletion.
	Delete(ctx api.Context, name string) (runtime.Object, error)
}

// GracefulDeleter knows how to pass deletion options to allow delayed deletion of a
// RESTful object.
type GracefulDeleter interface {
	// Delete finds a resource in the storage and deletes it.
	// If options are provided, the resource will attempt to honor them or return an invalid
	// request error.
	// Although it can return an arbitrary error value, IsNotFound(err) is true for the
	// returned error value err when the specified resource is not found.
	// Delete *may* return the object that was deleted, or a status object indicating additional
	// information about deletion.
	Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error)
}

// GracefulDeleteAdapter adapts the Deleter interface to GracefulDeleter
type GracefulDeleteAdapter struct {
	Deleter
}

// Delete implements RESTGracefulDeleter in terms of Deleter
func (w GracefulDeleteAdapter) Delete(ctx api.Context, name string, options *api.DeleteOptions) (runtime.Object, error) {
	return w.Deleter.Delete(ctx, name)
}

// CollectionDeleter is an object that can delete a collection
// of RESTful resources.
type CollectionDeleter interface {
	// DeleteCollection selects all resources in the storage matching given 'listOptions'
	// and deletes them. If 'options' are provided, the resource will attempt to honor
	// them or return an invalid request error.
	// DeleteCollection may not be atomic - i.e. it may delete some objects and still
	// return an error after it. On success, returns a list of deleted objects.
	DeleteCollection(ctx api.Context, options *api.DeleteOptions, listOptions *api.ListOptions) (runtime.Object, error)
}

// Creater is an object that can create an instance of a RESTful object.
type Creater interface {
	// New returns an empty object that can be used with Create after request data has been put into it.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	New() runtime.Object

	// Create creates a new version of a resource.
	Create(ctx api.Context, obj runtime.Object) (runtime.Object, error)
}

// NamedCreater is an object that can create an instance of a RESTful object using a name parameter.
type NamedCreater interface {
	// New returns an empty object that can be used with Create after request data has been put into it.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	New() runtime.Object

	// Create creates a new version of a resource. It expects a name parameter from the path.
	// This is needed for create operations on subresources which include the name of the parent
	// resource in the path.
	Create(ctx api.Context, name string, obj runtime.Object) (runtime.Object, error)
}

// UpdatedObjectInfo provides information about an updated object to an Updater.
// It requires access to the old object in order to return the newly updated object.
type UpdatedObjectInfo interface {
	// Returns preconditions built from the updated object, if applicable.
	// May return nil, or a preconditions object containing nil fields,
	// if no preconditions can be determined from the updated object.
	Preconditions() *api.Preconditions

	// UpdatedObject returns the updated object, given a context and old object.
	// The only time an empty oldObj should be passed in is if a "create on update" is occurring (there is no oldObj).
	UpdatedObject(ctx api.Context, oldObj runtime.Object) (newObj runtime.Object, err error)
}

// Updater is an object that can update an instance of a RESTful object.
type Updater interface {
	// New returns an empty object that can be used with Update after request data has been put into it.
	// This object must be a pointer type for use with Codec.DecodeInto([]byte, runtime.Object)
	New() runtime.Object

	// Update finds a resource in the storage and updates it. Some implementations
	// may allow updates creates the object - they should set the created boolean
	// to true.
	Update(ctx api.Context, name string, objInfo UpdatedObjectInfo) (runtime.Object, bool, error)
}

// CreaterUpdater is a storage object that must support both create and update.
// Go prevents embedded interfaces that implement the same method.
type CreaterUpdater interface {
	Creater
	Update(ctx api.Context, name string, objInfo UpdatedObjectInfo) (runtime.Object, bool, error)
}

// CreaterUpdater must satisfy the Updater interface.
var _ Updater = CreaterUpdater(nil)

// Patcher is a storage object that supports both get and update.
type Patcher interface {
	Getter
	Updater
}

// Watcher should be implemented by all Storage objects that
// want to offer the ability to watch for changes through the watch api.
type Watcher interface {
	// 'label' selects on labels; 'field' selects on the object's fields. Not all fields
	// are supported; an error should be returned if 'field' tries to select on a field that
	// isn't supported. 'resourceVersion' allows for continuing/starting a watch at a
	// particular version.
	Watch(ctx api.Context, options *api.ListOptions) (watch.Interface, error)
}

// StandardStorage is an interface covering the common verbs. Provided for testing whether a
// resource satisfies the normal storage methods. Use Storage when passing opaque storage objects.
type StandardStorage interface {
	Getter
	Lister
	CreaterUpdater
	GracefulDeleter
	CollectionDeleter
	Watcher
}

// Redirector know how to return a remote resource's location.
type Redirector interface {
	// ResourceLocation should return the remote location of the given resource, and an optional transport to use to request it, or an error.
	ResourceLocation(ctx api.Context, id string) (remoteLocation *url.URL, transport http.RoundTripper, err error)
}

// Responder abstracts the normal response behavior for a REST method and is passed to callers that
// may wish to handle the response directly in some cases, but delegate to the normal error or object
// behavior in other cases.
type Responder interface {
	// Object writes the provided object to the response. Invoking this method multiple times is undefined.
	Object(statusCode int, obj runtime.Object)
	// Error writes the provided error to the response. This method may only be invoked once.
	Error(err error)
}

// Connecter is a storage object that responds to a connection request.
type Connecter interface {
	// Connect returns an http.Handler that will handle the request/response for a given API invocation.
	// The provided responder may be used for common API responses. The responder will write both status
	// code and body, so the ServeHTTP method should exit after invoking the responder. The Handler will
	// be used for a single API request and then discarded. The Responder is guaranteed to write to the
	// same http.ResponseWriter passed to ServeHTTP.
	Connect(ctx api.Context, id string, options runtime.Object, r Responder) (http.Handler, error)

	// NewConnectOptions returns an empty options object that will be used to pass
	// options to the Connect method. If nil, then a nil options object is passed to
	// Connect. It may return a bool and a string. If true, the value of the request
	// path below the object will be included as the named string in the serialization
	// of the runtime object.
	NewConnectOptions() (runtime.Object, bool, string)

	// ConnectMethods returns the list of HTTP methods handled by Connect
	ConnectMethods() []string
}

// ResourceStreamer is an interface implemented by objects that prefer to be streamed from the server
// instead of decoded directly.
type ResourceStreamer interface {
	// InputStream should return an io.ReadCloser if the provided object supports streaming. The desired
	// api version and an accept header (may be empty) are passed to the call. If no error occurs,
	// the caller may return a flag indicating whether the result should be flushed as writes occur
	// and a content type string that indicates the type of the stream.
	// If a null stream is returned, a StatusNoContent response wil be generated.
	InputStream(apiVersion, acceptHeader string) (stream io.ReadCloser, flush bool, mimeType string, err error)
}

// StorageMetadata is an optional interface that callers can implement to provide additional
// information about their Storage objects.
type StorageMetadata interface {
	// ProducesMIMETypes returns a list of the MIME types the specified HTTP verb (GET, POST, DELETE,
	// PATCH) can respond with.
	ProducesMIMETypes(verb string) []string
}

// ConnectRequest is an object passed to admission control for Connect operations
type ConnectRequest struct {
	// Name is the name of the object on which the connect request was made
	Name string

	// Options is the options object passed to the connect request. See the NewConnectOptions method on Connecter
	Options runtime.Object

	// ResourcePath is the path for the resource in the REST server (ie. "pods/proxy")
	ResourcePath string
}

func (obj *ConnectRequest) GetObjectKind() unversioned.ObjectKind { return unversioned.EmptyObjectKind }

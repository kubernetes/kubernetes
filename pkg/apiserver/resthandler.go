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

package apiserver

import (
	"encoding/json"
	"fmt"
	"math/rand"
	"net/http"
	"net/url"
	gpath "path"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/strategicpatch"

	"github.com/emicklei/go-restful"
	"github.com/evanphx/json-patch"
	"github.com/golang/glog"
)

// ContextFunc returns a Context given a request - a context must be returned
type ContextFunc func(req *restful.Request) api.Context

// ScopeNamer handles accessing names from requests and objects
type ScopeNamer interface {
	// Namespace returns the appropriate namespace value from the request (may be empty) or an
	// error.
	Namespace(req *restful.Request) (namespace string, err error)
	// Name returns the name from the request, and an optional namespace value if this is a namespace
	// scoped call. An error is returned if the name is not available.
	Name(req *restful.Request) (namespace, name string, err error)
	// ObjectName returns the namespace and name from an object if they exist, or an error if the object
	// does not support names.
	ObjectName(obj runtime.Object) (namespace, name string, err error)
	// SetSelfLink sets the provided URL onto the object. The method should return nil if the object
	// does not support selfLinks.
	SetSelfLink(obj runtime.Object, url string) error
	// GenerateLink creates a path and query for a given runtime object that represents the canonical path.
	GenerateLink(req *restful.Request, obj runtime.Object) (path, query string, err error)
	// GenerateLink creates a path and query for a list that represents the canonical path.
	GenerateListLink(req *restful.Request) (path, query string, err error)
}

// RequestScope encapsulates common fields across all RESTful handler methods.
type RequestScope struct {
	Namer ScopeNamer
	ContextFunc
	runtime.Codec
	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor

	Resource    unversioned.GroupVersionResource
	Kind        unversioned.GroupVersionKind
	Subresource string
}

// getterFunc performs a get request with the given context and object name. The request
// may be used to deserialize an options object to pass to the getter.
type getterFunc func(ctx api.Context, name string, req *restful.Request) (runtime.Object, error)

// MaxPatchConflicts is the maximum number of conflicts retry for during a patch operation before returning failure
const MaxPatchConflicts = 5

// getResourceHandler is an HTTP handler function for get requests. It delegates to the
// passed-in getterFunc to perform the actual get.
func getResourceHandler(scope RequestScope, getter getterFunc) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter
		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		result, err := getter(ctx, name, req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		if err := setSelfLink(result, req, scope.Namer); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
	}
}

// GetResource returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResource(r rest.Getter, e rest.Exporter, scope RequestScope) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx api.Context, name string, req *restful.Request) (runtime.Object, error) {
			// For performance tracking purposes.
			trace := util.NewTrace("Get " + req.Request.URL.Path)
			defer trace.LogIfLong(250 * time.Millisecond)
			opts := v1.ExportOptions{}
			if err := scope.Codec.DecodeParametersInto(req.Request.URL.Query(), &opts); err != nil {
				return nil, err
			}
			internalOpts := unversioned.ExportOptions{}
			scope.Convertor.Convert(&opts, &internalOpts)
			if internalOpts.Export {
				if e == nil {
					return nil, errors.NewBadRequest("export unsupported")
				}
				return e.Export(ctx, name, internalOpts)
			}
			return r.Get(ctx, name)
		})
}

// GetResourceWithOptions returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResourceWithOptions(r rest.GetterWithOptions, e rest.Exporter, scope RequestScope, internalKind, externalKind unversioned.GroupVersionKind, subpath bool, subpathKey string) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx api.Context, name string, req *restful.Request) (runtime.Object, error) {
			opts, err := getRequestOptions(req, scope, internalKind, externalKind, subpath, subpathKey)
			if err != nil {
				return nil, err
			}
			exportOpts := unversioned.ExportOptions{}
			if err := scope.Codec.DecodeParametersInto(req.Request.URL.Query(), &exportOpts); err != nil {
				return nil, err
			}
			if exportOpts.Export {
				return nil, errors.NewBadRequest("export unsupported")
			}
			return r.Get(ctx, name, opts)
		})
}

func getRequestOptions(req *restful.Request, scope RequestScope, internalKind, externalKind unversioned.GroupVersionKind, subpath bool, subpathKey string) (runtime.Object, error) {
	if internalKind.IsEmpty() {
		return nil, nil
	}

	query := req.Request.URL.Query()
	if subpath {
		newQuery := make(url.Values)
		for k, v := range query {
			newQuery[k] = v
		}
		newQuery[subpathKey] = []string{req.PathParameter("path")}
		query = newQuery
	}

	versioned, err := scope.Creater.New(externalKind)
	if err != nil {
		return nil, err
	}

	if err := scope.Codec.DecodeParametersInto(query, versioned); err != nil {
		return nil, errors.NewBadRequest(err.Error())
	}
	out, err := scope.Convertor.ConvertToVersion(versioned, internalKind.GroupVersion().String())
	if err != nil {
		// programmer error
		return nil, err
	}
	return out, nil
}

// ConnectResource returns a function that handles a connect request on a rest.Storage object.
func ConnectResource(connecter rest.Connecter, scope RequestScope, admit admission.Interface, internalKind, externalKind unversioned.GroupVersionKind, restPath string, subpath bool, subpathKey string) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter
		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)
		opts, err := getRequestOptions(req, scope, internalKind, externalKind, subpath, subpathKey)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		if admit.Handles(admission.Connect) {
			connectRequest := &rest.ConnectRequest{
				Name:         name,
				Options:      opts,
				ResourcePath: restPath,
			}
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(connectRequest, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Connect, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}
		handler, err := connecter.Connect(ctx, name, opts, &responder{scope: scope, req: req.Request, w: w})
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		handler.ServeHTTP(w, req.Request)
	}
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	scope RequestScope
	req   *http.Request
	w     http.ResponseWriter
}

func (r *responder) Object(statusCode int, obj runtime.Object) {
	write(statusCode, r.scope.Kind.GroupVersion(), r.scope.Codec, obj, r.w, r.req)
}

func (r *responder) Error(err error) {
	errorJSON(err, r.scope.Codec, r.w)
}

// ListResource returns a function that handles retrieving a list of resources from a rest.Storage object.
func ListResource(r rest.Lister, rw rest.Watcher, scope RequestScope, forceWatch bool, minRequestTimeout time.Duration) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := util.NewTrace("List " + req.Request.URL.Path)

		w := res.ResponseWriter

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		// Watches for single objects are routed to this function.
		// Treat a /name parameter the same as a field selector entry.
		hasName := true
		_, name, err := scope.Namer.Name(req)
		if err != nil {
			hasName = false
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		listOptionsGVK := scope.Kind.GroupVersion().WithKind("ListOptions")
		versioned, err := scope.Creater.New(listOptionsGVK)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		if err := scope.Codec.DecodeParametersInto(req.Request.URL.Query(), versioned); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		opts := api.ListOptions{}
		if err := scope.Convertor.Convert(versioned, &opts); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		// transform fields
		// TODO: DecodeParametersInto should do this.
		if opts.FieldSelector != nil {
			fn := func(label, value string) (newLabel, newValue string, err error) {
				return scope.Convertor.ConvertFieldLabel(scope.Kind.GroupVersion().String(), scope.Kind.Kind, label, value)
			}
			if opts.FieldSelector, err = opts.FieldSelector.Transform(fn); err != nil {
				// TODO: allow bad request to set field causes based on query parameters
				err = errors.NewBadRequest(err.Error())
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		if hasName {
			// metadata.name is the canonical internal name.
			// generic.SelectionPredicate will notice that this is
			// a request for a single object and optimize the
			// storage query accordingly.
			nameSelector := fields.OneTermEqualSelector("metadata.name", name)
			if opts.FieldSelector != nil && !opts.FieldSelector.Empty() {
				// It doesn't make sense to ask for both a name
				// and a field selector, since just the name is
				// sufficient to narrow down the request to a
				// single object.
				errorJSON(
					errors.NewBadRequest("both a name and a field selector provided; please provide one or the other."),
					scope.Codec,
					w,
				)
				return
			}
			opts.FieldSelector = nameSelector
		}

		if (opts.Watch || forceWatch) && rw != nil {
			watcher, err := rw.Watch(ctx, &opts)
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
			// TODO: Currently we explicitly ignore ?timeout= and use only ?timeoutSeconds=.
			timeout := time.Duration(0)
			if opts.TimeoutSeconds != nil {
				timeout = time.Duration(*opts.TimeoutSeconds) * time.Second
			}
			if timeout == 0 && minRequestTimeout > 0 {
				timeout = time.Duration(float64(minRequestTimeout) * (rand.Float64() + 1.0))
			}
			serveWatch(watcher, scope, w, req, timeout)
			return
		}

		// Log only long List requests (ignore Watch).
		defer trace.LogIfLong(500 * time.Millisecond)
		trace.Step("About to List from storage")
		result, err := r.List(ctx, &opts)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Listing from storage done")
		numberOfItems, err := setListSelfLink(result, req, scope.Namer)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Self-linking done")
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
		trace.Step(fmt.Sprintf("Writing http response done (%d items)", numberOfItems))
	}
}

func createHandler(r rest.NamedCreater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface, includeName bool) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := util.NewTrace("Create " + req.Request.URL.Path)
		defer trace.LogIfLong(250 * time.Millisecond)

		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		var (
			namespace, name string
			err             error
		)
		if includeName {
			namespace, name, err = scope.Namer.Name(req)
		} else {
			namespace, err = scope.Namer.Namespace(req)
		}
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		obj := r.New()
		trace.Step("About to convert to expected version")
		// TODO this cleans up with proper typing
		if err := scope.Codec.DecodeIntoWithSpecifiedVersionKind(body, obj, scope.Kind); err != nil {
			err = transformDecodeError(typer, err, obj, body)
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Conversion done")

		if admit != nil && admit.Handles(admission.Create) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Create, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		trace.Step("About to store object in database")
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			out, err := r.Create(ctx, name, obj)
			if status, ok := out.(*unversioned.Status); ok && err == nil && status.Code == 0 {
				status.Code = http.StatusCreated
			}
			return out, err
		})
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Object stored in database")

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Self-link added")

		write(http.StatusCreated, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
	}
}

// CreateNamedResource returns a function that will handle a resource creation with name.
func CreateNamedResource(r rest.NamedCreater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) restful.RouteFunction {
	return createHandler(r, scope, typer, admit, true)
}

// CreateResource returns a function that will handle a resource creation.
func CreateResource(r rest.Creater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) restful.RouteFunction {
	return createHandler(&namedCreaterAdapter{r}, scope, typer, admit, false)
}

type namedCreaterAdapter struct {
	rest.Creater
}

func (c *namedCreaterAdapter) Create(ctx api.Context, name string, obj runtime.Object) (runtime.Object, error) {
	return c.Creater.Create(ctx, obj)
}

// PatchResource returns a function that will handle a resource patch
// TODO: Eventually PatchResource should just use GuaranteedUpdate and this routine should be a bit cleaner
func PatchResource(r rest.Patcher, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface, converter runtime.ObjectConvertor) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we
		// document, move timeout out of this function and declare it in
		// api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		obj := r.New()
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		// PATCH requires same permission as UPDATE
		if admit.Handles(admission.Update) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Update, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		versionedObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion().String())
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		contentType := req.HeaderParameter("Content-Type")
		// Remove "; charset=" if included in header.
		if idx := strings.Index(contentType, ";"); idx > 0 {
			contentType = contentType[:idx]
		}
		patchType := api.PatchType(contentType)

		patchJS, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		result, err := patchResource(ctx, timeout, versionedObj, r, name, patchType, patchJS, scope.Namer, scope.Codec)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
	}

}

// patchResource divides PatchResource for easier unit testing
func patchResource(ctx api.Context, timeout time.Duration, versionedObj runtime.Object, patcher rest.Patcher, name string, patchType api.PatchType, patchJS []byte, namer ScopeNamer, codec runtime.Codec) (runtime.Object, error) {
	namespace := api.NamespaceValue(ctx)

	original, err := patcher.Get(ctx, name)
	if err != nil {
		return nil, err
	}

	originalObjJS, err := runtime.Encode(codec, original)
	if err != nil {
		return nil, err
	}
	originalPatchedObjJS, err := getPatchedJS(patchType, originalObjJS, patchJS, versionedObj)
	if err != nil {
		return nil, err
	}

	objToUpdate := patcher.New()
	if err := runtime.DecodeInto(codec, originalPatchedObjJS, objToUpdate); err != nil {
		return nil, err
	}
	if err := checkName(objToUpdate, name, namespace, namer); err != nil {
		return nil, err
	}

	return finishRequest(timeout, func() (runtime.Object, error) {
		// update should never create as previous get would fail
		updateObject, _, updateErr := patcher.Update(ctx, objToUpdate)
		for i := 0; i < MaxPatchConflicts && (errors.IsConflict(updateErr)); i++ {

			// on a conflict,
			// 1. build a strategic merge patch from originalJS and the patchedJS.  Different patch types can
			//    be specified, but a strategic merge patch should be expressive enough handle them.  Build the
			//    patch with this type to handle those cases.
			// 2. build a strategic merge patch from originalJS and the currentJS
			// 3. ensure no conflicts between the two patches
			// 4. apply the #1 patch to the currentJS object
			// 5. retry the update
			currentObject, err := patcher.Get(ctx, name)
			if err != nil {
				return nil, err
			}
			currentObjectJS, err := runtime.Encode(codec, currentObject)
			if err != nil {
				return nil, err
			}

			currentPatch, err := strategicpatch.CreateStrategicMergePatch(originalObjJS, currentObjectJS, patcher.New())
			if err != nil {
				return nil, err
			}
			originalPatch, err := strategicpatch.CreateStrategicMergePatch(originalObjJS, originalPatchedObjJS, patcher.New())
			if err != nil {
				return nil, err
			}

			diff1 := make(map[string]interface{})
			if err := json.Unmarshal(originalPatch, &diff1); err != nil {
				return nil, err
			}
			diff2 := make(map[string]interface{})
			if err := json.Unmarshal(currentPatch, &diff2); err != nil {
				return nil, err
			}
			hasConflicts, err := strategicpatch.HasConflicts(diff1, diff2)
			if err != nil {
				return nil, err
			}
			if hasConflicts {
				return updateObject, updateErr
			}

			newlyPatchedObjJS, err := getPatchedJS(api.StrategicMergePatchType, currentObjectJS, originalPatch, versionedObj)
			if err != nil {
				return nil, err
			}
			if err := runtime.DecodeInto(codec, newlyPatchedObjJS, objToUpdate); err != nil {
				return nil, err
			}

			updateObject, _, updateErr = patcher.Update(ctx, objToUpdate)
		}

		return updateObject, updateErr
	})
}

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := util.NewTrace("Update " + req.Request.URL.Path)
		defer trace.LogIfLong(250 * time.Millisecond)

		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		obj := r.New()
		trace.Step("About to convert to expected version")
		if err := scope.Codec.DecodeIntoWithSpecifiedVersionKind(body, obj, scope.Kind); err != nil {
			err = transformDecodeError(typer, err, obj, body)
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Conversion done")

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		if admit != nil && admit.Handles(admission.Update) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Update, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		trace.Step("About to store object in database")
		wasCreated := false
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, created, err := r.Update(ctx, obj)
			wasCreated = created
			return obj, err
		})
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Object stored in database")

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Self-link added")

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}
		writeJSON(status, scope.Codec, result, w, isPrettyPrint(req.Request))
	}
}

// DeleteResource returns a function that will handle a resource deletion
func DeleteResource(r rest.GracefulDeleter, checkBody bool, scope RequestScope, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := util.NewTrace("Delete " + req.Request.URL.Path)
		defer trace.LogIfLong(250 * time.Millisecond)

		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		options := &api.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
			if len(body) > 0 {
				if err := scope.Codec.DecodeInto(body, options); err != nil {
					errorJSON(err, scope.Codec, w)
					return
				}
			}
		}

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		trace.Step("About do delete object from database")
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.Delete(ctx, name, options)
		})
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		trace.Step("Object deleted from database")

		// if the rest.Deleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &unversioned.Status{
				Status: unversioned.StatusSuccess,
				Code:   http.StatusOK,
				Details: &unversioned.StatusDetails{
					Name: name,
					Kind: scope.Kind.Kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*unversioned.Status); !ok {
				if err := setSelfLink(result, req, scope.Namer); err != nil {
					errorJSON(err, scope.Codec, w)
					return
				}
			}
		}
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
	}
}

// DeleteCollection returns a function that will handle a collection deletion
func DeleteCollection(r rest.CollectionDeleter, checkBody bool, scope RequestScope, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, scope.Kind.GroupKind(), namespace, "", scope.Resource.GroupResource(), scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		listOptionsGVK := scope.Kind.GroupVersion().WithKind("ListOptions")
		versioned, err := scope.Creater.New(listOptionsGVK)
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		if err := scope.Codec.DecodeParametersInto(req.Request.URL.Query(), versioned); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}
		listOptions := api.ListOptions{}
		if err := scope.Convertor.Convert(versioned, &listOptions); err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		// transform fields
		// TODO: DecodeParametersInto should do this.
		if listOptions.FieldSelector != nil {
			fn := func(label, value string) (newLabel, newValue string, err error) {
				return scope.Convertor.ConvertFieldLabel(scope.Kind.GroupVersion().String(), scope.Kind.Kind, label, value)
			}
			if listOptions.FieldSelector, err = listOptions.FieldSelector.Transform(fn); err != nil {
				// TODO: allow bad request to set field causes based on query parameters
				err = errors.NewBadRequest(err.Error())
				errorJSON(err, scope.Codec, w)
				return
			}
		}

		options := &api.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				errorJSON(err, scope.Codec, w)
				return
			}
			if len(body) > 0 {
				if err := scope.Codec.DecodeInto(body, options); err != nil {
					errorJSON(err, scope.Codec, w)
					return
				}
			}
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.DeleteCollection(ctx, options, &listOptions)
		})
		if err != nil {
			errorJSON(err, scope.Codec, w)
			return
		}

		// if the rest.Deleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &unversioned.Status{
				Status: unversioned.StatusSuccess,
				Code:   http.StatusOK,
				Details: &unversioned.StatusDetails{
					Kind: scope.Kind.Kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*unversioned.Status); !ok {
				if _, err := setListSelfLink(result, req, scope.Namer); err != nil {
					errorJSON(err, scope.Codec, w)
					return
				}
			}
		}
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Codec, result, w, req.Request)
	}
}

// resultFunc is a function that returns a rest result and can be run in a goroutine
type resultFunc func() (runtime.Object, error)

// finishRequest makes a given resultFunc asynchronous and handles errors returned by the response.
// Any api.Status object returned is considered an "error", which interrupts the normal response flow.
func finishRequest(timeout time.Duration, fn resultFunc) (result runtime.Object, err error) {
	// these channels need to be buffered to prevent the goroutine below from hanging indefinitely
	// when the select statement reads something other than the one the goroutine sends on.
	ch := make(chan runtime.Object, 1)
	errCh := make(chan error, 1)
	panicCh := make(chan interface{}, 1)
	go func() {
		// panics don't cross goroutine boundaries, so we have to handle ourselves
		defer util.HandleCrash(func(panicReason interface{}) {
			// Propagate to parent goroutine
			panicCh <- panicReason
		})

		if result, err := fn(); err != nil {
			errCh <- err
		} else {
			ch <- result
		}
	}()

	select {
	case result = <-ch:
		if status, ok := result.(*unversioned.Status); ok {
			return nil, errors.FromObject(status)
		}
		return result, nil
	case err = <-errCh:
		return nil, err
	case p := <-panicCh:
		panic(p)
	case <-time.After(timeout):
		return nil, errors.NewTimeoutError("request did not complete within allowed duration", 0)
	}
}

// transformDecodeError adds additional information when a decode fails.
func transformDecodeError(typer runtime.ObjectTyper, baseErr error, into runtime.Object, body []byte) error {
	objectGroupVersionKind, err := typer.ObjectKind(into)
	if err != nil {
		return err
	}
	if dataGroupVersionKind, err := typer.DataKind(body); err == nil && len(dataGroupVersionKind.Kind) > 0 {
		return errors.NewBadRequest(fmt.Sprintf("%s in version %v cannot be handled as a %s: %v", dataGroupVersionKind.Kind, dataGroupVersionKind.GroupVersion(), objectGroupVersionKind.Kind, baseErr))
	}
	return errors.NewBadRequest(fmt.Sprintf("the object provided is unrecognized (must be of type %s): %v", objectGroupVersionKind.Kind, baseErr))
}

// setSelfLink sets the self link of an object (or the child items in a list) to the base URL of the request
// plus the path and query generated by the provided linkFunc
func setSelfLink(obj runtime.Object, req *restful.Request, namer ScopeNamer) error {
	// TODO: SelfLink generation should return a full URL?
	path, query, err := namer.GenerateLink(req, obj)
	if err != nil {
		return nil
	}

	newURL := *req.Request.URL
	// use only canonical paths
	newURL.Path = gpath.Clean(path)
	newURL.RawQuery = query
	newURL.Fragment = ""

	return namer.SetSelfLink(obj, newURL.String())
}

// checkName checks the provided name against the request
func checkName(obj runtime.Object, name, namespace string, namer ScopeNamer) error {
	if objNamespace, objName, err := namer.ObjectName(obj); err == nil {
		if err != nil {
			return err
		}
		if objName != name {
			return errors.NewBadRequest(fmt.Sprintf(
				"the name of the object (%s) does not match the name on the URL (%s)", objName, name))
		}
		if len(namespace) > 0 {
			if len(objNamespace) > 0 && objNamespace != namespace {
				return errors.NewBadRequest(fmt.Sprintf(
					"the namespace of the object (%s) does not match the namespace on the request (%s)", objNamespace, namespace))
			}
		}
	}
	return nil
}

// setListSelfLink sets the self link of a list to the base URL, then sets the self links
// on all child objects returned. Returns the number of items in the list.
func setListSelfLink(obj runtime.Object, req *restful.Request, namer ScopeNamer) (int, error) {
	if !meta.IsListType(obj) {
		return 0, nil
	}

	// TODO: List SelfLink generation should return a full URL?
	path, query, err := namer.GenerateListLink(req)
	if err != nil {
		return 0, err
	}
	newURL := *req.Request.URL
	newURL.Path = path
	newURL.RawQuery = query
	// use the path that got us here
	newURL.Fragment = ""
	if err := namer.SetSelfLink(obj, newURL.String()); err != nil {
		glog.V(4).Infof("Unable to set self link on object: %v", err)
	}

	// Set self-link of objects in the list.
	items, err := meta.ExtractList(obj)
	if err != nil {
		return 0, err
	}
	for i := range items {
		if err := setSelfLink(items[i], req, namer); err != nil {
			return len(items), err
		}
	}
	return len(items), meta.SetList(obj, items)
}

func getPatchedJS(patchType api.PatchType, originalJS, patchJS []byte, obj runtime.Object) ([]byte, error) {
	switch patchType {
	case api.JSONPatchType:
		patchObj, err := jsonpatch.DecodePatch(patchJS)
		if err != nil {
			return nil, err
		}
		return patchObj.Apply(originalJS)
	case api.MergePatchType:
		return jsonpatch.MergePatch(originalJS, patchJS)
	case api.StrategicMergePatchType:
		return strategicpatch.StrategicMergePatchData(originalJS, patchJS, obj)
	default:
		// only here as a safety net - go-restful filters content-type
		return nil, fmt.Errorf("unknown Content-Type header for patch: %v", patchType)
	}
}

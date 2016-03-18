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
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
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
	Serializer runtime.NegotiatedSerializer
	runtime.ParameterCodec
	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor

	Resource    unversioned.GroupVersionResource
	Kind        unversioned.GroupVersionKind
	Subresource string
}

func (scope *RequestScope) err(err error, w http.ResponseWriter, req *http.Request) {
	errorNegotiated(err, scope.Serializer, scope.Kind.GroupVersion(), w, req)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		result, err := getter(ctx, name, req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
	}
}

// GetResource returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResource(r rest.Getter, e rest.Exporter, scope RequestScope) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx api.Context, name string, req *restful.Request) (runtime.Object, error) {
			// For performance tracking purposes.
			trace := util.NewTrace("Get " + req.Request.URL.Path)
			defer trace.LogIfLong(250 * time.Millisecond)

			// check for export
			if values := req.Request.URL.Query(); len(values) > 0 {
				// TODO: this is internal version, not unversioned
				exports := unversioned.ExportOptions{}
				if err := scope.ParameterCodec.DecodeParameters(values, unversioned.GroupVersion{Version: "v1"}, &exports); err != nil {
					return nil, err
				}
				if exports.Export {
					if e == nil {
						return nil, errors.NewBadRequest(fmt.Sprintf("export of %q is not supported", scope.Resource.Resource))
					}
					return e.Export(ctx, name, exports)
				}
			}

			return r.Get(ctx, name)
		})
}

// GetResourceWithOptions returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResourceWithOptions(r rest.GetterWithOptions, scope RequestScope) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx api.Context, name string, req *restful.Request) (runtime.Object, error) {
			opts, subpath, subpathKey := r.NewGetOptions()
			if err := getRequestOptions(req, scope, opts, subpath, subpathKey); err != nil {
				return nil, err
			}
			return r.Get(ctx, name, opts)
		})
}

func getRequestOptions(req *restful.Request, scope RequestScope, into runtime.Object, subpath bool, subpathKey string) error {
	if into == nil {
		return nil
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
	return scope.ParameterCodec.DecodeParameters(query, scope.Kind.GroupVersion(), into)
}

// ConnectResource returns a function that handles a connect request on a rest.Storage object.
func ConnectResource(connecter rest.Connecter, scope RequestScope, admit admission.Interface, restPath string) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		w := res.ResponseWriter
		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)
		opts, subpath, subpathKey := connecter.NewConnectOptions()
		if err := getRequestOptions(req, scope, opts, subpath, subpathKey); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}
		handler, err := connecter.Connect(ctx, name, opts, &responder{scope: scope, req: req, res: res})
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		handler.ServeHTTP(w, req.Request)
	}
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	scope RequestScope
	req   *restful.Request
	res   *restful.Response
}

func (r *responder) Object(statusCode int, obj runtime.Object) {
	write(statusCode, r.scope.Kind.GroupVersion(), r.scope.Serializer, obj, r.res.ResponseWriter, r.req.Request)
}

func (r *responder) Error(err error) {
	r.scope.err(err, r.res.ResponseWriter, r.req.Request)
}

// ListResource returns a function that handles retrieving a list of resources from a rest.Storage object.
func ListResource(r rest.Lister, rw rest.Watcher, scope RequestScope, forceWatch bool, minRequestTimeout time.Duration) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := util.NewTrace("List " + req.Request.URL.Path)

		w := res.ResponseWriter

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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

		opts := api.ListOptions{}
		if err := scope.ParameterCodec.DecodeParameters(req.Request.URL.Query(), scope.Kind.GroupVersion(), &opts); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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
				scope.err(err, res.ResponseWriter, req.Request)
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
				scope.err(errors.NewBadRequest("both a name and a field selector provided; please provide one or the other."), res.ResponseWriter, req.Request)
				return
			}
			opts.FieldSelector = nameSelector
		}

		if (opts.Watch || forceWatch) && rw != nil {
			watcher, err := rw.Watch(ctx, &opts)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
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
			serveWatch(watcher, scope, req, res, timeout)
			return
		}

		// Log only long List requests (ignore Watch).
		defer trace.LogIfLong(500 * time.Millisecond)
		trace.Step("About to List from storage")
		result, err := r.List(ctx, &opts)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Listing from storage done")
		numberOfItems, err := setListSelfLink(result, req, scope.Namer)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Self-linking done")
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		gv := scope.Kind.GroupVersion()
		s, err := negotiateInputSerializer(req.Request, scope.Serializer)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		decoder := scope.Serializer.DecoderToVersion(s, unversioned.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal})

		body, err := readBody(req.Request)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk)
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		if gvk.GroupVersion() != gv {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%v)", gvk.GroupVersion().String(), gv.String()))
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Conversion done")

		if admit != nil && admit.Handles(admission.Create) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Create, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Object stored in database")

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Self-link added")

		write(http.StatusCreated, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		versionedObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion().String())
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		// TODO: handle this in negotiation
		contentType := req.HeaderParameter("Content-Type")
		// Remove "; charset=" if included in header.
		if idx := strings.Index(contentType, ";"); idx > 0 {
			contentType = contentType[:idx]
		}
		patchType := api.PatchType(contentType)

		patchJS, err := readBody(req.Request)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		s, ok := scope.Serializer.SerializerForMediaType("application/json", nil)
		if !ok {
			scope.err(fmt.Errorf("no serializer defined for JSON"), res.ResponseWriter, req.Request)
			return
		}
		gv := scope.Kind.GroupVersion()
		codec := runtime.NewCodec(
			scope.Serializer.EncoderForVersion(s, gv),
			scope.Serializer.DecoderToVersion(s, unversioned.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal}),
		)

		updateAdmit := func(updatedObject runtime.Object) error {
			if admit != nil && admit.Handles(admission.Update) {
				userInfo, _ := api.UserFrom(ctx)
				return admit.Admit(admission.NewAttributesRecord(updatedObject, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Update, userInfo))
			}

			return nil
		}

		result, err := patchResource(ctx, updateAdmit, timeout, versionedObj, r, name, patchType, patchJS, scope.Namer, codec)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
	}

}

type updateAdmissionFunc func(updatedObject runtime.Object) error

// patchResource divides PatchResource for easier unit testing
func patchResource(ctx api.Context, admit updateAdmissionFunc, timeout time.Duration, versionedObj runtime.Object, patcher rest.Patcher, name string, patchType api.PatchType, patchJS []byte, namer ScopeNamer, codec runtime.Codec) (runtime.Object, error) {
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
		if err := admit(objToUpdate); err != nil {
			return nil, err
		}

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

			currentPatch, err := strategicpatch.CreateStrategicMergePatch(originalObjJS, currentObjectJS, versionedObj)
			if err != nil {
				return nil, err
			}
			originalPatch, err := strategicpatch.CreateStrategicMergePatch(originalObjJS, originalPatchedObjJS, versionedObj)
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
				glog.V(4).Infof("patchResource failed for resource %s, becauase there is a meaningful conflict.\n diff1=%v\n, diff2=%v\n", name, diff1, diff2)
				return updateObject, updateErr
			}

			newlyPatchedObjJS, err := getPatchedJS(api.StrategicMergePatchType, currentObjectJS, originalPatch, versionedObj)
			if err != nil {
				return nil, err
			}
			if err := runtime.DecodeInto(codec, newlyPatchedObjJS, objToUpdate); err != nil {
				return nil, err
			}

			if err := admit(objToUpdate); err != nil {
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		s, err := negotiateInputSerializer(req.Request, scope.Serializer)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		obj, gvk, err := scope.Serializer.DecoderToVersion(s, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk)
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		if gvk.GroupVersion() != defaultGVK.GroupVersion() {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%s)", gvk.GroupVersion(), defaultGVK.GroupVersion()))
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Conversion done")

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		if admit != nil && admit.Handles(admission.Update) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Update, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Object stored in database")

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		trace.Step("Self-link added")

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}
		write(status, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		options := &api.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
			if len(body) > 0 {
				s, err := negotiateInputSerializer(req.Request, scope.Serializer)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				defaultGVK := scope.Kind.GroupVersion().WithKind("DeleteOptions")
				obj, _, err := scope.Serializer.DecoderToVersion(s, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), res.ResponseWriter, req.Request)
					return
				}
			}
		}

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, scope.Kind.GroupKind(), namespace, name, scope.Resource.GroupResource(), scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}

		trace.Step("About do delete object from database")
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.Delete(ctx, name, options)
		})
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
			}
		}
		write(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = api.WithNamespace(ctx, namespace)

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := api.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, scope.Kind.GroupKind(), namespace, "", scope.Resource.GroupResource(), scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}

		listOptions := api.ListOptions{}
		if err := scope.ParameterCodec.DecodeParameters(req.Request.URL.Query(), scope.Kind.GroupVersion(), &listOptions); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}

		options := &api.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
			if len(body) > 0 {
				s, err := negotiateInputSerializer(req.Request, scope.Serializer)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				defaultGVK := scope.Kind.GroupVersion().WithKind("DeleteOptions")
				obj, _, err := scope.Serializer.DecoderToVersion(s, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), res.ResponseWriter, req.Request)
					return
				}
			}
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.DeleteCollection(ctx, options, &listOptions)
		})
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
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
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
			}
		}
		writeNegotiated(scope.Serializer, scope.Kind.GroupVersion(), w, req.Request, http.StatusOK, result)
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
		defer utilruntime.HandleCrash(func(panicReason interface{}) {
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
func transformDecodeError(typer runtime.ObjectTyper, baseErr error, into runtime.Object, gvk *unversioned.GroupVersionKind) error {
	objGVK, err := typer.ObjectKind(into)
	if err != nil {
		return err
	}
	if gvk != nil && len(gvk.Kind) > 0 {
		return errors.NewBadRequest(fmt.Sprintf("%s in version %q cannot be handled as a %s: %v", gvk.Kind, gvk.Version, objGVK.Kind, baseErr))
	}
	return errors.NewBadRequest(fmt.Sprintf("the object provided is unrecognized (must be of type %s): %v", objGVK.Kind, baseErr))
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

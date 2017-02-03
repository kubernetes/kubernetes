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

package handlers

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/emicklei/go-restful"
	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/request"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/registry/rest"
)

// ContextFunc returns a Context given a request - a context must be returned
type ContextFunc func(req *restful.Request) request.Context

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
	// GenerateLink creates an encoded URI for a given runtime object that represents the canonical path
	// and query.
	GenerateLink(req *restful.Request, obj runtime.Object) (uri string, err error)
	// GenerateLink creates an encoded URI for a list that represents the canonical path and query.
	GenerateListLink(req *restful.Request) (uri string, err error)
}

// RequestScope encapsulates common fields across all RESTful handler methods.
type RequestScope struct {
	Namer ScopeNamer
	ContextFunc

	Serializer runtime.NegotiatedSerializer
	runtime.ParameterCodec

	Creater   runtime.ObjectCreater
	Convertor runtime.ObjectConvertor
	Copier    runtime.ObjectCopier

	Resource    schema.GroupVersionResource
	Kind        schema.GroupVersionKind
	Subresource string

	MetaGroupVersion schema.GroupVersion
}

func (scope *RequestScope) err(err error, w http.ResponseWriter, req *http.Request) {
	responsewriters.ErrorNegotiated(err, scope.Serializer, scope.Kind.GroupVersion(), w, req)
}

// getterFunc performs a get request with the given context and object name. The request
// may be used to deserialize an options object to pass to the getter.
type getterFunc func(ctx request.Context, name string, req *restful.Request) (runtime.Object, error)

// maxRetryWhenPatchConflicts is the maximum number of conflicts retry during a patch operation before returning failure
const maxRetryWhenPatchConflicts = 5

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
		ctx = request.WithNamespace(ctx, namespace)

		result, err := getter(ctx, name, req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		responsewriters.WriteObject(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
	}
}

// GetResource returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResource(r rest.Getter, e rest.Exporter, scope RequestScope) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx request.Context, name string, req *restful.Request) (runtime.Object, error) {
			// For performance tracking purposes.
			trace := utiltrace.New("Get " + req.Request.URL.Path)
			defer trace.LogIfLong(500 * time.Millisecond)

			// check for export
			options := metav1.GetOptions{}
			if values := req.Request.URL.Query(); len(values) > 0 {
				exports := metav1.ExportOptions{}
				if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, &exports); err != nil {
					return nil, err
				}
				if exports.Export {
					if e == nil {
						return nil, errors.NewBadRequest(fmt.Sprintf("export of %q is not supported", scope.Resource.Resource))
					}
					return e.Export(ctx, name, exports)
				}
				if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, &options); err != nil {
					return nil, err
				}
			}

			return r.Get(ctx, name, &options)
		})
}

// GetResourceWithOptions returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResourceWithOptions(r rest.GetterWithOptions, scope RequestScope) restful.RouteFunction {
	return getResourceHandler(scope,
		func(ctx request.Context, name string, req *restful.Request) (runtime.Object, error) {
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
		ctx = request.WithNamespace(ctx, namespace)
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
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(connectRequest, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Connect, userInfo))
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
	responsewriters.WriteObject(statusCode, r.scope.Kind.GroupVersion(), r.scope.Serializer, obj, r.res.ResponseWriter, r.req.Request)
}

func (r *responder) Error(err error) {
	r.scope.err(err, r.res.ResponseWriter, r.req.Request)
}

// ListResource returns a function that handles retrieving a list of resources from a rest.Storage object.
func ListResource(r rest.Lister, rw rest.Watcher, scope RequestScope, forceWatch bool, minRequestTimeout time.Duration) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := utiltrace.New("List " + req.Request.URL.Path)

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
		ctx = request.WithNamespace(ctx, namespace)

		opts := metainternalversion.ListOptions{}
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.Request.URL.Query(), scope.MetaGroupVersion, &opts); err != nil {
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
			// SelectionPredicate will notice that this is
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
			glog.Infof("Started to log from %v for %v", ctx, req.Request.URL.RequestURI())
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
		// Ensure empty lists return a non-nil items slice
		if numberOfItems == 0 && meta.IsListType(result) {
			if err := meta.SetList(result, []runtime.Object{}); err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}
		responsewriters.WriteObject(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
		trace.Step(fmt.Sprintf("Writing http response done (%d items)", numberOfItems))
	}
}

func createHandler(r rest.NamedCreater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface, includeName bool) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := utiltrace.New("Create " + req.Request.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

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
		ctx = request.WithNamespace(ctx, namespace)

		gv := scope.Kind.GroupVersion()
		s, err := negotiation.NegotiateInputSerializer(req.Request, scope.Serializer)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		decoder := scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal})

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
			err = transformDecodeError(typer, err, original, gvk, body)
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
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}

		trace.Step("About to store object in database")
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			out, err := r.Create(ctx, name, obj)
			if status, ok := out.(*metav1.Status); ok && err == nil && status.Code == 0 {
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

		responsewriters.WriteObject(http.StatusCreated, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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

func (c *namedCreaterAdapter) Create(ctx request.Context, name string, obj runtime.Object) (runtime.Object, error) {
	return c.Creater.Create(ctx, obj)
}

// PatchResource returns a function that will handle a resource patch
// TODO: Eventually PatchResource should just use GuaranteedUpdate and this routine should be a bit cleaner
func PatchResource(r rest.Patcher, scope RequestScope, admit admission.Interface, converter runtime.ObjectConvertor) restful.RouteFunction {
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
		ctx = request.WithNamespace(ctx, namespace)

		versionedObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion())
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
		patchType := types.PatchType(contentType)

		patchJS, err := readBody(req.Request)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		s, ok := runtime.SerializerInfoForMediaType(scope.Serializer.SupportedMediaTypes(), runtime.ContentTypeJSON)
		if !ok {
			scope.err(fmt.Errorf("no serializer defined for JSON"), res.ResponseWriter, req.Request)
			return
		}
		gv := scope.Kind.GroupVersion()
		codec := runtime.NewCodec(
			scope.Serializer.EncoderForVersion(s.Serializer, gv),
			scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal}),
		)

		updateAdmit := func(updatedObject runtime.Object, currentObject runtime.Object) error {
			if admit != nil && admit.Handles(admission.Update) {
				userInfo, _ := request.UserFrom(ctx)
				return admit.Admit(admission.NewAttributesRecord(updatedObject, currentObject, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo))
			}

			return nil
		}

		result, err := patchResource(ctx, updateAdmit, timeout, versionedObj, r, name, patchType, patchJS, scope.Namer, scope.Copier, scope.Resource, codec)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		if err := setSelfLink(result, req, scope.Namer); err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		responsewriters.WriteObject(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
	}

}

type updateAdmissionFunc func(updatedObject runtime.Object, currentObject runtime.Object) error

// patchResource divides PatchResource for easier unit testing
func patchResource(
	ctx request.Context,
	admit updateAdmissionFunc,
	timeout time.Duration,
	versionedObj runtime.Object,
	patcher rest.Patcher,
	name string,
	patchType types.PatchType,
	patchJS []byte,
	namer ScopeNamer,
	copier runtime.ObjectCopier,
	resource schema.GroupVersionResource,
	codec runtime.Codec,
) (runtime.Object, error) {

	namespace := request.NamespaceValue(ctx)

	var (
		originalObjJS        []byte
		originalPatchedObjJS []byte
		originalObjMap       map[string]interface{}
		originalPatchMap     map[string]interface{}
		lastConflictErr      error
	)

	// applyPatch is called every time GuaranteedUpdate asks for the updated object,
	// and is given the currently persisted object as input.
	applyPatch := func(_ request.Context, _, currentObject runtime.Object) (runtime.Object, error) {
		// Make sure we actually have a persisted currentObject
		if hasUID, err := hasUID(currentObject); err != nil {
			return nil, err
		} else if !hasUID {
			return nil, errors.NewNotFound(resource.GroupResource(), name)
		}

		switch {
		case originalObjJS == nil && originalObjMap == nil:
			// first time through,
			// 1. apply the patch
			// 2. save the original and patched to detect whether there were conflicting changes on retries

			objToUpdate := patcher.New()

			// For performance reasons, in case of strategicpatch, we avoid json
			// marshaling and unmarshaling and operate just on map[string]interface{}.
			// In case of other patch types, we still have to operate on JSON
			// representations.
			switch patchType {
			case types.JSONPatchType, types.MergePatchType:
				originalJS, patchedJS, err := patchObjectJSON(patchType, codec, currentObject, patchJS, objToUpdate, versionedObj)
				if err != nil {
					return nil, err
				}
				originalObjJS, originalPatchedObjJS = originalJS, patchedJS
			case types.StrategicMergePatchType:
				originalMap, patchMap, err := strategicPatchObject(codec, currentObject, patchJS, objToUpdate, versionedObj)
				if err != nil {
					return nil, err
				}
				originalObjMap, originalPatchMap = originalMap, patchMap
			}
			if err := checkName(objToUpdate, name, namespace, namer); err != nil {
				return nil, err
			}
			return objToUpdate, nil

		default:
			// on a conflict,
			// 1. build a strategic merge patch from originalJS and the patchedJS.  Different patch types can
			//    be specified, but a strategic merge patch should be expressive enough handle them.  Build the
			//    patch with this type to handle those cases.
			// 2. build a strategic merge patch from originalJS and the currentJS
			// 3. ensure no conflicts between the two patches
			// 4. apply the #1 patch to the currentJS object

			// TODO: This should be one-step conversion that doesn't require
			// json marshaling and unmarshaling once #39017 is fixed.
			data, err := runtime.Encode(codec, currentObject)
			if err != nil {
				return nil, err
			}
			currentObjMap := make(map[string]interface{})
			if err := json.Unmarshal(data, &currentObjMap); err != nil {
				return nil, err
			}

			var currentPatchMap map[string]interface{}
			if originalObjMap != nil {
				var err error
				currentPatchMap, err = strategicpatch.CreateTwoWayMergeMapPatch(originalObjMap, currentObjMap, versionedObj)
				if err != nil {
					return nil, err
				}
			} else {
				if originalPatchMap == nil {
					// Compute original patch, if we already didn't do this in previous retries.
					originalPatch, err := strategicpatch.CreateTwoWayMergePatch(originalObjJS, originalPatchedObjJS, versionedObj)
					if err != nil {
						return nil, err
					}
					originalPatchMap = make(map[string]interface{})
					if err := json.Unmarshal(originalPatch, &originalPatchMap); err != nil {
						return nil, err
					}
				}
				// Compute current patch.
				currentObjJS, err := runtime.Encode(codec, currentObject)
				if err != nil {
					return nil, err
				}
				currentPatch, err := strategicpatch.CreateTwoWayMergePatch(originalObjJS, currentObjJS, versionedObj)
				if err != nil {
					return nil, err
				}
				currentPatchMap = make(map[string]interface{})
				if err := json.Unmarshal(currentPatch, &currentPatchMap); err != nil {
					return nil, err
				}
			}

			hasConflicts, err := strategicpatch.HasConflicts(originalPatchMap, currentPatchMap)
			if err != nil {
				return nil, err
			}
			if hasConflicts {
				diff1, _ := json.Marshal(currentPatchMap)
				diff2, _ := json.Marshal(originalPatchMap)
				patchDiffErr := fmt.Errorf("there is a meaningful conflict:\n diff1=%v\n, diff2=%v\n", diff1, diff2)
				glog.V(4).Infof("patchResource failed for resource %s, because there is a meaningful conflict.\n diff1=%v\n, diff2=%v\n", name, diff1, diff2)

				// Return the last conflict error we got if we have one
				if lastConflictErr != nil {
					return nil, lastConflictErr
				}
				// Otherwise manufacture one of our own
				return nil, errors.NewConflict(resource.GroupResource(), name, patchDiffErr)
			}

			objToUpdate := patcher.New()
			if err := applyPatchToObject(codec, currentObjMap, originalPatchMap, objToUpdate, versionedObj); err != nil {
				return nil, err
			}

			return objToUpdate, nil
		}
	}

	// applyAdmission is called every time GuaranteedUpdate asks for the updated object,
	// and is given the currently persisted object and the patched object as input.
	applyAdmission := func(ctx request.Context, patchedObject runtime.Object, currentObject runtime.Object) (runtime.Object, error) {
		return patchedObject, admit(patchedObject, currentObject)
	}

	updatedObjectInfo := rest.DefaultUpdatedObjectInfo(nil, copier, applyPatch, applyAdmission)

	return finishRequest(timeout, func() (runtime.Object, error) {
		updateObject, _, updateErr := patcher.Update(ctx, name, updatedObjectInfo)
		for i := 0; i < maxRetryWhenPatchConflicts && (errors.IsConflict(updateErr)); i++ {
			lastConflictErr = updateErr
			updateObject, _, updateErr = patcher.Update(ctx, name, updatedObjectInfo)
		}
		return updateObject, updateErr
	})
}

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := utiltrace.New("Update " + req.Request.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		body, err := readBody(req.Request)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}

		s, err := negotiation.NegotiateInputSerializer(req.Request, scope.Serializer)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		obj, gvk, err := scope.Serializer.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk, body)
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

		var transformers []rest.TransformFunc
		if admit != nil && admit.Handles(admission.Update) {
			transformers = append(transformers, func(ctx request.Context, newObj, oldObj runtime.Object) (runtime.Object, error) {
				userInfo, _ := request.UserFrom(ctx)
				return newObj, admit.Admit(admission.NewAttributesRecord(newObj, oldObj, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Update, userInfo))
			})
		}

		trace.Step("About to store object in database")
		wasCreated := false
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, created, err := r.Update(ctx, name, rest.DefaultUpdatedObjectInfo(obj, scope.Copier, transformers...))
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
		responsewriters.WriteObject(status, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
	}
}

// DeleteResource returns a function that will handle a resource deletion
func DeleteResource(r rest.GracefulDeleter, allowsOptions bool, scope RequestScope, admit admission.Interface) restful.RouteFunction {
	return func(req *restful.Request, res *restful.Response) {
		// For performance tracking purposes.
		trace := utiltrace.New("Delete " + req.Request.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		w := res.ResponseWriter

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.Request.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, res.ResponseWriter, req.Request)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		options := &metav1.DeleteOptions{}
		if allowsOptions {
			body, err := readBody(req.Request)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req.Request, metainternalversion.Codecs)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				// For backwards compatibility, we need to allow existing clients to submit per group DeleteOptions
				// It is also allowed to pass a body with meta.k8s.io/v1.DeleteOptions
				defaultGVK := scope.MetaGroupVersion.WithKind("DeleteOptions")
				obj, _, err := metainternalversion.Codecs.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), res.ResponseWriter, req.Request)
					return
				}
			} else {
				if values := req.Request.URL.Query(); len(values) > 0 {
					if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, options); err != nil {
						scope.err(err, res.ResponseWriter, req.Request)
						return
					}
				}
			}
		}

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Delete, userInfo))
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
			result = &metav1.Status{
				Status: metav1.StatusSuccess,
				Code:   http.StatusOK,
				Details: &metav1.StatusDetails{
					Name: name,
					Kind: scope.Kind.Kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*metav1.Status); !ok {
				if err := setSelfLink(result, req, scope.Namer); err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
			}
		}
		responsewriters.WriteObject(http.StatusOK, scope.Kind.GroupVersion(), scope.Serializer, result, w, req.Request)
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
		ctx = request.WithNamespace(ctx, namespace)

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, "", scope.Resource, scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
		}

		listOptions := metainternalversion.ListOptions{}
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.Request.URL.Query(), scope.MetaGroupVersion, &listOptions); err != nil {
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

		options := &metav1.DeleteOptions{}
		if checkBody {
			body, err := readBody(req.Request)
			if err != nil {
				scope.err(err, res.ResponseWriter, req.Request)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req.Request, scope.Serializer)
				if err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
				defaultGVK := scope.Kind.GroupVersion().WithKind("DeleteOptions")
				obj, _, err := scope.Serializer.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
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
			result = &metav1.Status{
				Status: metav1.StatusSuccess,
				Code:   http.StatusOK,
				Details: &metav1.StatusDetails{
					Kind: scope.Kind.Kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			if _, ok := result.(*metav1.Status); !ok {
				if _, err := setListSelfLink(result, req, scope.Namer); err != nil {
					scope.err(err, res.ResponseWriter, req.Request)
					return
				}
			}
		}
		responsewriters.WriteObjectNegotiated(scope.Serializer, scope.Kind.GroupVersion(), w, req.Request, http.StatusOK, result)
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
		if status, ok := result.(*metav1.Status); ok {
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
func transformDecodeError(typer runtime.ObjectTyper, baseErr error, into runtime.Object, gvk *schema.GroupVersionKind, body []byte) error {
	objGVKs, _, err := typer.ObjectKinds(into)
	if err != nil {
		return err
	}
	objGVK := objGVKs[0]
	if gvk != nil && len(gvk.Kind) > 0 {
		return errors.NewBadRequest(fmt.Sprintf("%s in version %q cannot be handled as a %s: %v", gvk.Kind, gvk.Version, objGVK.Kind, baseErr))
	}
	summary := summarizeData(body, 30)
	return errors.NewBadRequest(fmt.Sprintf("the object provided is unrecognized (must be of type %s): %v (%s)", objGVK.Kind, baseErr, summary))
}

// setSelfLink sets the self link of an object (or the child items in a list) to the base URL of the request
// plus the path and query generated by the provided linkFunc
func setSelfLink(obj runtime.Object, req *restful.Request, namer ScopeNamer) error {
	// TODO: SelfLink generation should return a full URL?
	uri, err := namer.GenerateLink(req, obj)
	if err != nil {
		return nil
	}

	return namer.SetSelfLink(obj, uri)
}

func hasUID(obj runtime.Object) (bool, error) {
	if obj == nil {
		return false, nil
	}
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return false, errors.NewInternalError(err)
	}
	if len(accessor.GetUID()) == 0 {
		return false, nil
	}
	return true, nil
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

	uri, err := namer.GenerateListLink(req)
	if err != nil {
		return 0, err
	}
	if err := namer.SetSelfLink(obj, uri); err != nil {
		glog.V(4).Infof("Unable to set self link on object: %v", err)
	}

	count := 0
	err = meta.EachListItem(obj, func(obj runtime.Object) error {
		count++
		return setSelfLink(obj, req, namer)
	})
	return count, err
}

func summarizeData(data []byte, maxLength int) string {
	switch {
	case len(data) == 0:
		return "<empty>"
	case data[0] == '{':
		if len(data) > maxLength {
			return string(data[:maxLength]) + " ..."
		}
		return string(data)
	default:
		if len(data) > maxLength {
			return hex.EncodeToString(data[:maxLength]) + " ..."
		}
		return hex.EncodeToString(data)
	}
}

func readBody(req *http.Request) ([]byte, error) {
	defer req.Body.Close()
	return ioutil.ReadAll(req.Body)
}

func parseTimeout(str string) time.Duration {
	if str != "" {
		timeout, err := time.ParseDuration(str)
		if err == nil {
			return timeout
		}
		glog.Errorf("Failed to parse %q: %v", str, err)
	}
	return 30 * time.Second
}

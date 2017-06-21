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
	"fmt"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"strings"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1alpha1 "k8s.io/apimachinery/pkg/apis/meta/v1alpha1"
	"k8s.io/apimachinery/pkg/conversion/unstructured"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/mergepatch"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/endpoints/handlers/negotiation"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	utiltrace "k8s.io/apiserver/pkg/util/trace"
)

// RequestScope encapsulates common fields across all RESTful handler methods.
type RequestScope struct {
	Namer ScopeNamer
	ContextFunc

	Serializer runtime.NegotiatedSerializer
	runtime.ParameterCodec

	Creater         runtime.ObjectCreater
	Convertor       runtime.ObjectConvertor
	Defaulter       runtime.ObjectDefaulter
	Copier          runtime.ObjectCopier
	Typer           runtime.ObjectTyper
	UnsafeConvertor runtime.ObjectConvertor

	TableConvertor rest.TableConvertor

	Resource    schema.GroupVersionResource
	Kind        schema.GroupVersionKind
	Subresource string

	MetaGroupVersion schema.GroupVersion
}

func (scope *RequestScope) err(err error, w http.ResponseWriter, req *http.Request) {
	ctx := scope.ContextFunc(req)
	responsewriters.ErrorNegotiated(ctx, err, scope.Serializer, scope.Kind.GroupVersion(), w, req)
}

func (scope *RequestScope) AllowsConversion(gvk schema.GroupVersionKind) bool {
	// TODO: this is temporary, replace with an abstraction calculated at endpoint installation time
	if gvk.GroupVersion() == metav1alpha1.SchemeGroupVersion {
		switch gvk.Kind {
		case "Table":
			return scope.TableConvertor != nil
		case "PartialObjectMetadata", "PartialObjectMetadataList":
			// TODO: should delineate between lists and non-list endpoints
			return true
		default:
			return false
		}
	}
	return false
}

func (scope *RequestScope) AllowsServerVersion(version string) bool {
	return version == scope.MetaGroupVersion.Version
}

func (scope *RequestScope) AllowsStreamSchema(s string) bool {
	return s == "watch"
}

// getterFunc performs a get request with the given context and object name. The request
// may be used to deserialize an options object to pass to the getter.
type getterFunc func(ctx request.Context, name string, req *http.Request, trace *utiltrace.Trace) (runtime.Object, error)

// MaxRetryWhenPatchConflicts is the maximum number of conflicts retry during a patch operation before returning failure
const MaxRetryWhenPatchConflicts = 5

// getResourceHandler is an HTTP handler function for get requests. It delegates to the
// passed-in getterFunc to perform the actual get.
func getResourceHandler(scope RequestScope, getter getterFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		trace := utiltrace.New("Get " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		result, err := getter(ctx, name, req, trace)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		trace.Step("About to write a response")
		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
	}
}

// GetResource returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResource(r rest.Getter, e rest.Exporter, scope RequestScope) http.HandlerFunc {
	return getResourceHandler(scope,
		func(ctx request.Context, name string, req *http.Request, trace *utiltrace.Trace) (runtime.Object, error) {
			// check for export
			options := metav1.GetOptions{}
			if values := req.URL.Query(); len(values) > 0 {
				exports := metav1.ExportOptions{}
				if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, &exports); err != nil {
					err = errors.NewBadRequest(err.Error())
					return nil, err
				}
				if exports.Export {
					if e == nil {
						return nil, errors.NewBadRequest(fmt.Sprintf("export of %q is not supported", scope.Resource.Resource))
					}
					return e.Export(ctx, name, exports)
				}
				if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, &options); err != nil {
					err = errors.NewBadRequest(err.Error())
					return nil, err
				}
			}
			if trace != nil {
				trace.Step("About to Get from storage")
			}
			return r.Get(ctx, name, &options)
		})
}

// GetResourceWithOptions returns a function that handles retrieving a single resource from a rest.Storage object.
func GetResourceWithOptions(r rest.GetterWithOptions, scope RequestScope, isSubresource bool) http.HandlerFunc {
	return getResourceHandler(scope,
		func(ctx request.Context, name string, req *http.Request, trace *utiltrace.Trace) (runtime.Object, error) {
			opts, subpath, subpathKey := r.NewGetOptions()
			trace.Step("About to process Get options")
			if err := getRequestOptions(req, scope, opts, subpath, subpathKey, isSubresource); err != nil {
				err = errors.NewBadRequest(err.Error())
				return nil, err
			}
			if trace != nil {
				trace.Step("About to Get from storage")
			}
			return r.Get(ctx, name, opts)
		})
}

// getRequestOptions parses out options and can include path information.  The path information shouldn't include the subresource.
func getRequestOptions(req *http.Request, scope RequestScope, into runtime.Object, subpath bool, subpathKey string, isSubresource bool) error {
	if into == nil {
		return nil
	}

	query := req.URL.Query()
	if subpath {
		newQuery := make(url.Values)
		for k, v := range query {
			newQuery[k] = v
		}

		ctx := scope.ContextFunc(req)
		requestInfo, _ := request.RequestInfoFrom(ctx)
		startingIndex := 2
		if isSubresource {
			startingIndex = 3
		}
		newQuery[subpathKey] = []string{strings.Join(requestInfo.Parts[startingIndex:], "/")}
		query = newQuery
	}
	return scope.ParameterCodec.DecodeParameters(query, scope.Kind.GroupVersion(), into)
}

// ConnectResource returns a function that handles a connect request on a rest.Storage object.
func ConnectResource(connecter rest.Connecter, scope RequestScope, admit admission.Interface, restPath string, isSubresource bool) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)
		opts, subpath, subpathKey := connecter.NewConnectOptions()
		if err := getRequestOptions(req, scope, opts, subpath, subpathKey, isSubresource); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
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
				scope.err(err, w, req)
				return
			}
		}
		handler, err := connecter.Connect(ctx, name, opts, &responder{scope: scope, req: req, w: w})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		handler.ServeHTTP(w, req)
	}
}

// responder implements rest.Responder for assisting a connector in writing objects or errors.
type responder struct {
	scope RequestScope
	req   *http.Request
	w     http.ResponseWriter
}

func (r *responder) Object(statusCode int, obj runtime.Object) {
	ctx := r.scope.ContextFunc(r.req)
	responsewriters.WriteObject(ctx, statusCode, r.scope.Kind.GroupVersion(), r.scope.Serializer, obj, r.w, r.req)
}

func (r *responder) Error(err error) {
	r.scope.err(err, r.w, r.req)
}

func ListResource(r rest.Lister, rw rest.Watcher, scope RequestScope, forceWatch bool, minRequestTimeout time.Duration) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("List " + req.URL.Path)

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			scope.err(err, w, req)
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
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, &opts); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
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
				scope.err(err, w, req)
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
				scope.err(errors.NewBadRequest("both a name and a field selector provided; please provide one or the other."), w, req)
				return
			}
			opts.FieldSelector = nameSelector
		}

		if opts.Watch || forceWatch {
			if rw == nil {
				scope.err(errors.NewMethodNotSupported(scope.Resource.GroupResource(), "watch"), w, req)
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
			glog.V(2).Infof("Starting watch for %s, rv=%s labels=%s fields=%s timeout=%s", req.URL.Path, opts.ResourceVersion, opts.LabelSelector, opts.FieldSelector, timeout)

			watcher, err := rw.Watch(ctx, &opts)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			serveWatch(watcher, scope, req, w, timeout)
			return
		}

		// Log only long List requests (ignore Watch).
		defer trace.LogIfLong(500 * time.Millisecond)
		trace.Step("About to List from storage")
		result, err := r.List(ctx, &opts)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Listing from storage done")
		numberOfItems, err := setListSelfLink(result, ctx, req, scope.Namer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Self-linking done")
		// Ensure empty lists return a non-nil items slice
		if numberOfItems == 0 && meta.IsListType(result) {
			if err := meta.SetList(result, []runtime.Object{}); err != nil {
				scope.err(err, w, req)
				return
			}
		}

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
		trace.Step(fmt.Sprintf("Writing http response done (%d items)", numberOfItems))
	}
}

func createHandler(r rest.NamedCreater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface, includeName bool) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Create " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

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
			scope.err(err, w, req)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		gv := scope.Kind.GroupVersion()
		s, err := negotiation.NegotiateInputSerializer(req, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		decoder := scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: gv.Group, Version: runtime.APIVersionInternal})

		body, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk, body)
			scope.err(err, w, req)
			return
		}
		if gvk.GroupVersion() != gv {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%v)", gvk.GroupVersion().String(), gv.String()))
			scope.err(err, w, req)
			return
		}
		trace.Step("Conversion done")

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)

		if admit != nil && admit.Handles(admission.Create) {
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(obj, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Create, userInfo))
			if err != nil {
				scope.err(err, w, req)
				return
			}
		}

		// TODO: replace with content type negotiation?
		includeUninitialized := req.URL.Query().Get("includeUninitialized") == "1"

		trace.Step("About to store object in database")
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.Create(ctx, name, obj, includeUninitialized)
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object stored in database")

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Self-link added")

		// If the object is partially initialized, always indicate it via StatusAccepted
		code := http.StatusCreated
		if accessor, err := meta.Accessor(result); err == nil {
			if accessor.GetInitializers() != nil {
				code = http.StatusAccepted
			}
		}
		status, ok := result.(*metav1.Status)
		if ok && err == nil && status.Code == 0 {
			status.Code = int32(code)
		}

		transformResponseObject(ctx, scope, req, w, code, result)
	}
}

// CreateNamedResource returns a function that will handle a resource creation with name.
func CreateNamedResource(r rest.NamedCreater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) http.HandlerFunc {
	return createHandler(r, scope, typer, admit, true)
}

// CreateResource returns a function that will handle a resource creation.
func CreateResource(r rest.Creater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) http.HandlerFunc {
	return createHandler(&namedCreaterAdapter{r}, scope, typer, admit, false)
}

type namedCreaterAdapter struct {
	rest.Creater
}

func (c *namedCreaterAdapter) Create(ctx request.Context, name string, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	return c.Creater.Create(ctx, obj, includeUninitialized)
}

// PatchResource returns a function that will handle a resource patch
// TODO: Eventually PatchResource should just use GuaranteedUpdate and this routine should be a bit cleaner
func PatchResource(r rest.Patcher, scope RequestScope, admit admission.Interface, converter runtime.ObjectConvertor) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// TODO: we either want to remove timeout or document it (if we
		// document, move timeout out of this function and declare it in
		// api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		versionedObj, err := converter.ConvertToVersion(r.New(), scope.Kind.GroupVersion())
		if err != nil {
			scope.err(err, w, req)
			return
		}

		// TODO: handle this in negotiation
		contentType := req.Header.Get("Content-Type")
		// Remove "; charset=" if included in header.
		if idx := strings.Index(contentType, ";"); idx > 0 {
			contentType = contentType[:idx]
		}
		patchType := types.PatchType(contentType)

		patchJS, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestPatch(ae, patchJS)

		s, ok := runtime.SerializerInfoForMediaType(scope.Serializer.SupportedMediaTypes(), runtime.ContentTypeJSON)
		if !ok {
			scope.err(fmt.Errorf("no serializer defined for JSON"), w, req)
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

		result, err := patchResource(ctx, updateAdmit, timeout, versionedObj, r, name, patchType, patchJS,
			scope.Namer, scope.Copier, scope.Creater, scope.Defaulter, scope.UnsafeConvertor, scope.Kind, scope.Resource, codec)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
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
	creater runtime.ObjectCreater,
	defaulter runtime.ObjectDefaulter,
	unsafeConvertor runtime.ObjectConvertor,
	kind schema.GroupVersionKind,
	resource schema.GroupVersionResource,
	codec runtime.Codec,
) (runtime.Object, error) {

	namespace := request.NamespaceValue(ctx)

	var (
		originalObjJS           []byte
		originalPatchedObjJS    []byte
		originalObjMap          map[string]interface{}
		getOriginalPatchMap     func() (map[string]interface{}, error)
		lastConflictErr         error
		originalResourceVersion string
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

		currentResourceVersion := ""
		if currentMetadata, err := meta.Accessor(currentObject); err == nil {
			currentResourceVersion = currentMetadata.GetResourceVersion()
		}

		switch {
		case originalObjJS == nil && originalObjMap == nil:
			// first time through,
			// 1. apply the patch
			// 2. save the original and patched to detect whether there were conflicting changes on retries

			originalResourceVersion = currentResourceVersion
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

				// Make a getter that can return a fresh strategic patch map if needed for conflict retries
				// We have to rebuild it each time we need it, because the map gets mutated when being applied
				var originalPatchBytes []byte
				getOriginalPatchMap = func() (map[string]interface{}, error) {
					if originalPatchBytes == nil {
						// Compute once
						originalPatchBytes, err = strategicpatch.CreateTwoWayMergePatch(originalObjJS, originalPatchedObjJS, versionedObj)
						if err != nil {
							return nil, err
						}
					}
					// Return a fresh map every time
					originalPatchMap := make(map[string]interface{})
					if err := json.Unmarshal(originalPatchBytes, &originalPatchMap); err != nil {
						return nil, err
					}
					return originalPatchMap, nil
				}

			case types.StrategicMergePatchType:
				// Since the patch is applied on versioned objects, we need to convert the
				// current object to versioned representation first.
				currentVersionedObject, err := unsafeConvertor.ConvertToVersion(currentObject, kind.GroupVersion())
				if err != nil {
					return nil, err
				}
				versionedObjToUpdate, err := creater.New(kind)
				if err != nil {
					return nil, err
				}
				// Capture the original object map and patch for possible retries.
				originalMap := make(map[string]interface{})
				if err := unstructured.DefaultConverter.ToUnstructured(currentVersionedObject, &originalMap); err != nil {
					return nil, err
				}
				if err := strategicPatchObject(codec, defaulter, currentVersionedObject, patchJS, versionedObjToUpdate, versionedObj); err != nil {
					return nil, err
				}
				// Convert the object back to unversioned.
				gvk := kind.GroupKind().WithVersion(runtime.APIVersionInternal)
				unversionedObjToUpdate, err := unsafeConvertor.ConvertToVersion(versionedObjToUpdate, gvk.GroupVersion())
				if err != nil {
					return nil, err
				}
				objToUpdate = unversionedObjToUpdate
				// Store unstructured representation for possible retries.
				originalObjMap = originalMap
				// Make a getter that can return a fresh strategic patch map if needed for conflict retries
				// We have to rebuild it each time we need it, because the map gets mutated when being applied
				getOriginalPatchMap = func() (map[string]interface{}, error) {
					patchMap := make(map[string]interface{})
					if err := json.Unmarshal(patchJS, &patchMap); err != nil {
						return nil, err
					}
					return patchMap, nil
				}
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

			currentObjMap := make(map[string]interface{})

			// Since the patch is applied on versioned objects, we need to convert the
			// current object to versioned representation first.
			currentVersionedObject, err := unsafeConvertor.ConvertToVersion(currentObject, kind.GroupVersion())
			if err != nil {
				return nil, err
			}
			if err := unstructured.DefaultConverter.ToUnstructured(currentVersionedObject, &currentObjMap); err != nil {
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

			// Get a fresh copy of the original strategic patch each time through, since applying it mutates the map
			originalPatchMap, err := getOriginalPatchMap()
			if err != nil {
				return nil, err
			}

			hasConflicts, err := mergepatch.HasConflicts(originalPatchMap, currentPatchMap)
			if err != nil {
				return nil, err
			}

			if hasConflicts {
				diff1, _ := json.Marshal(currentPatchMap)
				diff2, _ := json.Marshal(originalPatchMap)
				patchDiffErr := fmt.Errorf("there is a meaningful conflict (firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))
				glog.V(4).Infof("patchResource failed for resource %s, because there is a meaningful conflict(firstResourceVersion: %q, currentResourceVersion: %q):\n diff1=%v\n, diff2=%v\n", name, originalResourceVersion, currentResourceVersion, string(diff1), string(diff2))

				// Return the last conflict error we got if we have one
				if lastConflictErr != nil {
					return nil, lastConflictErr
				}
				// Otherwise manufacture one of our own
				return nil, errors.NewConflict(resource.GroupResource(), name, patchDiffErr)
			}

			versionedObjToUpdate, err := creater.New(kind)
			if err != nil {
				return nil, err
			}
			if err := applyPatchToObject(codec, defaulter, currentObjMap, originalPatchMap, versionedObjToUpdate, versionedObj); err != nil {
				return nil, err
			}
			// Convert the object back to unversioned.
			gvk := kind.GroupKind().WithVersion(runtime.APIVersionInternal)
			objToUpdate, err := unsafeConvertor.ConvertToVersion(versionedObjToUpdate, gvk.GroupVersion())
			if err != nil {
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
		for i := 0; i < MaxRetryWhenPatchConflicts && (errors.IsConflict(updateErr)); i++ {
			lastConflictErr = updateErr
			updateObject, _, updateErr = patcher.Update(ctx, name, updatedObjectInfo)
		}
		return updateObject, updateErr
	})
}

// UpdateResource returns a function that will handle a resource update
func UpdateResource(r rest.Updater, scope RequestScope, typer runtime.ObjectTyper, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Update " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		body, err := readBody(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		s, err := negotiation.NegotiateInputSerializer(req, scope.Serializer)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		defaultGVK := scope.Kind
		original := r.New()
		trace.Step("About to convert to expected version")
		decoder := scope.Serializer.DecoderToVersion(s.Serializer, schema.GroupVersion{Group: defaultGVK.Group, Version: runtime.APIVersionInternal})
		obj, gvk, err := decoder.Decode(body, &defaultGVK, original)
		if err != nil {
			err = transformDecodeError(typer, err, original, gvk, body)
			scope.err(err, w, req)
			return
		}
		if gvk.GroupVersion() != defaultGVK.GroupVersion() {
			err = errors.NewBadRequest(fmt.Sprintf("the API version in the data (%s) does not match the expected API version (%s)", gvk.GroupVersion(), defaultGVK.GroupVersion()))
			scope.err(err, w, req)
			return
		}
		trace.Step("Conversion done")

		ae := request.AuditEventFrom(ctx)
		audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)

		if err := checkName(obj, name, namespace, scope.Namer); err != nil {
			scope.err(err, w, req)
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
			scope.err(err, w, req)
			return
		}
		trace.Step("Object stored in database")

		requestInfo, ok := request.RequestInfoFrom(ctx)
		if !ok {
			scope.err(fmt.Errorf("missing requestInfo"), w, req)
			return
		}
		if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Self-link added")

		status := http.StatusOK
		if wasCreated {
			status = http.StatusCreated
		}

		transformResponseObject(ctx, scope, req, w, status, result)
	}
}

// DeleteResource returns a function that will handle a resource deletion
func DeleteResource(r rest.GracefulDeleter, allowsOptions bool, scope RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// For performance tracking purposes.
		trace := utiltrace.New("Delete " + req.URL.Path)
		defer trace.LogIfLong(500 * time.Millisecond)

		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, name, err := scope.Namer.Name(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}
		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		options := &metav1.DeleteOptions{}
		if allowsOptions {
			body, err := readBody(req)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, metainternalversion.Codecs)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				// For backwards compatibility, we need to allow existing clients to submit per group DeleteOptions
				// It is also allowed to pass a body with meta.k8s.io/v1.DeleteOptions
				defaultGVK := scope.MetaGroupVersion.WithKind("DeleteOptions")
				obj, _, err := metainternalversion.Codecs.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}

				ae := request.AuditEventFrom(ctx)
				audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)
			} else {
				if values := req.URL.Query(); len(values) > 0 {
					if err := metainternalversion.ParameterCodec.DecodeParameters(values, scope.MetaGroupVersion, options); err != nil {
						err = errors.NewBadRequest(err.Error())
						scope.err(err, w, req)
						return
					}
				}
			}
		}

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, name, scope.Resource, scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				scope.err(err, w, req)
				return
			}
		}

		trace.Step("About do delete object from database")
		wasDeleted := true
		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			obj, deleted, err := r.Delete(ctx, name, options)
			wasDeleted = deleted
			return obj, err
		})
		if err != nil {
			scope.err(err, w, req)
			return
		}
		trace.Step("Object deleted from database")

		status := http.StatusOK
		// Return http.StatusAccepted if the resource was not deleted immediately and
		// user requested cascading deletion by setting OrphanDependents=false.
		// Note: We want to do this always if resource was not deleted immediately, but
		// that will break existing clients.
		// Other cases where resource is not instantly deleted are: namespace deletion
		// and pod graceful deletion.
		if !wasDeleted && options.OrphanDependents != nil && *options.OrphanDependents == false {
			status = http.StatusAccepted
		}
		// if the rest.Deleter returns a nil object, fill out a status. Callers may return a valid
		// object with the response.
		if result == nil {
			result = &metav1.Status{
				Status: metav1.StatusSuccess,
				Code:   int32(status),
				Details: &metav1.StatusDetails{
					Name: name,
					Kind: scope.Kind.Kind,
				},
			}
		} else {
			// when a non-status response is returned, set the self link
			requestInfo, ok := request.RequestInfoFrom(ctx)
			if !ok {
				scope.err(fmt.Errorf("missing requestInfo"), w, req)
				return
			}
			if _, ok := result.(*metav1.Status); !ok {
				if err := setSelfLink(result, requestInfo, scope.Namer); err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		transformResponseObject(ctx, scope, req, w, status, result)
	}
}

// DeleteCollection returns a function that will handle a collection deletion
func DeleteCollection(r rest.CollectionDeleter, checkBody bool, scope RequestScope, admit admission.Interface) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		// TODO: we either want to remove timeout or document it (if we document, move timeout out of this function and declare it in api_installer)
		timeout := parseTimeout(req.URL.Query().Get("timeout"))

		namespace, err := scope.Namer.Namespace(req)
		if err != nil {
			scope.err(err, w, req)
			return
		}

		ctx := scope.ContextFunc(req)
		ctx = request.WithNamespace(ctx, namespace)

		if admit != nil && admit.Handles(admission.Delete) {
			userInfo, _ := request.UserFrom(ctx)

			err = admit.Admit(admission.NewAttributesRecord(nil, nil, scope.Kind, namespace, "", scope.Resource, scope.Subresource, admission.Delete, userInfo))
			if err != nil {
				scope.err(err, w, req)
				return
			}
		}

		listOptions := metainternalversion.ListOptions{}
		if err := metainternalversion.ParameterCodec.DecodeParameters(req.URL.Query(), scope.MetaGroupVersion, &listOptions); err != nil {
			err = errors.NewBadRequest(err.Error())
			scope.err(err, w, req)
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
				scope.err(err, w, req)
				return
			}
		}

		options := &metav1.DeleteOptions{}
		if checkBody {
			body, err := readBody(req)
			if err != nil {
				scope.err(err, w, req)
				return
			}
			if len(body) > 0 {
				s, err := negotiation.NegotiateInputSerializer(req, scope.Serializer)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				defaultGVK := scope.Kind.GroupVersion().WithKind("DeleteOptions")
				obj, _, err := scope.Serializer.DecoderToVersion(s.Serializer, defaultGVK.GroupVersion()).Decode(body, &defaultGVK, options)
				if err != nil {
					scope.err(err, w, req)
					return
				}
				if obj != options {
					scope.err(fmt.Errorf("decoded object cannot be converted to DeleteOptions"), w, req)
					return
				}

				ae := request.AuditEventFrom(ctx)
				audit.LogRequestObject(ae, obj, scope.Resource, scope.Subresource, scope.Serializer)
			}
		}

		result, err := finishRequest(timeout, func() (runtime.Object, error) {
			return r.DeleteCollection(ctx, options, &listOptions)
		})
		if err != nil {
			scope.err(err, w, req)
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
				if _, err := setListSelfLink(result, ctx, req, scope.Namer); err != nil {
					scope.err(err, w, req)
					return
				}
			}
		}

		transformResponseObject(ctx, scope, req, w, http.StatusOK, result)
	}
}

// resultFunc is a function that returns a rest result and can be run in a goroutine
type resultFunc func() (runtime.Object, error)

// finishRequest makes a given resultFunc asynchronous and handles errors returned by the response.
// An api.Status object with status != success is considered an "error", which interrupts the normal response flow.
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
			if status.Status != metav1.StatusSuccess {
				return nil, errors.FromObject(status)
			}
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
func setSelfLink(obj runtime.Object, requestInfo *request.RequestInfo, namer ScopeNamer) error {
	// TODO: SelfLink generation should return a full URL?
	uri, err := namer.GenerateLink(requestInfo, obj)
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
	objNamespace, objName, err := namer.ObjectName(obj)
	if err != nil {
		return errors.NewBadRequest(fmt.Sprintf(
			"the name of the object (%s based on URL) was undeterminable: %v", name, err))
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

	return nil
}

// setListSelfLink sets the self link of a list to the base URL, then sets the self links
// on all child objects returned. Returns the number of items in the list.
func setListSelfLink(obj runtime.Object, ctx request.Context, req *http.Request, namer ScopeNamer) (int, error) {
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
	requestInfo, ok := request.RequestInfoFrom(ctx)
	if !ok {
		return 0, fmt.Errorf("missing requestInfo")
	}

	count := 0
	err = meta.EachListItem(obj, func(obj runtime.Object) error {
		count++
		return setSelfLink(obj, requestInfo, namer)
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

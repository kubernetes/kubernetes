/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"fmt"
	"io"
	"net/http"
	"path"
	"sync"
	"sync/atomic"
	"time"

	openapispec "github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
	"github.com/golang/glog"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/discovery"
	cache "k8s.io/client-go/tools/cache"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	"k8s.io/apiextensions-apiserver/pkg/controller/finalizer"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
)

// crdHandler serves the `/apis` endpoint.
// This is registered as a filter so that it never collides with any explicitly registered endpoints
type crdHandler struct {
	versionDiscoveryHandler *versionDiscoveryHandler
	groupDiscoveryHandler   *groupDiscoveryHandler

	customStorageLock sync.Mutex
	// customStorage contains a crdStorageMap
	customStorage atomic.Value

	requestContextMapper apirequest.RequestContextMapper

	crdLister listers.CustomResourceDefinitionLister

	delegate          http.Handler
	restOptionsGetter generic.RESTOptionsGetter
	admission         admission.Interface
}

// crdInfo stores enough information to serve the storage for the custom resource
type crdInfo struct {
	storage      *customresource.REST
	requestScope handlers.RequestScope
}

// crdStorageMap goes from customresourcedefinition to its storage
type crdStorageMap map[types.UID]*crdInfo

func NewCustomResourceDefinitionHandler(
	versionDiscoveryHandler *versionDiscoveryHandler,
	groupDiscoveryHandler *groupDiscoveryHandler,
	requestContextMapper apirequest.RequestContextMapper,
	crdLister listers.CustomResourceDefinitionLister,
	crdInformer informers.CustomResourceDefinitionInformer,
	delegate http.Handler,
	restOptionsGetter generic.RESTOptionsGetter,
	admission admission.Interface) *crdHandler {
	ret := &crdHandler{
		versionDiscoveryHandler: versionDiscoveryHandler,
		groupDiscoveryHandler:   groupDiscoveryHandler,
		customStorage:           atomic.Value{},
		requestContextMapper:    requestContextMapper,
		crdLister:               crdLister,
		delegate:                delegate,
		restOptionsGetter:       restOptionsGetter,
		admission:               admission,
	}

	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: ret.updateCustomResourceDefinition,
	})

	ret.customStorage.Store(crdStorageMap{})
	return ret
}

func (r *crdHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ctx, ok := r.requestContextMapper.Get(req)
	if !ok {
		// programmer error
		panic("missing context")
	}
	requestInfo, ok := apirequest.RequestInfoFrom(ctx)
	if !ok {
		// programmer error
		panic("missing requestInfo")
	}
	if !requestInfo.IsResourceRequest {
		pathParts := splitPath(requestInfo.Path)
		// only match /apis/<group>/<version>
		// only registered under /apis
		if len(pathParts) == 3 {
			r.versionDiscoveryHandler.ServeHTTP(w, req)
			return
		}
		// only match /apis/<group>
		if len(pathParts) == 2 {
			r.groupDiscoveryHandler.ServeHTTP(w, req)
			return
		}

		r.delegate.ServeHTTP(w, req)
		return
	}

	crdName := requestInfo.Resource + "." + requestInfo.APIGroup
	crd, err := r.crdLister.Get(crdName)
	if apierrors.IsNotFound(err) {
		r.delegate.ServeHTTP(w, req)
		return
	}
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	if crd.Spec.Version != requestInfo.APIVersion {
		r.delegate.ServeHTTP(w, req)
		return
	}
	if !apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
		r.delegate.ServeHTTP(w, req)
	}
	if len(requestInfo.Subresource) > 0 {
		http.NotFound(w, req)
		return
	}

	terminating := apiextensions.IsCRDConditionTrue(crd, apiextensions.Terminating)

	crdInfo, err := r.getServingInfoFor(crd)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	storage := crdInfo.storage
	requestScope := crdInfo.requestScope
	minRequestTimeout := 1 * time.Minute

	switch requestInfo.Verb {
	case "get":
		handler := handlers.GetResource(storage, storage, requestScope)
		handler(w, req)
		return
	case "list":
		forceWatch := false
		handler := handlers.ListResource(storage, storage, requestScope, forceWatch, minRequestTimeout)
		handler(w, req)
		return
	case "watch":
		forceWatch := true
		handler := handlers.ListResource(storage, storage, requestScope, forceWatch, minRequestTimeout)
		handler(w, req)
		return
	case "create":
		if terminating {
			http.Error(w, fmt.Sprintf("%v not allowed while CustomResourceDefinition is terminating", requestInfo.Verb), http.StatusMethodNotAllowed)
			return
		}
		handler := handlers.CreateResource(storage, requestScope, discovery.NewUnstructuredObjectTyper(nil), r.admission)
		handler(w, req)
		return
	case "update":
		if terminating {
			http.Error(w, fmt.Sprintf("%v not allowed while CustomResourceDefinition is terminating", requestInfo.Verb), http.StatusMethodNotAllowed)
			return
		}
		handler := handlers.UpdateResource(storage, requestScope, discovery.NewUnstructuredObjectTyper(nil), r.admission)
		handler(w, req)
		return
	case "patch":
		if terminating {
			http.Error(w, fmt.Sprintf("%v not allowed while CustomResourceDefinition is terminating", requestInfo.Verb), http.StatusMethodNotAllowed)
			return
		}
		handler := handlers.PatchResource(storage, requestScope, r.admission, unstructured.UnstructuredObjectConverter{})
		handler(w, req)
		return
	case "delete":
		allowsOptions := true
		handler := handlers.DeleteResource(storage, allowsOptions, requestScope, r.admission)
		handler(w, req)
		return
	case "deletecollection":
		checkBody := true
		handler := handlers.DeleteCollection(storage, checkBody, requestScope, r.admission)
		handler(w, req)
		return

	default:
		http.Error(w, fmt.Sprintf("unhandled verb %q", requestInfo.Verb), http.StatusMethodNotAllowed)
		return
	}
}

// removeDeadStorage removes REST storage that isn't being used
func (r *crdHandler) removeDeadStorage() {
	// these don't have to be live.  A snapshot is fine
	// if we wrongly delete, that's ok.  The rest storage will be recreated on the next request
	// if we wrongly miss one, that's ok.  We'll get it next time
	storageMap := r.customStorage.Load().(crdStorageMap)
	allCustomResourceDefinitions, err := r.crdLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	for uid := range storageMap {
		found := false
		for _, crd := range allCustomResourceDefinitions {
			if crd.UID == uid {
				found = true
				break
			}
		}
		if !found {
			delete(storageMap, uid)
		}
	}

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	r.customStorage.Store(storageMap)
}

// GetCustomResourceListerCollectionDeleter returns the ListerCollectionDeleter for
// the given uid, or nil if one does not exist.
func (r *crdHandler) GetCustomResourceListerCollectionDeleter(crd *apiextensions.CustomResourceDefinition) finalizer.ListerCollectionDeleter {
	info, err := r.getServingInfoFor(crd)
	if err != nil {
		utilruntime.HandleError(err)
	}
	return info.storage
}

func (r *crdHandler) getServingInfoFor(crd *apiextensions.CustomResourceDefinition) (*crdInfo, error) {
	storageMap := r.customStorage.Load().(crdStorageMap)
	ret, ok := storageMap[crd.UID]
	if ok {
		return ret, nil
	}

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	ret, ok = storageMap[crd.UID]
	if ok {
		return ret, nil
	}

	// In addition to Unstructured objects (Custom Resources), we also may sometimes need to
	// decode unversioned Options objects, so we delegate to parameterScheme for such types.
	parameterScheme := runtime.NewScheme()
	parameterScheme.AddUnversionedTypes(schema.GroupVersion{Group: crd.Spec.Group, Version: crd.Spec.Version},
		&metav1.ListOptions{},
		&metav1.ExportOptions{},
		&metav1.GetOptions{},
		&metav1.DeleteOptions{},
	)
	parameterScheme.AddGeneratedDeepCopyFuncs(metav1.GetGeneratedDeepCopyFuncs()...)
	parameterCodec := runtime.NewParameterCodec(parameterScheme)

	kind := schema.GroupVersionKind{Group: crd.Spec.Group, Version: crd.Spec.Version, Kind: crd.Spec.Names.Kind}
	typer := unstructuredObjectTyper{
		delegate:          parameterScheme,
		unstructuredTyper: discovery.NewUnstructuredObjectTyper(nil),
	}
	creator := unstructuredCreator{}

	// convert CRD schema to openapi schema
	openapiSchema := &openapispec.Schema{}
	if err := apiservervalidation.ConvertToOpenAPITypes(crd, openapiSchema); err != nil {
		return nil, err
	}
	if err := openapispec.ExpandSchema(openapiSchema, nil, nil); err != nil {
		return nil, err
	}
	validator := validate.NewSchemaValidator(openapiSchema, nil, "", strfmt.Default)

	storage := customresource.NewREST(
		schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural},
		schema.GroupVersionKind{Group: crd.Spec.Group, Version: crd.Spec.Version, Kind: crd.Spec.Names.ListKind},
		UnstructuredCopier{},
		customresource.NewStrategy(
			typer,
			crd.Spec.Scope == apiextensions.NamespaceScoped,
			kind,
			validator,
		),
		r.restOptionsGetter,
	)

	selfLinkPrefix := ""
	switch crd.Spec.Scope {
	case apiextensions.ClusterScoped:
		selfLinkPrefix = "/" + path.Join("apis", crd.Spec.Group, crd.Spec.Version) + "/"
	case apiextensions.NamespaceScoped:
		selfLinkPrefix = "/" + path.Join("apis", crd.Spec.Group, crd.Spec.Version, "namespaces") + "/"
	}

	requestScope := handlers.RequestScope{
		Namer: handlers.ContextBasedNaming{
			GetContext: func(req *http.Request) apirequest.Context {
				ret, _ := r.requestContextMapper.Get(req)
				return ret
			},
			SelfLinker:         meta.NewAccessor(),
			ClusterScoped:      crd.Spec.Scope == apiextensions.ClusterScoped,
			SelfLinkPathPrefix: selfLinkPrefix,
		},
		ContextFunc: func(req *http.Request) apirequest.Context {
			ret, _ := r.requestContextMapper.Get(req)
			return ret
		},

		Serializer:     unstructuredNegotiatedSerializer{typer: typer, creator: creator},
		ParameterCodec: parameterCodec,

		Creater:         creator,
		Convertor:       unstructured.UnstructuredObjectConverter{},
		Defaulter:       unstructuredDefaulter{parameterScheme},
		Copier:          UnstructuredCopier{},
		Typer:           typer,
		UnsafeConvertor: unstructured.UnstructuredObjectConverter{},

		Resource:    schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural},
		Kind:        kind,
		Subresource: "",

		MetaGroupVersion: metav1.SchemeGroupVersion,
	}

	ret = &crdInfo{
		storage:      storage,
		requestScope: requestScope,
	}

	storageMap2 := make(crdStorageMap, len(storageMap))

	// Copy because we cannot write to storageMap without a race
	// as it is used without locking elsewhere
	for k, v := range storageMap {
		storageMap2[k] = v
	}

	storageMap2[crd.UID] = ret
	r.customStorage.Store(storageMap2)
	return ret, nil
}

func (c *crdHandler) updateCustomResourceDefinition(oldObj, _ interface{}) {
	oldCRD := oldObj.(*apiextensions.CustomResourceDefinition)
	glog.V(4).Infof("Updating customresourcedefinition %s", oldCRD.Name)

	c.customStorageLock.Lock()
	defer c.customStorageLock.Unlock()

	storageMap := c.customStorage.Load().(crdStorageMap)
	storageMap2 := make(crdStorageMap, len(storageMap))

	// Copy because we cannot write to storageMap without a race
	// as it is used without locking elsewhere
	for k, v := range storageMap {
		storageMap2[k] = v
	}

	delete(storageMap2, oldCRD.UID)
	c.customStorage.Store(storageMap2)
}

type unstructuredNegotiatedSerializer struct {
	typer   runtime.ObjectTyper
	creator runtime.ObjectCreater
}

func (s unstructuredNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{
		{
			MediaType:        "application/json",
			EncodesAsText:    true,
			Serializer:       json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, false),
			PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, true),
			StreamSerializer: &runtime.StreamSerializerInfo{
				EncodesAsText: true,
				Serializer:    json.NewSerializer(json.DefaultMetaFactory, s.creator, s.typer, false),
				Framer:        json.Framer,
			},
		},
	}
}

func (s unstructuredNegotiatedSerializer) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewDefaultingCodecForScheme(Scheme, crEncoderInstance, nil, gv, nil)
}

func (s unstructuredNegotiatedSerializer) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return unstructuredDecoder{delegate: Codecs.DecoderToVersion(serializer, gv)}
}

type unstructuredDecoder struct {
	delegate runtime.Decoder
}

func (d unstructuredDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	// Delegate for things other than Unstructured.
	if _, ok := into.(runtime.Unstructured); !ok && into != nil {
		return d.delegate.Decode(data, defaults, into)
	}
	return unstructured.UnstructuredJSONScheme.Decode(data, defaults, into)
}

type unstructuredObjectTyper struct {
	delegate          runtime.ObjectTyper
	unstructuredTyper runtime.ObjectTyper
}

func (t unstructuredObjectTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	// Delegate for things other than Unstructured.
	if _, ok := obj.(runtime.Unstructured); !ok {
		return t.delegate.ObjectKinds(obj)
	}
	return t.unstructuredTyper.ObjectKinds(obj)
}

func (t unstructuredObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return t.delegate.Recognizes(gvk) || t.unstructuredTyper.Recognizes(gvk)
}

var crEncoderInstance = crEncoder{}

// crEncoder *usually* encodes using the unstructured.UnstructuredJSONScheme, but if the type is Status or WatchEvent
// it will serialize them out using the converting codec.
type crEncoder struct{}

func (crEncoder) Encode(obj runtime.Object, w io.Writer) error {
	switch t := obj.(type) {
	case *metav1.Status, *metav1.WatchEvent:
		for _, info := range Codecs.SupportedMediaTypes() {
			// we are always json
			if info.MediaType == "application/json" {
				return info.Serializer.Encode(obj, w)
			}
		}

		return fmt.Errorf("unable to find json serializer for %T", t)

	default:
		return unstructured.UnstructuredJSONScheme.Encode(obj, w)
	}
}

type unstructuredCreator struct{}

func (c unstructuredCreator) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	ret := &unstructured.Unstructured{}
	ret.SetGroupVersionKind(kind)
	return ret, nil
}

type UnstructuredCopier struct{}

func (UnstructuredCopier) Copy(obj runtime.Object) (runtime.Object, error) {
	if _, ok := obj.(runtime.Unstructured); !ok {
		// Callers should not use this UnstructuredCopier for things other than Unstructured.
		// If they do, the copy they get back will become Unstructured, which can lead to
		// difficult-to-debug errors downstream. To make such errors more obvious,
		// we explicitly reject anything that isn't Unstructured.
		return nil, fmt.Errorf("UnstructuredCopier can't copy type %T", obj)
	}

	// serialize and deserialize to ensure a clean copy
	buf := &bytes.Buffer{}
	err := unstructured.UnstructuredJSONScheme.Encode(obj, buf)
	if err != nil {
		return nil, err
	}
	out := &unstructured.Unstructured{}
	result, _, err := unstructured.UnstructuredJSONScheme.Decode(buf.Bytes(), nil, out)
	return result, err
}

type unstructuredDefaulter struct {
	delegate runtime.ObjectDefaulter
}

func (d unstructuredDefaulter) Default(in runtime.Object) {
	// Delegate for things other than Unstructured.
	if _, ok := in.(runtime.Unstructured); !ok {
		d.delegate.Default(in)
	}
}

type CRDRESTOptionsGetter struct {
	StorageConfig           storagebackend.Config
	StoragePrefix           string
	EnableWatchCache        bool
	DefaultWatchCacheSize   int
	EnableGarbageCollection bool
	DeleteCollectionWorkers int
}

func (t CRDRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	ret := generic.RESTOptions{
		StorageConfig:           &t.StorageConfig,
		Decorator:               generic.UndecoratedStorage,
		EnableGarbageCollection: t.EnableGarbageCollection,
		DeleteCollectionWorkers: t.DeleteCollectionWorkers,
		ResourcePrefix:          resource.Group + "/" + resource.Resource,
	}
	if t.EnableWatchCache {
		ret.Decorator = genericregistry.StorageWithCacher(t.DefaultWatchCacheSize)
	}
	return ret, nil
}

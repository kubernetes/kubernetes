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
	"net/http"
	"path"
	"sync"
	"sync/atomic"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/client-go/discovery"

	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	listers "k8s.io/kube-apiextensions-server/pkg/client/listers/apiextensions/internalversion"
	"k8s.io/kube-apiextensions-server/pkg/registry/customresource"
)

// crdHandler serves the `/apis` endpoint.
// This is registered as a filter so that it never collides with any explictly registered endpoints
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

	ret.customStorage.Store(crdStorageMap{})
	return ret
}

func (r *crdHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ctx, ok := r.requestContextMapper.Get(req)
	if !ok {
		// programmer error
		panic("missing context")
		return
	}
	requestInfo, ok := apirequest.RequestInfoFrom(ctx)
	if !ok {
		// programmer error
		panic("missing requestInfo")
		return
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
	if len(requestInfo.Subresource) > 0 {
		http.NotFound(w, req)
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
	// if we can't definitively determine that our names are good, delegate
	if !apiextensions.IsCRDConditionFalse(crd, apiextensions.NameConflict) {
		r.delegate.ServeHTTP(w, req)
	}

	crdInfo := r.getServingInfoFor(crd)
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
		handler := handlers.CreateResource(storage, requestScope, discovery.NewUnstructuredObjectTyper(nil), r.admission)
		handler(w, req)
		return
	case "update":
		handler := handlers.UpdateResource(storage, requestScope, discovery.NewUnstructuredObjectTyper(nil), r.admission)
		handler(w, req)
		return
	case "patch":
		handler := handlers.PatchResource(storage, requestScope, r.admission, unstructured.UnstructuredObjectConverter{})
		handler(w, req)
		return
	case "delete":
		allowsOptions := true
		handler := handlers.DeleteResource(storage, allowsOptions, requestScope, r.admission)
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

func (r *crdHandler) getServingInfoFor(crd *apiextensions.CustomResourceDefinition) *crdInfo {
	storageMap := r.customStorage.Load().(crdStorageMap)
	ret, ok := storageMap[crd.UID]
	if ok {
		return ret
	}

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	ret, ok = storageMap[crd.UID]
	if ok {
		return ret
	}

	storage := customresource.NewREST(
		schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Spec.Names.Plural},
		schema.GroupVersionKind{Group: crd.Spec.Group, Version: crd.Spec.Version, Kind: crd.Spec.Names.ListKind},
		UnstructuredCopier{},
		customresource.NewStrategy(discovery.NewUnstructuredObjectTyper(nil), crd.Spec.Scope == apiextensions.NamespaceScoped),
		r.restOptionsGetter,
	)

	parameterScheme := runtime.NewScheme()
	parameterScheme.AddUnversionedTypes(schema.GroupVersion{Group: crd.Spec.Group, Version: crd.Spec.Version},
		&metav1.ListOptions{},
		&metav1.ExportOptions{},
		&metav1.GetOptions{},
		&metav1.DeleteOptions{},
	)
	parameterScheme.AddGeneratedDeepCopyFuncs(metav1.GetGeneratedDeepCopyFuncs()...)
	parameterCodec := runtime.NewParameterCodec(parameterScheme)

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

		Serializer:     UnstructuredNegotiatedSerializer{},
		ParameterCodec: parameterCodec,

		Creater:         UnstructuredCreator{},
		Convertor:       unstructured.UnstructuredObjectConverter{},
		Defaulter:       UnstructuredDefaulter{},
		Copier:          UnstructuredCopier{},
		Typer:           discovery.NewUnstructuredObjectTyper(nil),
		UnsafeConvertor: unstructured.UnstructuredObjectConverter{},

		Resource:    schema.GroupVersionResource{Group: crd.Spec.Group, Version: crd.Spec.Version, Resource: crd.Spec.Names.Plural},
		Kind:        schema.GroupVersionKind{Group: crd.Spec.Group, Version: crd.Spec.Version, Kind: crd.Spec.Names.Kind},
		Subresource: "",

		MetaGroupVersion: metav1.SchemeGroupVersion,
	}

	ret = &crdInfo{
		storage:      storage,
		requestScope: requestScope,
	}
	storageMap[crd.UID] = ret
	r.customStorage.Store(storageMap)
	return ret
}

type UnstructuredNegotiatedSerializer struct{}

func (s UnstructuredNegotiatedSerializer) SupportedMediaTypes() []runtime.SerializerInfo {
	return []runtime.SerializerInfo{
		{
			MediaType:        "application/json",
			EncodesAsText:    true,
			Serializer:       json.NewSerializer(json.DefaultMetaFactory, UnstructuredCreator{}, discovery.NewUnstructuredObjectTyper(nil), false),
			PrettySerializer: json.NewSerializer(json.DefaultMetaFactory, UnstructuredCreator{}, discovery.NewUnstructuredObjectTyper(nil), true),
			StreamSerializer: &runtime.StreamSerializerInfo{
				EncodesAsText: true,
				Serializer:    json.NewSerializer(json.DefaultMetaFactory, UnstructuredCreator{}, discovery.NewUnstructuredObjectTyper(nil), false),
				Framer:        json.Framer,
			},
		},
	}
}

func (s UnstructuredNegotiatedSerializer) EncoderForVersion(serializer runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return unstructured.UnstructuredJSONScheme
}

func (s UnstructuredNegotiatedSerializer) DecoderToVersion(serializer runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	return unstructured.UnstructuredJSONScheme
}

type UnstructuredCreator struct{}

func (UnstructuredCreator) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	ret := &unstructured.Unstructured{}
	ret.SetGroupVersionKind(kind)
	return ret, nil
}

type UnstructuredCopier struct{}

func (UnstructuredCopier) Copy(obj runtime.Object) (runtime.Object, error) {
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

type UnstructuredDefaulter struct{}

func (UnstructuredDefaulter) Default(in runtime.Object) {}

type CustomResourceDefinitionRESTOptionsGetter struct {
	StorageConfig           storagebackend.Config
	StoragePrefix           string
	EnableWatchCache        bool
	DefaultWatchCacheSize   int
	EnableGarbageCollection bool
	DeleteCollectionWorkers int
}

func (t CustomResourceDefinitionRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	ret := generic.RESTOptions{
		StorageConfig:           &t.StorageConfig,
		Decorator:               generic.UndecoratedStorage,
		EnableGarbageCollection: t.EnableGarbageCollection,
		DeleteCollectionWorkers: t.DeleteCollectionWorkers,
		ResourcePrefix:          t.StoragePrefix + "/" + resource.Group + "/" + resource.Resource,
	}
	if t.EnableWatchCache {
		ret.Decorator = genericregistry.StorageWithCacher(t.DefaultWatchCacheSize)
	}
	return ret, nil
}

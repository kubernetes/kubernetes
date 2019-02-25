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
	"fmt"
	"net/http"
	"path"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/go-openapi/spec"
	"github.com/go-openapi/strfmt"
	"github.com/go-openapi/validate"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/conversion"
	apiservervalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	informers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion"
	listers "k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion"
	"k8s.io/apiextensions-apiserver/pkg/controller/establish"
	"k8s.io/apiextensions-apiserver/pkg/controller/finalizer"
	"k8s.io/apiextensions-apiserver/pkg/crdserverscheme"
	apiextensionsfeatures "k8s.io/apiextensions-apiserver/pkg/features"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresource/tableconvertor"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/handlers"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/metrics"
	apirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
	"k8s.io/client-go/scale"
	"k8s.io/client-go/scale/scheme/autoscalingv1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/klog"
)

// crdHandler serves the `/apis` endpoint.
// This is registered as a filter so that it never collides with any explicitly registered endpoints
type crdHandler struct {
	versionDiscoveryHandler *versionDiscoveryHandler
	groupDiscoveryHandler   *groupDiscoveryHandler

	customStorageLock sync.Mutex
	// customStorage contains a crdStorageMap
	// atomic.Value has a very good read performance compared to sync.RWMutex
	// see https://gist.github.com/dim/152e6bf80e1384ea72e17ac717a5000a
	// which is suited for most read and rarely write cases
	customStorage atomic.Value

	crdLister listers.CustomResourceDefinitionLister

	delegate          http.Handler
	restOptionsGetter generic.RESTOptionsGetter
	admission         admission.Interface

	establishingController *establish.EstablishingController

	// MasterCount is used to implement sleep to improve
	// CRD establishing process for HA clusters.
	masterCount int

	converterFactory *conversion.CRConverterFactory

	// so that we can do create on update.
	authorizer authorizer.Authorizer
}

// crdInfo stores enough information to serve the storage for the custom resource
type crdInfo struct {
	// spec and acceptedNames are used to compare against if a change is made on a CRD. We only update
	// the storage if one of these changes.
	spec          *apiextensions.CustomResourceDefinitionSpec
	acceptedNames *apiextensions.CustomResourceDefinitionNames

	// Storage per version
	storages map[string]customresource.CustomResourceStorage

	// Request scope per version
	requestScopes map[string]handlers.RequestScope

	// Scale scope per version
	scaleRequestScopes map[string]handlers.RequestScope

	// Status scope per version
	statusRequestScopes map[string]handlers.RequestScope

	// storageVersion is the CRD version used when storing the object in etcd.
	storageVersion string
}

// crdStorageMap goes from customresourcedefinition to its storage
type crdStorageMap map[types.UID]*crdInfo

func NewCustomResourceDefinitionHandler(
	versionDiscoveryHandler *versionDiscoveryHandler,
	groupDiscoveryHandler *groupDiscoveryHandler,
	crdInformer informers.CustomResourceDefinitionInformer,
	delegate http.Handler,
	restOptionsGetter generic.RESTOptionsGetter,
	admission admission.Interface,
	establishingController *establish.EstablishingController,
	serviceResolver webhook.ServiceResolver,
	authResolverWrapper webhook.AuthenticationInfoResolverWrapper,
	masterCount int,
	authorizer authorizer.Authorizer) (*crdHandler, error) {
	ret := &crdHandler{
		versionDiscoveryHandler: versionDiscoveryHandler,
		groupDiscoveryHandler:   groupDiscoveryHandler,
		customStorage:           atomic.Value{},
		crdLister:               crdInformer.Lister(),
		delegate:                delegate,
		restOptionsGetter:       restOptionsGetter,
		admission:               admission,
		establishingController:  establishingController,
		masterCount:             masterCount,
		authorizer:              authorizer,
	}
	crdInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: ret.updateCustomResourceDefinition,
		DeleteFunc: func(obj interface{}) {
			ret.removeDeadStorage()
		},
	})
	crConverterFactory, err := conversion.NewCRConverterFactory(serviceResolver, authResolverWrapper)
	if err != nil {
		return nil, err
	}
	ret.converterFactory = crConverterFactory

	ret.customStorage.Store(crdStorageMap{})

	return ret, nil
}

func (r *crdHandler) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	ctx := req.Context()
	requestInfo, ok := apirequest.RequestInfoFrom(ctx)
	if !ok {
		responsewriters.InternalError(w, req, fmt.Errorf("no RequestInfo found in the context"))
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
	if !apiextensions.HasServedCRDVersion(crd, requestInfo.APIVersion) {
		r.delegate.ServeHTTP(w, req)
		return
	}
	// There is a small chance that a CRD is being served because NamesAccepted condition is true,
	// but it becomes "unserved" because another names update leads to a conflict
	// and EstablishingController wasn't fast enough to put the CRD into the Established condition.
	// We accept this as the problem is small and self-healing.
	if !apiextensions.IsCRDConditionTrue(crd, apiextensions.NamesAccepted) &&
		!apiextensions.IsCRDConditionTrue(crd, apiextensions.Established) {
		r.delegate.ServeHTTP(w, req)
		return
	}

	terminating := apiextensions.IsCRDConditionTrue(crd, apiextensions.Terminating)

	crdInfo, err := r.getOrCreateServingInfoFor(crd)
	if err != nil {
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	verb := strings.ToUpper(requestInfo.Verb)
	resource := requestInfo.Resource
	subresource := requestInfo.Subresource
	scope := metrics.CleanScope(requestInfo)
	supportedTypes := []string{
		string(types.JSONPatchType),
		string(types.MergePatchType),
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.ServerSideApply) {
		supportedTypes = append(supportedTypes, string(types.ApplyPatchType))
	}

	var handler http.HandlerFunc
	subresources, err := apiextensions.GetSubresourcesForVersion(crd, requestInfo.APIVersion)
	if err != nil {
		utilruntime.HandleError(err)
		http.Error(w, "the server could not properly serve the CR subresources", http.StatusInternalServerError)
		return
	}
	switch {
	case subresource == "status" && subresources != nil && subresources.Status != nil:
		handler = r.serveStatus(w, req, requestInfo, crdInfo, terminating, supportedTypes)
	case subresource == "scale" && subresources != nil && subresources.Scale != nil:
		handler = r.serveScale(w, req, requestInfo, crdInfo, terminating, supportedTypes)
	case len(subresource) == 0:
		handler = r.serveResource(w, req, requestInfo, crdInfo, terminating, supportedTypes)
	default:
		http.Error(w, "the server could not find the requested resource", http.StatusNotFound)
	}

	if handler != nil {
		handler = metrics.InstrumentHandlerFunc(verb, requestInfo.APIGroup, requestInfo.APIVersion, resource, subresource, scope, metrics.APIServerComponent, handler)
		handler(w, req)
		return
	}
}

func (r *crdHandler) serveResource(w http.ResponseWriter, req *http.Request, requestInfo *apirequest.RequestInfo, crdInfo *crdInfo, terminating bool, supportedTypes []string) http.HandlerFunc {
	requestScope := crdInfo.requestScopes[requestInfo.APIVersion]
	storage := crdInfo.storages[requestInfo.APIVersion].CustomResource
	minRequestTimeout := 1 * time.Minute

	switch requestInfo.Verb {
	case "get":
		return handlers.GetResource(storage, storage, requestScope)
	case "list":
		forceWatch := false
		return handlers.ListResource(storage, storage, requestScope, forceWatch, minRequestTimeout)
	case "watch":
		forceWatch := true
		return handlers.ListResource(storage, storage, requestScope, forceWatch, minRequestTimeout)
	case "create":
		if terminating {
			http.Error(w, fmt.Sprintf("%v not allowed while CustomResourceDefinition is terminating", requestInfo.Verb), http.StatusMethodNotAllowed)
			return nil
		}
		return handlers.CreateResource(storage, requestScope, r.admission)
	case "update":
		return handlers.UpdateResource(storage, requestScope, r.admission)
	case "patch":
		return handlers.PatchResource(storage, requestScope, r.admission, supportedTypes)
	case "delete":
		allowsOptions := true
		return handlers.DeleteResource(storage, allowsOptions, requestScope, r.admission)
	case "deletecollection":
		checkBody := true
		return handlers.DeleteCollection(storage, checkBody, requestScope, r.admission)
	default:
		http.Error(w, fmt.Sprintf("unhandled verb %q", requestInfo.Verb), http.StatusMethodNotAllowed)
		return nil
	}
}

func (r *crdHandler) serveStatus(w http.ResponseWriter, req *http.Request, requestInfo *apirequest.RequestInfo, crdInfo *crdInfo, terminating bool, supportedTypes []string) http.HandlerFunc {
	requestScope := crdInfo.statusRequestScopes[requestInfo.APIVersion]
	storage := crdInfo.storages[requestInfo.APIVersion].Status

	switch requestInfo.Verb {
	case "get":
		return handlers.GetResource(storage, nil, requestScope)
	case "update":
		return handlers.UpdateResource(storage, requestScope, r.admission)
	case "patch":
		return handlers.PatchResource(storage, requestScope, r.admission, supportedTypes)
	default:
		http.Error(w, fmt.Sprintf("unhandled verb %q", requestInfo.Verb), http.StatusMethodNotAllowed)
		return nil
	}
}

func (r *crdHandler) serveScale(w http.ResponseWriter, req *http.Request, requestInfo *apirequest.RequestInfo, crdInfo *crdInfo, terminating bool, supportedTypes []string) http.HandlerFunc {
	requestScope := crdInfo.scaleRequestScopes[requestInfo.APIVersion]
	storage := crdInfo.storages[requestInfo.APIVersion].Scale

	switch requestInfo.Verb {
	case "get":
		return handlers.GetResource(storage, nil, requestScope)
	case "update":
		return handlers.UpdateResource(storage, requestScope, r.admission)
	case "patch":
		return handlers.PatchResource(storage, requestScope, r.admission, supportedTypes)
	default:
		http.Error(w, fmt.Sprintf("unhandled verb %q", requestInfo.Verb), http.StatusMethodNotAllowed)
		return nil
	}
}

func (r *crdHandler) updateCustomResourceDefinition(oldObj, newObj interface{}) {
	oldCRD := oldObj.(*apiextensions.CustomResourceDefinition)
	newCRD := newObj.(*apiextensions.CustomResourceDefinition)

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	// Add CRD to the establishing controller queue.
	// For HA clusters, we want to prevent race conditions when changing status to Established,
	// so we want to be sure that CRD is Installing at least for 5 seconds before Establishing it.
	// TODO: find a real HA safe checkpointing mechanism instead of an arbitrary wait.
	if !apiextensions.IsCRDConditionTrue(newCRD, apiextensions.Established) &&
		apiextensions.IsCRDConditionTrue(newCRD, apiextensions.NamesAccepted) {
		if r.masterCount > 1 {
			r.establishingController.QueueCRD(newCRD.Name, 5*time.Second)
		} else {
			r.establishingController.QueueCRD(newCRD.Name, 0)
		}
	}

	storageMap := r.customStorage.Load().(crdStorageMap)
	oldInfo, found := storageMap[newCRD.UID]
	if !found {
		return
	}
	if apiequality.Semantic.DeepEqual(&newCRD.Spec, oldInfo.spec) && apiequality.Semantic.DeepEqual(&newCRD.Status.AcceptedNames, oldInfo.acceptedNames) {
		klog.V(6).Infof("Ignoring customresourcedefinition %s update because neither spec, nor accepted names changed", oldCRD.Name)
		return
	}

	klog.V(4).Infof("Updating customresourcedefinition %s", oldCRD.Name)

	// Copy because we cannot write to storageMap without a race
	// as it is used without locking elsewhere.
	storageMap2 := storageMap.clone()
	if oldInfo, ok := storageMap2[types.UID(oldCRD.UID)]; ok {
		for _, storage := range oldInfo.storages {
			// destroy only the main storage. Those for the subresources share cacher and etcd clients.
			storage.CustomResource.DestroyFunc()
		}
		delete(storageMap2, types.UID(oldCRD.UID))
	}

	r.customStorage.Store(storageMap2)
}

// removeDeadStorage removes REST storage that isn't being used
func (r *crdHandler) removeDeadStorage() {
	allCustomResourceDefinitions, err := r.crdLister.List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	storageMap := r.customStorage.Load().(crdStorageMap)
	// Copy because we cannot write to storageMap without a race
	// as it is used without locking elsewhere
	storageMap2 := storageMap.clone()
	for uid, s := range storageMap2 {
		found := false
		for _, crd := range allCustomResourceDefinitions {
			if crd.UID == uid {
				found = true
				break
			}
		}
		if !found {
			klog.V(4).Infof("Removing dead CRD storage for %s/%s", s.spec.Group, s.spec.Names.Kind)
			for _, storage := range s.storages {
				// destroy only the main storage. Those for the subresources share cacher and etcd clients.
				storage.CustomResource.DestroyFunc()
			}
			delete(storageMap2, uid)
		}
	}
	r.customStorage.Store(storageMap2)
}

// GetCustomResourceListerCollectionDeleter returns the ListerCollectionDeleter of
// the given crd.
func (r *crdHandler) GetCustomResourceListerCollectionDeleter(crd *apiextensions.CustomResourceDefinition) (finalizer.ListerCollectionDeleter, error) {
	info, err := r.getOrCreateServingInfoFor(crd)
	if err != nil {
		return nil, err
	}
	return info.storages[info.storageVersion].CustomResource, nil
}

func (r *crdHandler) getOrCreateServingInfoFor(crd *apiextensions.CustomResourceDefinition) (*crdInfo, error) {
	storageMap := r.customStorage.Load().(crdStorageMap)
	if ret, ok := storageMap[crd.UID]; ok {
		return ret, nil
	}

	r.customStorageLock.Lock()
	defer r.customStorageLock.Unlock()

	storageMap = r.customStorage.Load().(crdStorageMap)
	if ret, ok := storageMap[crd.UID]; ok {
		return ret, nil
	}

	storageVersion, err := apiextensions.GetCRDStorageVersion(crd)
	if err != nil {
		return nil, err
	}

	// Scope/Storages per version.
	requestScopes := map[string]handlers.RequestScope{}
	storages := map[string]customresource.CustomResourceStorage{}
	statusScopes := map[string]handlers.RequestScope{}
	scaleScopes := map[string]handlers.RequestScope{}

	for _, v := range crd.Spec.Versions {
		safeConverter, unsafeConverter, err := r.converterFactory.NewConverter(crd)
		if err != nil {
			return nil, err
		}
		// In addition to Unstructured objects (Custom Resources), we also may sometimes need to
		// decode unversioned Options objects, so we delegate to parameterScheme for such types.
		parameterScheme := runtime.NewScheme()
		parameterScheme.AddUnversionedTypes(schema.GroupVersion{Group: crd.Spec.Group, Version: v.Name},
			&metav1.ListOptions{},
			&metav1.ExportOptions{},
			&metav1.GetOptions{},
			&metav1.DeleteOptions{},
		)
		parameterCodec := runtime.NewParameterCodec(parameterScheme)

		kind := schema.GroupVersionKind{Group: crd.Spec.Group, Version: v.Name, Kind: crd.Status.AcceptedNames.Kind}
		typer := newUnstructuredObjectTyper(parameterScheme)
		creator := unstructuredCreator{}

		validationSchema, err := apiextensions.GetSchemaForVersion(crd, v.Name)
		if err != nil {
			utilruntime.HandleError(err)
			return nil, fmt.Errorf("the server could not properly serve the CR schema")
		}
		validator, _, err := apiservervalidation.NewSchemaValidator(validationSchema)
		if err != nil {
			return nil, err
		}

		var statusSpec *apiextensions.CustomResourceSubresourceStatus
		var statusValidator *validate.SchemaValidator
		subresources, err := apiextensions.GetSubresourcesForVersion(crd, v.Name)
		if err != nil {
			utilruntime.HandleError(err)
			return nil, fmt.Errorf("the server could not properly serve the CR subresources")
		}
		if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceSubresources) && subresources != nil && subresources.Status != nil {
			statusSpec = subresources.Status
			// for the status subresource, validate only against the status schema
			if validationSchema != nil && validationSchema.OpenAPIV3Schema != nil && validationSchema.OpenAPIV3Schema.Properties != nil {
				if statusSchema, ok := validationSchema.OpenAPIV3Schema.Properties["status"]; ok {
					openapiSchema := &spec.Schema{}
					if err := apiservervalidation.ConvertJSONSchemaProps(&statusSchema, openapiSchema); err != nil {
						return nil, err
					}
					statusValidator = validate.NewSchemaValidator(openapiSchema, nil, "", strfmt.Default)
				}
			}
		}

		var scaleSpec *apiextensions.CustomResourceSubresourceScale
		if utilfeature.DefaultFeatureGate.Enabled(apiextensionsfeatures.CustomResourceSubresources) && subresources != nil && subresources.Scale != nil {
			scaleSpec = subresources.Scale
		}

		columns, err := apiextensions.GetColumnsForVersion(crd, v.Name)
		if err != nil {
			utilruntime.HandleError(err)
			return nil, fmt.Errorf("the server could not properly serve the CR columns")
		}
		table, err := tableconvertor.New(columns)
		if err != nil {
			klog.V(2).Infof("The CRD for %v has an invalid printer specification, falling back to default printing: %v", kind, err)
		}

		storages[v.Name] = customresource.NewStorage(
			schema.GroupResource{Group: crd.Spec.Group, Resource: crd.Status.AcceptedNames.Plural},
			schema.GroupVersionKind{Group: crd.Spec.Group, Version: v.Name, Kind: crd.Status.AcceptedNames.Kind},
			schema.GroupVersionKind{Group: crd.Spec.Group, Version: v.Name, Kind: crd.Status.AcceptedNames.ListKind},
			customresource.NewStrategy(
				typer,
				crd.Spec.Scope == apiextensions.NamespaceScoped,
				kind,
				validator,
				statusValidator,
				statusSpec,
				scaleSpec,
			),
			crdConversionRESTOptionsGetter{
				RESTOptionsGetter: r.restOptionsGetter,
				converter:         safeConverter,
				decoderVersion:    schema.GroupVersion{Group: crd.Spec.Group, Version: v.Name},
				encoderVersion:    schema.GroupVersion{Group: crd.Spec.Group, Version: storageVersion},
			},
			crd.Status.AcceptedNames.Categories,
			table,
		)

		selfLinkPrefix := ""
		switch crd.Spec.Scope {
		case apiextensions.ClusterScoped:
			selfLinkPrefix = "/" + path.Join("apis", crd.Spec.Group, v.Name) + "/" + crd.Status.AcceptedNames.Plural + "/"
		case apiextensions.NamespaceScoped:
			selfLinkPrefix = "/" + path.Join("apis", crd.Spec.Group, v.Name, "namespaces") + "/"
		}

		clusterScoped := crd.Spec.Scope == apiextensions.ClusterScoped

		requestScopes[v.Name] = handlers.RequestScope{
			Namer: handlers.ContextBasedNaming{
				SelfLinker:         meta.NewAccessor(),
				ClusterScoped:      clusterScoped,
				SelfLinkPathPrefix: selfLinkPrefix,
			},
			Serializer:     unstructuredNegotiatedSerializer{typer: typer, creator: creator, converter: safeConverter},
			ParameterCodec: parameterCodec,

			Creater:         creator,
			Convertor:       safeConverter,
			Defaulter:       unstructuredDefaulter{parameterScheme},
			Typer:           typer,
			UnsafeConvertor: unsafeConverter,

			Resource: schema.GroupVersionResource{Group: crd.Spec.Group, Version: v.Name, Resource: crd.Status.AcceptedNames.Plural},
			Kind:     kind,

			// a handler for a specific group-version of a custom resource uses that version as the in-memory representation
			HubGroupVersion: kind.GroupVersion(),

			MetaGroupVersion: metav1.SchemeGroupVersion,

			TableConvertor: storages[v.Name].CustomResource,

			Authorizer: r.authorizer,
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.ServerSideApply) {
			reqScope := requestScopes[v.Name]
			reqScope.FieldManager = fieldmanager.NewCRDFieldManager(
				reqScope.Convertor,
				reqScope.Defaulter,
				reqScope.Kind.GroupVersion(),
				reqScope.HubGroupVersion,
			)
			requestScopes[v.Name] = reqScope
		}

		// override scaleSpec subresource values
		// shallow copy
		scaleScope := requestScopes[v.Name]
		scaleConverter := scale.NewScaleConverter()
		scaleScope.Subresource = "scale"
		scaleScope.Serializer = serializer.NewCodecFactory(scaleConverter.Scheme())
		scaleScope.Kind = autoscalingv1.SchemeGroupVersion.WithKind("Scale")
		scaleScope.Namer = handlers.ContextBasedNaming{
			SelfLinker:         meta.NewAccessor(),
			ClusterScoped:      clusterScoped,
			SelfLinkPathPrefix: selfLinkPrefix,
			SelfLinkPathSuffix: "/scale",
		}
		scaleScopes[v.Name] = scaleScope

		// override status subresource values
		// shallow copy
		statusScope := requestScopes[v.Name]
		statusScope.Subresource = "status"
		statusScope.Namer = handlers.ContextBasedNaming{
			SelfLinker:         meta.NewAccessor(),
			ClusterScoped:      clusterScoped,
			SelfLinkPathPrefix: selfLinkPrefix,
			SelfLinkPathSuffix: "/status",
		}
		statusScopes[v.Name] = statusScope
	}

	ret := &crdInfo{
		spec:                &crd.Spec,
		acceptedNames:       &crd.Status.AcceptedNames,
		storages:            storages,
		requestScopes:       requestScopes,
		scaleRequestScopes:  scaleScopes,
		statusRequestScopes: statusScopes,
		storageVersion:      storageVersion,
	}

	// Copy because we cannot write to storageMap without a race
	// as it is used without locking elsewhere.
	storageMap2 := storageMap.clone()

	storageMap2[crd.UID] = ret
	r.customStorage.Store(storageMap2)

	return ret, nil
}

type unstructuredNegotiatedSerializer struct {
	typer     runtime.ObjectTyper
	creator   runtime.ObjectCreater
	converter runtime.ObjectConvertor
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
		{
			MediaType:     "application/yaml",
			EncodesAsText: true,
			Serializer:    json.NewYAMLSerializer(json.DefaultMetaFactory, s.creator, s.typer),
		},
	}
}

func (s unstructuredNegotiatedSerializer) EncoderForVersion(encoder runtime.Encoder, gv runtime.GroupVersioner) runtime.Encoder {
	return versioning.NewCodec(encoder, nil, s.converter, Scheme, Scheme, Scheme, gv, nil, "crdNegotiatedSerializer")
}

func (s unstructuredNegotiatedSerializer) DecoderToVersion(decoder runtime.Decoder, gv runtime.GroupVersioner) runtime.Decoder {
	d := schemaCoercingDecoder{delegate: decoder, validator: unstructuredSchemaCoercer{}}
	return versioning.NewDefaultingCodecForScheme(Scheme, nil, d, nil, gv)
}

type UnstructuredObjectTyper struct {
	Delegate          runtime.ObjectTyper
	UnstructuredTyper runtime.ObjectTyper
}

func newUnstructuredObjectTyper(Delegate runtime.ObjectTyper) UnstructuredObjectTyper {
	return UnstructuredObjectTyper{
		Delegate:          Delegate,
		UnstructuredTyper: crdserverscheme.NewUnstructuredObjectTyper(),
	}
}

func (t UnstructuredObjectTyper) ObjectKinds(obj runtime.Object) ([]schema.GroupVersionKind, bool, error) {
	// Delegate for things other than Unstructured.
	if _, ok := obj.(runtime.Unstructured); !ok {
		return t.Delegate.ObjectKinds(obj)
	}
	return t.UnstructuredTyper.ObjectKinds(obj)
}

func (t UnstructuredObjectTyper) Recognizes(gvk schema.GroupVersionKind) bool {
	return t.Delegate.Recognizes(gvk) || t.UnstructuredTyper.Recognizes(gvk)
}

type unstructuredCreator struct{}

func (c unstructuredCreator) New(kind schema.GroupVersionKind) (runtime.Object, error) {
	ret := &unstructured.Unstructured{}
	ret.SetGroupVersionKind(kind)
	return ret, nil
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
	CountMetricPollPeriod   time.Duration
}

func (t CRDRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	ret := generic.RESTOptions{
		StorageConfig:           &t.StorageConfig,
		Decorator:               generic.UndecoratedStorage,
		EnableGarbageCollection: t.EnableGarbageCollection,
		DeleteCollectionWorkers: t.DeleteCollectionWorkers,
		ResourcePrefix:          resource.Group + "/" + resource.Resource,
		CountMetricPollPeriod:   t.CountMetricPollPeriod,
	}
	if t.EnableWatchCache {
		ret.Decorator = genericregistry.StorageWithCacher(t.DefaultWatchCacheSize)
	}
	return ret, nil
}

// clone returns a clone of the provided crdStorageMap.
// The clone is a shallow copy of the map.
func (in crdStorageMap) clone() crdStorageMap {
	if in == nil {
		return nil
	}
	out := make(crdStorageMap, len(in))
	for key, value := range in {
		out[key] = value
	}
	return out
}

// crdConversionRESTOptionsGetter overrides the codec with one using the
// provided custom converter and custom encoder and decoder version.
type crdConversionRESTOptionsGetter struct {
	generic.RESTOptionsGetter
	converter      runtime.ObjectConvertor
	encoderVersion schema.GroupVersion
	decoderVersion schema.GroupVersion
}

func (t crdConversionRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource) (generic.RESTOptions, error) {
	ret, err := t.RESTOptionsGetter.GetRESTOptions(resource)
	if err == nil {
		d := schemaCoercingDecoder{delegate: ret.StorageConfig.Codec, validator: unstructuredSchemaCoercer{
			// drop invalid fields while decoding old CRs (before we had any ObjectMeta validation)
			dropInvalidMetadata: true,
		}}
		c := schemaCoercingConverter{delegate: t.converter, validator: unstructuredSchemaCoercer{}}
		ret.StorageConfig.Codec = versioning.NewCodec(
			ret.StorageConfig.Codec,
			d,
			c,
			&unstructuredCreator{},
			crdserverscheme.NewUnstructuredObjectTyper(),
			&unstructuredDefaulter{delegate: Scheme},
			t.encoderVersion,
			t.decoderVersion,
			"crdRESTOptions",
		)
	}
	return ret, err
}

// schemaCoercingDecoder calls the delegate decoder, and then applies the Unstructured schema validator
// to coerce the schema.
type schemaCoercingDecoder struct {
	delegate  runtime.Decoder
	validator unstructuredSchemaCoercer
}

var _ runtime.Decoder = schemaCoercingDecoder{}

func (d schemaCoercingDecoder) Decode(data []byte, defaults *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	obj, gvk, err := d.delegate.Decode(data, defaults, into)
	if err != nil {
		return nil, gvk, err
	}
	if u, ok := obj.(*unstructured.Unstructured); ok {
		if err := d.validator.apply(u); err != nil {
			return nil, gvk, err
		}
	}

	return obj, gvk, nil
}

// schemaCoercingConverter calls the delegate converter and applies the Unstructured validator to
// coerce the schema.
type schemaCoercingConverter struct {
	delegate  runtime.ObjectConvertor
	validator unstructuredSchemaCoercer
}

var _ runtime.ObjectConvertor = schemaCoercingConverter{}

func (v schemaCoercingConverter) Convert(in, out, context interface{}) error {
	if err := v.delegate.Convert(in, out, context); err != nil {
		return err
	}

	if u, ok := out.(*unstructured.Unstructured); ok {
		if err := v.validator.apply(u); err != nil {
			return err
		}
	}

	return nil
}

func (v schemaCoercingConverter) ConvertToVersion(in runtime.Object, gv runtime.GroupVersioner) (runtime.Object, error) {
	out, err := v.delegate.ConvertToVersion(in, gv)
	if err != nil {
		return nil, err
	}

	if u, ok := out.(*unstructured.Unstructured); ok {
		if err := v.validator.apply(u); err != nil {
			return nil, err
		}
	}

	return out, nil
}

func (v schemaCoercingConverter) ConvertFieldLabel(gvk schema.GroupVersionKind, label, value string) (string, string, error) {
	return v.delegate.ConvertFieldLabel(gvk, label, value)
}

// unstructuredSchemaCoercer does the validation for Unstructured that json.Unmarshal
// does for native types. This includes:
// - validating and pruning ObjectMeta (here with optional error instead of pruning)
// - TODO: application of an OpenAPI validator (against the whole object or a top-level field of it).
// - TODO: optionally application of post-validation algorithms like defaulting and/or OpenAPI based pruning.
type unstructuredSchemaCoercer struct {
	dropInvalidMetadata bool
}

func (v *unstructuredSchemaCoercer) apply(u *unstructured.Unstructured) error {
	// save implicit meta fields that don't have to be specified in the validation spec
	kind, foundKind, err := unstructured.NestedString(u.UnstructuredContent(), "kind")
	if err != nil {
		return err
	}
	apiVersion, foundApiVersion, err := unstructured.NestedString(u.UnstructuredContent(), "apiVersion")
	if err != nil {
		return err
	}
	objectMeta, foundObjectMeta, err := getObjectMeta(u, v.dropInvalidMetadata)
	if err != nil {
		return err
	}

	// restore meta fields, starting clean
	if foundKind {
		u.SetKind(kind)
	}
	if foundApiVersion {
		u.SetAPIVersion(apiVersion)
	}
	if foundObjectMeta {
		if err := setObjectMeta(u, objectMeta); err != nil {
			return err
		}
	}

	return nil
}

var encodingjson = json.CaseSensitiveJsonIterator()

func getObjectMeta(u *unstructured.Unstructured, dropMalformedFields bool) (*metav1.ObjectMeta, bool, error) {
	metadata, found := u.UnstructuredContent()["metadata"]
	if !found {
		return nil, false, nil
	}

	// round-trip through JSON first, hoping that unmarshaling just works
	objectMeta := &metav1.ObjectMeta{}
	metadataBytes, err := encodingjson.Marshal(metadata)
	if err != nil {
		return nil, false, err
	}
	if err = encodingjson.Unmarshal(metadataBytes, objectMeta); err == nil {
		// if successful, return
		return objectMeta, true, nil
	}
	if !dropMalformedFields {
		// if we're not trying to drop malformed fields, return the error
		return nil, true, err
	}

	metadataMap, ok := metadata.(map[string]interface{})
	if !ok {
		return nil, false, fmt.Errorf("invalid metadata: expected object, got %T", metadata)
	}

	// Go field by field accumulating into the metadata object.
	// This takes advantage of the fact that you can repeatedly unmarshal individual fields into a single struct,
	// each iteration preserving the old key-values.
	accumulatedObjectMeta := &metav1.ObjectMeta{}
	testObjectMeta := &metav1.ObjectMeta{}
	for k, v := range metadataMap {
		// serialize a single field
		if singleFieldBytes, err := encodingjson.Marshal(map[string]interface{}{k: v}); err == nil {
			// do a test unmarshal
			if encodingjson.Unmarshal(singleFieldBytes, testObjectMeta) == nil {
				// if that succeeds, unmarshal for real
				encodingjson.Unmarshal(singleFieldBytes, accumulatedObjectMeta)
			}
		}
	}

	return accumulatedObjectMeta, true, nil
}

func setObjectMeta(u *unstructured.Unstructured, objectMeta *metav1.ObjectMeta) error {
	if objectMeta == nil {
		unstructured.RemoveNestedField(u.UnstructuredContent(), "metadata")
		return nil
	}

	metadata, err := runtime.DefaultUnstructuredConverter.ToUnstructured(objectMeta)
	if err != nil {
		return err
	}

	u.UnstructuredContent()["metadata"] = metadata
	return nil
}

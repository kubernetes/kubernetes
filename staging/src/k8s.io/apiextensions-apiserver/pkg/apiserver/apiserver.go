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
	"time"

	kcpapiextensionsv1client "github.com/kcp-dev/client-go/apiextensions/client"
	kcpapiextensionsv1informers "github.com/kcp-dev/client-go/apiextensions/informers"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/install"
	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver/conversion"
	"k8s.io/apiextensions-apiserver/pkg/controller/apiapproval"
	"k8s.io/apiextensions-apiserver/pkg/controller/establish"
	"k8s.io/apiextensions-apiserver/pkg/controller/finalizer"
	"k8s.io/apiextensions-apiserver/pkg/controller/nonstructuralschema"
	openapicontroller "k8s.io/apiextensions-apiserver/pkg/controller/openapi"
	openapiv3controller "k8s.io/apiextensions-apiserver/pkg/controller/openapiv3"
	"k8s.io/apiextensions-apiserver/pkg/controller/status"
	"k8s.io/apiextensions-apiserver/pkg/kcp"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresourcedefinition"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	"k8s.io/apiserver/pkg/endpoints/discovery/aggregated"
	genericregistry "k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
)

var (
	Scheme = runtime.NewScheme()
	Codecs = serializer.NewCodecFactory(Scheme)

	// if you modify this, make sure you update the crEncoder
	unversionedVersion = schema.GroupVersion{Group: "", Version: "v1"}
	unversionedTypes   = []runtime.Object{
		&metav1.Status{},
		&metav1.WatchEvent{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	}
)

func init() {
	install.Install(Scheme)

	// we need to add the options to empty v1
	metav1.AddToGroupVersion(Scheme, schema.GroupVersion{Group: "", Version: "v1"})

	Scheme.AddUnversionedTypes(unversionedVersion, unversionedTypes...)
}

type ExtraConfig struct {
	CRDRESTOptionsGetter genericregistry.RESTOptionsGetter

	// MasterCount is used to detect whether cluster is HA, and if it is
	// the CRD Establishing will be hold by 5 seconds.
	MasterCount int

	// ConversionFactory is used to provider converters for CRs.
	ConversionFactory conversion.Factory

	Client    kcpapiextensionsv1client.ClusterInterface
	Informers kcpapiextensionsv1informers.SharedInformerFactory

	// KCP
	ClusterAwareCRDLister  kcp.ClusterAwareCRDClusterLister
	TableConverterProvider TableConverterProvider
	// DisableServerSideApply deactivates Server Side Apply for a specific API server instead of globally through the feature gate
	// used with the embedded cache server in kcp
	DisableServerSideApply bool
}

type Config struct {
	GenericConfig *genericapiserver.RecommendedConfig
	ExtraConfig   ExtraConfig
}

type completedConfig struct {
	GenericConfig genericapiserver.CompletedConfig
	ExtraConfig   *ExtraConfig
}

type CompletedConfig struct {
	// Embed a private pointer that cannot be instantiated outside of this package.
	*completedConfig
}

type CustomResourceDefinitions struct {
	GenericAPIServer *genericapiserver.GenericAPIServer

	// provided for easier embedding
	Informers kcpapiextensionsv1informers.SharedInformerFactory

	DiscoveryGroupLister discovery.GroupLister

	crdHandler              *crdHandler
	versionDiscoveryHandler *versionDiscoveryHandler
	groupDiscoveryHandler   *groupDiscoveryHandler
	rootDiscoveryHandler    *rootDiscoveryHandler

	// KCP
	ClusterAwareCRDLister kcp.ClusterAwareCRDClusterLister
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (cfg *Config) Complete() CompletedConfig {
	c := completedConfig{
		cfg.GenericConfig.Complete(),
		&cfg.ExtraConfig,
	}

	c.GenericConfig.EnableDiscovery = false

	return CompletedConfig{&c}
}

// New returns a new instance of CustomResourceDefinitions from the given config.
func (c completedConfig) New(delegationTarget genericapiserver.DelegationTarget) (*CustomResourceDefinitions, error) {
	genericServer, err := c.GenericConfig.New("apiextensions-apiserver", delegationTarget)
	if err != nil {
		return nil, err
	}

	// hasCRDInformerSyncedSignal is closed when the CRD informer this server uses has been fully synchronized.
	// It ensures that requests to potential custom resource endpoints while the server hasn't installed all known HTTP paths get a 503 error instead of a 404
	hasCRDInformerSyncedSignal := make(chan struct{})
	if err := genericServer.RegisterMuxAndDiscoveryCompleteSignal("CRDInformerHasNotSynced", hasCRDInformerSyncedSignal); err != nil {
		return nil, err
	}

	s := &CustomResourceDefinitions{
		GenericAPIServer: genericServer,
	}

	apiResourceConfig := c.GenericConfig.MergedResourceConfig
	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiextensions.GroupName, Scheme, metav1.ParameterCodec, Codecs)
	storage := map[string]rest.Storage{}
	// customresourcedefinitions
	if resource := "customresourcedefinitions"; apiResourceConfig.ResourceEnabled(v1.SchemeGroupVersion.WithResource(resource)) {
		customResourceDefinitionStorage, err := customresourcedefinition.NewREST(Scheme, c.GenericConfig.RESTOptionsGetter)
		if err != nil {
			return nil, err
		}
		storage[resource] = customResourceDefinitionStorage
		storage[resource+"/status"] = customresourcedefinition.NewStatusREST(Scheme, customResourceDefinitionStorage)
	}
	if len(storage) > 0 {
		apiGroupInfo.VersionedResourcesStorageMap[v1.SchemeGroupVersion.Version] = storage
	}

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	crdClient := c.ExtraConfig.Client
	if crdClient == nil {
		crdClient, err = kcpapiextensionsv1client.NewForConfig(s.GenericAPIServer.LoopbackClientConfig)
		if err != nil {
			// it's really bad that this is leaking here, but until we can fix the test (which I'm pretty sure isn't even testing what it wants to test),
			// we need to be able to move forward
			return nil, fmt.Errorf("failed to create clientset: %v", err)
		}
	}

	s.Informers = c.ExtraConfig.Informers
	if s.Informers == nil {
		s.Informers = kcpapiextensionsv1informers.NewSharedInformerFactory(crdClient, 5*time.Minute)
	}

	s.ClusterAwareCRDLister = c.ExtraConfig.ClusterAwareCRDLister

	delegateHandler := delegationTarget.UnprotectedHandler()
	if delegateHandler == nil {
		delegateHandler = http.NotFoundHandler()
	}

	s.versionDiscoveryHandler = &versionDiscoveryHandler{
		crdLister: c.ExtraConfig.ClusterAwareCRDLister,
		delegate:  delegateHandler,
	}

	s.groupDiscoveryHandler = &groupDiscoveryHandler{
		crdLister: c.ExtraConfig.ClusterAwareCRDLister,
		delegate:  delegateHandler,
	}

	s.rootDiscoveryHandler = &rootDiscoveryHandler{
		crdLister: c.ExtraConfig.ClusterAwareCRDLister,
		delegate:  delegateHandler,
	}
	s.DiscoveryGroupLister = s.rootDiscoveryHandler

	establishingController := establish.NewEstablishingController(s.Informers.Apiextensions().V1().CustomResourceDefinitions(), crdClient.ApiextensionsV1())

	crdHandler, err := NewCustomResourceDefinitionHandler(
		s.versionDiscoveryHandler,
		s.groupDiscoveryHandler,
		s.Informers.Apiextensions().V1().CustomResourceDefinitions(),
		delegateHandler,
		c.ExtraConfig.CRDRESTOptionsGetter,
		c.GenericConfig.AdmissionControl,
		establishingController,
		c.ExtraConfig.ConversionFactory,
		c.ExtraConfig.MasterCount,
		s.GenericAPIServer.Authorizer,
		c.GenericConfig.RequestTimeout,
		time.Duration(c.GenericConfig.MinRequestTimeout)*time.Second,
		apiGroupInfo.StaticOpenAPISpec,
		c.GenericConfig.MaxRequestBodyBytes,
		c.ExtraConfig.DisableServerSideApply,
	)
	if err != nil {
		return nil, err
	}
	s.crdHandler = crdHandler

	// Begin kcp additions
	crdHandler.clusterAwareCRDLister = c.ExtraConfig.ClusterAwareCRDLister
	crdHandler.tableConverterProvider = c.ExtraConfig.TableConverterProvider
	// End kcp additions

	s.GenericAPIServer.Handler.NonGoRestfulMux.Handle("/apis", crdHandler)
	s.GenericAPIServer.Handler.NonGoRestfulMux.HandlePrefix("/apis/", crdHandler)
	s.GenericAPIServer.RegisterDestroyFunc(crdHandler.destroy)

	aggregatedDiscoveryManager := genericServer.AggregatedDiscoveryGroupManager
	if aggregatedDiscoveryManager != nil {
		aggregatedDiscoveryManager = aggregatedDiscoveryManager.WithSource(aggregated.CRDSource)
	}
	// HACK: Added to allow serving core resources registered through CRDs (for the KCP scenario)
	s.GenericAPIServer.Handler.NonGoRestfulMux.UnlistedHandlePrefix("/api/v1/", crdHandler)

	namingController := status.NewNamingConditionController(s.Informers.Apiextensions().V1().CustomResourceDefinitions(), crdClient.ApiextensionsV1(), s.ClusterAwareCRDLister)
	nonStructuralSchemaController := nonstructuralschema.NewConditionController(s.Informers.Apiextensions().V1().CustomResourceDefinitions(), crdClient.ApiextensionsV1())
	apiApprovalController := apiapproval.NewKubernetesAPIApprovalPolicyConformantConditionController(s.Informers.Apiextensions().V1().CustomResourceDefinitions(), crdClient.ApiextensionsV1())
	finalizingController := finalizer.NewCRDFinalizer(
		s.Informers.Apiextensions().V1().CustomResourceDefinitions(),
		crdClient.ApiextensionsV1(),
		crdHandler,
	)

	s.GenericAPIServer.AddPostStartHookOrDie("start-apiextensions-informers", func(context genericapiserver.PostStartHookContext) error {
		s.Informers.Start(context.Done())
		return nil
	})
	s.GenericAPIServer.AddPostStartHookOrDie("start-apiextensions-controllers", func(context genericapiserver.PostStartHookContext) error {
		// OpenAPIVersionedService and StaticOpenAPISpec are populated in generic apiserver PrepareRun().
		// Together they serve the /openapi/v2 endpoint on a generic apiserver. A generic apiserver may
		// choose to not enable OpenAPI by having null openAPIConfig, and thus OpenAPIVersionedService
		// and StaticOpenAPISpec are both null. In that case we don't run the CRD OpenAPI controller.
		if s.GenericAPIServer.StaticOpenAPISpec != nil {
			if s.GenericAPIServer.OpenAPIVersionedService != nil {
				openapiController := openapicontroller.NewController(s.Informers.Apiextensions().V1().CustomResourceDefinitions())
				go openapiController.Run(s.GenericAPIServer.StaticOpenAPISpec, s.GenericAPIServer.OpenAPIVersionedService, context.Done())
			}

			if s.GenericAPIServer.OpenAPIV3VersionedService != nil {
				openapiv3Controller := openapiv3controller.NewController(s.Informers.Apiextensions().V1().CustomResourceDefinitions())
				go openapiv3Controller.Run(s.GenericAPIServer.OpenAPIV3VersionedService, context.Done())
			}
		}

		go namingController.Run(context.Done())
		go establishingController.Run(context.Done())
		go nonStructuralSchemaController.Run(5, context.Done())
		go apiApprovalController.Run(5, context.Done())
		go finalizingController.Run(5, context.Done())

		return nil
	})
	// we don't want to report healthy until we can handle all CRDs that have already been registered.  Waiting for the informer
	// to sync makes sure that the lister will be valid before we begin.  There may still be races for CRDs added after startup,
	// but we won't go healthy until we can handle the ones already present.
	s.GenericAPIServer.AddPostStartHookOrDie("crd-informer-synced", func(context genericapiserver.PostStartHookContext) error {
		return wait.PollImmediateUntil(100*time.Millisecond, func() (bool, error) {
			if s.Informers.Apiextensions().V1().CustomResourceDefinitions().Informer().HasSynced() {
				close(hasCRDInformerSyncedSignal)
				return true, nil
			}
			return false, nil
		}, context.Done())
	})

	return s, nil
}

func DefaultAPIResourceConfigSource() *serverstorage.ResourceConfig {
	ret := serverstorage.NewResourceConfig()
	// NOTE: GroupVersions listed here will be enabled by default. Don't put alpha versions in the list.
	ret.EnableVersions(
		v1beta1.SchemeGroupVersion,
		v1.SchemeGroupVersion,
	)

	return ret
}

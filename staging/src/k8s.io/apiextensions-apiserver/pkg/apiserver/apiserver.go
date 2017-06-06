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
	"os"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/apimachinery/announced"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/endpoints/discovery"
	genericregistry "k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/install"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset"
	internalinformers "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion"
	"k8s.io/apiextensions-apiserver/pkg/controller/finalizer"
	"k8s.io/apiextensions-apiserver/pkg/controller/status"
	"k8s.io/apiextensions-apiserver/pkg/registry/customresourcedefinition"

	// make sure the generated client works
	_ "k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	_ "k8s.io/apiextensions-apiserver/pkg/client/informers/externalversions"
	_ "k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion"
)

var (
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	registry             = registered.NewOrDie("")
	Scheme               = runtime.NewScheme()
	Codecs               = serializer.NewCodecFactory(Scheme)

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
	install.Install(groupFactoryRegistry, registry, Scheme)

	// we need to add the options to empty v1
	metav1.AddToGroupVersion(Scheme, schema.GroupVersion{Group: "", Version: "v1"})

	Scheme.AddUnversionedTypes(unversionedVersion, unversionedTypes...)
}

type Config struct {
	GenericConfig *genericapiserver.Config

	CRDRESTOptionsGetter genericregistry.RESTOptionsGetter
}

type CustomResourceDefinitions struct {
	GenericAPIServer *genericapiserver.GenericAPIServer

	// provided for easier embedding
	Informers internalinformers.SharedInformerFactory
}

type completedConfig struct {
	*Config
}

// Complete fills in any fields not set that are required to have valid data. It's mutating the receiver.
func (c *Config) Complete() completedConfig {
	c.GenericConfig.EnableDiscovery = false
	c.GenericConfig.Complete()

	c.GenericConfig.Version = &version.Info{
		Major: "0",
		Minor: "1",
	}

	return completedConfig{c}
}

// SkipComplete provides a way to construct a server instance without config completion.
func (c *Config) SkipComplete() completedConfig {
	return completedConfig{c}
}

// New returns a new instance of CustomResourceDefinitions from the given config.
func (c completedConfig) New(delegationTarget genericapiserver.DelegationTarget) (*CustomResourceDefinitions, error) {
	genericServer, err := c.Config.GenericConfig.SkipComplete().New("apiextensions-apiserver", delegationTarget) // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	s := &CustomResourceDefinitions{
		GenericAPIServer: genericServer,
	}

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiextensions.GroupName, registry, Scheme, metav1.ParameterCodec, Codecs)
	apiGroupInfo.GroupMeta.GroupVersion = v1beta1.SchemeGroupVersion
	customResourceDefintionStorage := customresourcedefinition.NewREST(Scheme, c.GenericConfig.RESTOptionsGetter)
	v1beta1storage := map[string]rest.Storage{}
	v1beta1storage["customresourcedefinitions"] = customResourceDefintionStorage
	v1beta1storage["customresourcedefinitions/status"] = customresourcedefinition.NewStatusREST(Scheme, customResourceDefintionStorage)
	apiGroupInfo.VersionedResourcesStorageMap["v1beta1"] = v1beta1storage

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	crdClient, err := internalclientset.NewForConfig(s.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		// it's really bad that this is leaking here, but until we can fix the test (which I'm pretty sure isn't even testing what it wants to test),
		// we need to be able to move forward
		kubeAPIVersions := os.Getenv("KUBE_API_VERSIONS")
		if len(kubeAPIVersions) == 0 {
			return nil, fmt.Errorf("failed to create clientset: %v", err)
		}

		// KUBE_API_VERSIONS is used in test-update-storage-objects.sh, disabling a number of API
		// groups. This leads to a nil client above and undefined behaviour further down.
		//
		// TODO: get rid of KUBE_API_VERSIONS or define sane behaviour if set
		glog.Errorf("Failed to create clientset with KUBE_API_VERSIONS=%q. KUBE_API_VERSIONS is only for testing. Things will break.", kubeAPIVersions)
	}
	s.Informers = internalinformers.NewSharedInformerFactory(crdClient, 5*time.Minute)

	delegateHandler := delegationTarget.UnprotectedHandler()
	if delegateHandler == nil {
		delegateHandler = http.NotFoundHandler()
	}

	versionDiscoveryHandler := &versionDiscoveryHandler{
		discovery: map[schema.GroupVersion]*discovery.APIVersionHandler{},
		delegate:  delegateHandler,
	}
	groupDiscoveryHandler := &groupDiscoveryHandler{
		discovery: map[string]*discovery.APIGroupHandler{},
		delegate:  delegateHandler,
	}
	crdHandler := NewCustomResourceDefinitionHandler(
		versionDiscoveryHandler,
		groupDiscoveryHandler,
		s.GenericAPIServer.RequestContextMapper(),
		s.Informers.Apiextensions().InternalVersion().CustomResourceDefinitions().Lister(),
		delegateHandler,
		c.CRDRESTOptionsGetter,
		c.GenericConfig.AdmissionControl,
	)
	s.GenericAPIServer.Handler.NonGoRestfulMux.Handle("/apis", crdHandler)
	s.GenericAPIServer.Handler.NonGoRestfulMux.HandlePrefix("/apis/", crdHandler)

	crdController := NewDiscoveryController(s.Informers.Apiextensions().InternalVersion().CustomResourceDefinitions(), versionDiscoveryHandler, groupDiscoveryHandler, c.GenericConfig.RequestContextMapper)
	namingController := status.NewNamingConditionController(s.Informers.Apiextensions().InternalVersion().CustomResourceDefinitions(), crdClient)
	finalizingController := finalizer.NewCRDFinalizer(
		s.Informers.Apiextensions().InternalVersion().CustomResourceDefinitions(),
		crdClient,
		crdHandler,
	)

	// this only happens when KUBE_API_VERSIONS is set.  We must return without adding poststarthooks which would affect healthz
	if crdClient == nil {
		return s, nil
	}

	s.GenericAPIServer.AddPostStartHook("start-apiextensions-informers", func(context genericapiserver.PostStartHookContext) error {
		s.Informers.Start(context.StopCh)
		return nil
	})
	s.GenericAPIServer.AddPostStartHook("start-apiextensions-controllers", func(context genericapiserver.PostStartHookContext) error {
		go crdController.Run(context.StopCh)
		go namingController.Run(context.StopCh)
		go finalizingController.Run(5, context.StopCh)
		return nil
	})

	return s, nil
}

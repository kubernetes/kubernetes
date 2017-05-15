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
	"net/http"
	"time"

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

	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions"
	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/install"
	"k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1"
	"k8s.io/kube-apiextensions-server/pkg/client/clientset/internalclientset"
	internalinformers "k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion"
	"k8s.io/kube-apiextensions-server/pkg/controller/status"
	"k8s.io/kube-apiextensions-server/pkg/registry/customresourcedefinition"

	// make sure the generated client works
	_ "k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset"
	_ "k8s.io/kube-apiextensions-server/pkg/client/informers/externalversions"
	_ "k8s.io/kube-apiextensions-server/pkg/client/informers/internalversion"
)

var (
	groupFactoryRegistry = make(announced.APIGroupFactoryRegistry)
	registry             = registered.NewOrDie("")
	Scheme               = runtime.NewScheme()
	Codecs               = serializer.NewCodecFactory(Scheme)
)

func init() {
	install.Install(groupFactoryRegistry, registry, Scheme)

	// we need to add the options to empty v1
	metav1.AddToGroupVersion(Scheme, schema.GroupVersion{Group: "", Version: "v1"})

	unversioned := schema.GroupVersion{Group: "", Version: "v1"}
	Scheme.AddUnversionedTypes(unversioned,
		&metav1.Status{},
		&metav1.APIVersions{},
		&metav1.APIGroupList{},
		&metav1.APIGroup{},
		&metav1.APIResourceList{},
	)
}

type Config struct {
	GenericConfig *genericapiserver.Config

	CustomResourceDefinitionRESTOptionsGetter genericregistry.RESTOptionsGetter
}

type CustomResourceDefinitions struct {
	GenericAPIServer *genericapiserver.GenericAPIServer
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
	genericServer, err := c.Config.GenericConfig.SkipComplete().New(genericapiserver.EmptyDelegate) // completion is done in Complete, no need for a second time
	if err != nil {
		return nil, err
	}

	s := &CustomResourceDefinitions{
		GenericAPIServer: genericServer,
	}

	apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(apiextensions.GroupName, registry, Scheme, metav1.ParameterCodec, Codecs)
	apiGroupInfo.GroupMeta.GroupVersion = v1alpha1.SchemeGroupVersion
	customResourceDefintionStorage := customresourcedefinition.NewREST(Scheme, c.GenericConfig.RESTOptionsGetter)
	v1alpha1storage := map[string]rest.Storage{}
	v1alpha1storage["customresourcedefinitions"] = customResourceDefintionStorage
	v1alpha1storage["customresourcedefinitions/status"] = customresourcedefinition.NewStatusREST(Scheme, customResourceDefintionStorage)
	apiGroupInfo.VersionedResourcesStorageMap["v1alpha1"] = v1alpha1storage

	if err := s.GenericAPIServer.InstallAPIGroup(&apiGroupInfo); err != nil {
		return nil, err
	}

	customResourceDefinitionClient, err := internalclientset.NewForConfig(s.GenericAPIServer.LoopbackClientConfig)
	if err != nil {
		return nil, err
	}
	customResourceDefinitionInformers := internalinformers.NewSharedInformerFactory(customResourceDefinitionClient, 5*time.Minute)

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
	customResourceDefinitionHandler := NewCustomResourceDefinitionHandler(
		versionDiscoveryHandler,
		groupDiscoveryHandler,
		s.GenericAPIServer.RequestContextMapper(),
		customResourceDefinitionInformers.Apiextensions().InternalVersion().CustomResourceDefinitions().Lister(),
		delegateHandler,
		c.CustomResourceDefinitionRESTOptionsGetter,
		c.GenericConfig.AdmissionControl,
	)
	s.GenericAPIServer.Handler.PostGoRestfulMux.Handle("/apis", customResourceDefinitionHandler)
	s.GenericAPIServer.Handler.PostGoRestfulMux.HandlePrefix("/apis/", customResourceDefinitionHandler)

	customResourceDefinitionController := NewDiscoveryController(customResourceDefinitionInformers.Apiextensions().InternalVersion().CustomResourceDefinitions(), versionDiscoveryHandler, groupDiscoveryHandler)
	namingController := status.NewNamingConditionController(customResourceDefinitionInformers.Apiextensions().InternalVersion().CustomResourceDefinitions(), customResourceDefinitionClient)

	s.GenericAPIServer.AddPostStartHook("start-apiextensions-informers", func(context genericapiserver.PostStartHookContext) error {
		customResourceDefinitionInformers.Start(context.StopCh)
		return nil
	})
	s.GenericAPIServer.AddPostStartHook("start-apiextensions-controllers", func(context genericapiserver.PostStartHookContext) error {
		go customResourceDefinitionController.Run(context.StopCh)
		go namingController.Run(context.StopCh)
		return nil
	})

	return s, nil
}

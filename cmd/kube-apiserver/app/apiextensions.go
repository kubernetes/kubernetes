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

// Package app does all of the work necessary to create a Kubernetes
// APIServer by binding together the API, master and APIServer infrastructure.
// It can be configured and called directly or via the hyperkube framework.
package app

import (
	"github.com/go-openapi/spec"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	apiextensionsapiserver "k8s.io/apiextensions-apiserver/pkg/apiserver"
	apiextensionsoptions "k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/features"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/apiserver/pkg/util/webhook"
)

func createAPIExtensionsConfig(
	commandOptions completedServerRunOptions,
	baseConfig genericapiserver.RecommendedConfig,
	pluginInitializers []admission.PluginInitializer,
	masterCount int,
	serviceResolver webhook.ServiceResolver,
	authResolverWrapper webhook.AuthenticationInfoResolverWrapper,
	securityDefinitions *spec.SecurityDefinitions,
) (*apiextensionsapiserver.Config, error) {
	// override etcd options with apiextensions-server based scheme
	etcdOptions := *commandOptions.Etcd // copy the etcd options so we don't mutate originals
	etcdOptions.StorageConfig.Paging = utilfeature.DefaultFeatureGate.Enabled(features.APIListChunking)
	etcdOptions.StorageConfig.Codec = apiextensionsapiserver.Codecs.LegacyCodec(v1beta1.SchemeGroupVersion, v1.SchemeGroupVersion)
	etcdOptions.StorageConfig.EncodeVersioner = runtime.NewMultiGroupVersioner(v1beta1.SchemeGroupVersion, schema.GroupKind{Group: v1beta1.GroupName})

	c := apiextensionsapiserver.NewConfig(baseConfig, apiextensionsapiserver.ExtraConfig{
		CRDRESTOptionsGetter: apiextensionsoptions.NewCRDRESTOptionsGetter(etcdOptions),
		MasterCount:          masterCount,
		AuthResolverWrapper:  authResolverWrapper,
		ServiceResolver:      serviceResolver,
	})
	c.GenericConfig.RESTOptionsGetter = &genericoptions.SimpleRestOptionsFactory{Options: etcdOptions}
	c.GenericConfig.OpenAPIConfig.SecurityDefinitions = securityDefinitions

	// override MergedResourceConfig with apiextensions defaults and registry
	if err := commandOptions.APIEnablement.ApplyTo(
		&c.GenericConfig.Config,
		apiextensionsapiserver.DefaultAPIResourceConfigSource(),
		apiextensionsapiserver.Scheme); err != nil {
		return nil, err
	}

	// override genericConfig.AdmissionControl with apiextensions' scheme,
	// because apiextentions apiserver should use its own scheme to convert resources.
	err := commandOptions.Admission.ApplyTo(
		&c.GenericConfig.Config,
		c.GenericConfig.SharedInformerFactory,
		c.GenericConfig.LoopbackClientConfig,
		pluginInitializers...)
	if err != nil {
		return nil, err
	}

	return c, nil
}

func createAPIExtensionsServer(apiextensionsConfig *apiextensionsapiserver.Config, delegateAPIServer genericapiserver.DelegationTarget) (*apiextensionsapiserver.CustomResourceDefinitions, error) {
	return apiextensionsConfig.Complete().New(delegateAPIServer)
}

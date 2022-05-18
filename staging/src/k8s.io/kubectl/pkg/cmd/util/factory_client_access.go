/*
Copyright 2016 The Kubernetes Authors.

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

// this file contains factories with no other dependencies

package util

import (
	"errors"
	"sync"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/resource"
	"k8s.io/client-go/discovery"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubectl/pkg/util/openapi"
	openapivalidation "k8s.io/kubectl/pkg/util/openapi/validation"
	"k8s.io/kubectl/pkg/validation"
)

type factoryImpl struct {
	clientGetter genericclioptions.RESTClientGetter

	// Caches OpenAPI document and parsed resources
	openAPIParser *openapi.CachedOpenAPIParser
	openAPIGetter *openapi.CachedOpenAPIGetter
	parser        sync.Once
	getter        sync.Once
}

func NewFactory(clientGetter genericclioptions.RESTClientGetter) Factory {
	if clientGetter == nil {
		panic("attempt to instantiate client_access_factory with nil clientGetter")
	}
	f := &factoryImpl{
		clientGetter: clientGetter,
	}

	return f
}

func (f *factoryImpl) ToRESTConfig() (*restclient.Config, error) {
	return f.clientGetter.ToRESTConfig()
}

func (f *factoryImpl) ToRESTMapper() (meta.RESTMapper, error) {
	return f.clientGetter.ToRESTMapper()
}

func (f *factoryImpl) ToDiscoveryClient() (discovery.CachedDiscoveryInterface, error) {
	return f.clientGetter.ToDiscoveryClient()
}

func (f *factoryImpl) ToRawKubeConfigLoader() clientcmd.ClientConfig {
	return f.clientGetter.ToRawKubeConfigLoader()
}

func (f *factoryImpl) KubernetesClientSet() (*kubernetes.Clientset, error) {
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	return kubernetes.NewForConfig(clientConfig)
}

func (f *factoryImpl) DynamicClient() (dynamic.Interface, error) {
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	return dynamic.NewForConfig(clientConfig)
}

// NewBuilder returns a new resource builder for structured api objects.
func (f *factoryImpl) NewBuilder() *resource.Builder {
	return resource.NewBuilder(f.clientGetter)
}

func (f *factoryImpl) RESTClient() (*restclient.RESTClient, error) {
	clientConfig, err := f.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	setKubernetesDefaults(clientConfig)
	return restclient.RESTClientFor(clientConfig)
}

func (f *factoryImpl) ClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	if err := setKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	gvk := mapping.GroupVersionKind
	switch gvk.Group {
	case corev1.GroupName:
		cfg.APIPath = "/api"
	default:
		cfg.APIPath = "/apis"
	}
	gv := gvk.GroupVersion()
	cfg.GroupVersion = &gv
	return restclient.RESTClientFor(cfg)
}

func (f *factoryImpl) UnstructuredClientForMapping(mapping *meta.RESTMapping) (resource.RESTClient, error) {
	cfg, err := f.clientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}
	if err := restclient.SetKubernetesDefaults(cfg); err != nil {
		return nil, err
	}
	cfg.APIPath = "/apis"
	if mapping.GroupVersionKind.Group == corev1.GroupName {
		cfg.APIPath = "/api"
	}
	gv := mapping.GroupVersionKind.GroupVersion()
	cfg.ContentConfig = resource.UnstructuredPlusDefaultContentConfig()
	cfg.GroupVersion = &gv
	return restclient.RESTClientFor(cfg)
}

func (f *factoryImpl) Validator(validationDirective string, verifier *resource.QueryParamVerifier) (validation.Schema, error) {
	// client-side schema validation is only performed
	// when the validationDirective is strict.
	// If the directive is warn, we rely on the ParamVerifyingSchema
	// to ignore the client-side validation and provide a warning
	// to the user that attempting warn validation when SS validation
	// is unsupported is inert.
	if validationDirective == metav1.FieldValidationIgnore {
		return validation.NullSchema{}, nil
	}

	resources, err := f.OpenAPISchema()
	if err != nil {
		return nil, err
	}

	schema := validation.ConjunctiveSchema{
		openapivalidation.NewSchemaValidation(resources),
		validation.NoDoubleKeySchema{},
	}
	return validation.NewParamVerifyingSchema(schema, verifier, string(validationDirective)), nil
}

// OpenAPISchema returns metadata and structural information about
// Kubernetes object definitions.
func (f *factoryImpl) OpenAPISchema() (openapi.Resources, error) {
	openAPIGetter := f.OpenAPIGetter()
	if openAPIGetter == nil {
		return nil, errors.New("no openapi getter")
	}

	// Lazily initialize the OpenAPIParser once
	f.parser.Do(func() {
		// Create the caching OpenAPIParser
		f.openAPIParser = openapi.NewOpenAPIParser(f.OpenAPIGetter())
	})

	// Delegate to the OpenAPIPArser
	return f.openAPIParser.Parse()
}

func (f *factoryImpl) OpenAPIGetter() discovery.OpenAPISchemaInterface {
	discovery, err := f.clientGetter.ToDiscoveryClient()
	if err != nil {
		return nil
	}
	f.getter.Do(func() {
		f.openAPIGetter = openapi.NewOpenAPIGetter(discovery)
	})

	return f.openAPIGetter
}

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
	"context"
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
	openapiclient "k8s.io/client-go/openapi"
	"k8s.io/client-go/openapi/cached"
	"k8s.io/client-go/openapi3"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/kubectl/pkg/util/openapi"
	"k8s.io/kubectl/pkg/validation"
)

type factoryImpl struct {
	clientGetter genericclioptions.RESTClientGetter

	// Caches OpenAPI document and parsed resources
	openAPIParser *openapi.CachedOpenAPIParser
	oapi          *openapi.CachedOpenAPIGetter
	parser        sync.Once
	getter        sync.Once

	// Caches the OpenAPI V3 client, so that every command sharing this
	// factory downloads each schema once and parses it once.
	oapiV3   openapiclient.Client
	v3Getter sync.Once
}

// openAPIV3Client is a cached OpenAPI V3 client paired with the parse cache
// that every openapi3.Root built from it should use. Callers each construct
// their own Root from the factory client, so carrying the cache on the client
// is what lets them share parsed documents without changing the Factory
// interface or any call site.
type openAPIV3Client struct {
	openapiclient.Client

	withContext openapiclient.ClientWithContext
	cache       *openapi3.Cache
}

var (
	//nolint:staticcheck // Intentionally using the old interface here.
	_ openapiclient.Client            = &openAPIV3Client{}
	_ openapiclient.ClientWithContext = &openAPIV3Client{}
	_ openapi3.CacheProvider          = &openAPIV3Client{}
)

// PathsWithContext keeps the delegate contextual logging and cancellation
// support, which plain embedding of the older interface would hide.
func (c *openAPIV3Client) PathsWithContext(ctx context.Context) (map[string]openapiclient.GroupVersionWithContext, error) {
	return c.withContext.PathsWithContext(ctx)
}

// OpenAPIV3Cache implements openapi3.CacheProvider.
func (c *openAPIV3Client) OpenAPIV3Cache() *openapi3.Cache {
	return c.cache
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

func (f *factoryImpl) Validator(validationDirective string) (validation.Schema, error) {
	// client-side schema validation is only performed
	// when the validationDirective is strict.
	// If the directive is warn, we rely on the ParamVerifyingSchema
	// to ignore the client-side validation and provide a warning
	// to the user that attempting warn validation when SS validation
	// is unsupported is inert.
	if validationDirective == metav1.FieldValidationIgnore {
		return validation.NullSchema{}, nil
	}

	// Client-side validation is limited to checks that need no schema. The
	// server validates the fields; if it cannot (fieldValidation unsupported),
	// the ParamVerifyingSchema below warns or runs this schema instead of the
	// former OpenAPI v2 schema validation.
	schema := validation.NoDoubleKeySchema{}

	dynamicClient, err := f.DynamicClient()
	if err != nil {
		return nil, err
	}
	// Create the FieldValidationVerifier for use in the ParamVerifyingSchema.
	// Reuse the factory OpenAPI V3 client rather than building a second one,
	// so that field validation, apply and diff download each schema once and
	// unmarshal it once between them. Memory caching of the responses comes
	// from that client; the disk cache behavior is determined by the
	// discovery client underneath it.
	oapiV3Client, err := f.OpenAPIV3Client()
	if err != nil {
		return nil, err
	}
	queryParam := resource.QueryParamFieldValidation
	verifier := newCachingVerifier(resource.NewQueryParamVerifierV3(dynamicClient, oapiV3Client, queryParam))
	return validation.NewParamVerifyingSchema(schema, verifier, string(validationDirective)), nil
}

// OpenAPISchema returns metadata and structural information about
// Kubernetes object definitions.
func (f *factoryImpl) OpenAPISchema() (openapi.Resources, error) {
	openAPIGetter := f.openAPIGetter()
	if openAPIGetter == nil {
		return nil, errors.New("no openapi getter")
	}

	// Lazily initialize the OpenAPIParser once
	f.parser.Do(func() {
		// Create the caching OpenAPIParser
		f.openAPIParser = openapi.NewOpenAPIParser(f.openAPIGetter())
	})

	// Delegate to the OpenAPIPArser
	return f.openAPIParser.Parse()
}

func (f *factoryImpl) openAPIGetter() discovery.OpenAPISchemaInterface {
	discovery, err := f.clientGetter.ToDiscoveryClient()
	if err != nil {
		return nil
	}
	f.getter.Do(func() {
		f.oapi = openapi.NewOpenAPIGetter(discovery)
	})

	return f.oapi
}

func (f *factoryImpl) OpenAPIV3Client() (openapiclient.Client, error) {
	discovery, err := f.clientGetter.ToDiscoveryClient()
	if err != nil {
		return nil, err
	}

	// Build the client once per factory. Returning the same client to every
	// caller means the schema bytes are downloaded once, and the parse cache
	// it carries means each schema is unmarshaled once, however many
	// commands build an openapi3.Root from it.
	f.v3Getter.Do(func() {
		delegate := cached.NewClient(discovery.OpenAPIV3())
		f.oapiV3 = &openAPIV3Client{
			Client:      delegate,
			withContext: openapiclient.ToClientWithContext(delegate),
			cache:       openapi3.NewCache(),
		}
	})

	return f.oapiV3, nil
}

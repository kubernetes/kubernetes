/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"io"
	"net"
	"net/url"
	"sort"
	"strings"

	"github.com/spf13/pflag"
	noopoteltrace "go.opentelemetry.io/otel/trace/noop"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/apiserver"
	generatedopenapi "k8s.io/apiextensions-apiserver/pkg/generated/openapi"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured/unstructuredscheme"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	openapinamer "k8s.io/apiserver/pkg/endpoints/openapi"
	"k8s.io/apiserver/pkg/features"
	genericregistry "k8s.io/apiserver/pkg/registry/generic"
	genericapiserver "k8s.io/apiserver/pkg/server"
	genericoptions "k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	storagevalue "k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	flowcontrolrequest "k8s.io/apiserver/pkg/util/flowcontrol/request"
	"k8s.io/apiserver/pkg/util/openapi"
	"k8s.io/apiserver/pkg/util/proxy"
	"k8s.io/apiserver/pkg/util/webhook"
	scheme "k8s.io/client-go/kubernetes/scheme"
	corev1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/klog/v2"
	netutils "k8s.io/utils/net"
)

const defaultEtcdPathPrefix = "/registry/apiextensions.kubernetes.io"

// CustomResourceDefinitionsServerOptions describes the runtime options of an apiextensions-apiserver.
type CustomResourceDefinitionsServerOptions struct {
	ServerRunOptions   *genericoptions.ServerRunOptions
	RecommendedOptions *genericoptions.RecommendedOptions
	APIEnablement      *genericoptions.APIEnablementOptions

	StdOut io.Writer
	StdErr io.Writer
}

// NewCustomResourceDefinitionsServerOptions creates default options of an apiextensions-apiserver.
func NewCustomResourceDefinitionsServerOptions(out, errOut io.Writer) *CustomResourceDefinitionsServerOptions {
	o := &CustomResourceDefinitionsServerOptions{
		ServerRunOptions: genericoptions.NewServerRunOptions(),
		RecommendedOptions: genericoptions.NewRecommendedOptions(
			defaultEtcdPathPrefix,
			apiserver.Codecs.LegacyCodec(v1beta1.SchemeGroupVersion, v1.SchemeGroupVersion),
		),
		APIEnablement: genericoptions.NewAPIEnablementOptions(),

		StdOut: out,
		StdErr: errOut,
	}

	return o
}

// AddFlags adds the apiextensions-apiserver flags to the flagset.
func (o CustomResourceDefinitionsServerOptions) AddFlags(fs *pflag.FlagSet) {
	o.ServerRunOptions.AddUniversalFlags(fs)
	o.RecommendedOptions.AddFlags(fs)
	o.APIEnablement.AddFlags(fs)
}

// Validate validates the apiextensions-apiserver options.
func (o CustomResourceDefinitionsServerOptions) Validate() error {
	errors := []error{}
	errors = append(errors, o.ServerRunOptions.Validate()...)
	errors = append(errors, o.RecommendedOptions.Validate()...)
	errors = append(errors, o.APIEnablement.Validate(apiserver.Scheme)...)
	return utilerrors.NewAggregate(errors)
}

// Complete fills in missing options.
func (o *CustomResourceDefinitionsServerOptions) Complete() error {
	return o.ServerRunOptions.Complete()
}

// Config returns an apiextensions-apiserver configuration.
func (o CustomResourceDefinitionsServerOptions) Config() (*apiserver.Config, error) {
	// TODO have a "real" external address
	if err := o.RecommendedOptions.SecureServing.MaybeDefaultWithSelfSignedCerts("localhost", nil, []net.IP{netutils.ParseIPSloppy("127.0.0.1")}); err != nil {
		return nil, fmt.Errorf("error creating self-signed certificates: %v", err)
	}

	serverConfig := genericapiserver.NewRecommendedConfig(apiserver.Codecs)
	if err := o.ServerRunOptions.ApplyTo(&serverConfig.Config); err != nil {
		return nil, err
	}
	if err := o.RecommendedOptions.ApplyTo(serverConfig); err != nil {
		return nil, err
	}
	if err := o.APIEnablement.ApplyTo(&serverConfig.Config, apiserver.DefaultAPIResourceConfigSource(), apiserver.Scheme); err != nil {
		return nil, err
	}

	serverConfig.OpenAPIV3Config = genericapiserver.DefaultOpenAPIV3Config(openapi.GetOpenAPIDefinitionsWithoutDisabledFeatures(generatedopenapi.GetOpenAPIDefinitions), openapinamer.NewDefinitionNamer(apiserver.Scheme, scheme.Scheme))
	config := &apiserver.Config{
		GenericConfig: serverConfig,
		ExtraConfig: apiserver.ExtraConfig{
			CRDRESTOptionsGetter: NewCRDRESTOptionsGetter(*o.RecommendedOptions.Etcd, serverConfig.ResourceTransformers, serverConfig.StorageObjectCountTracker),
			ServiceResolver:      &serviceResolver{serverConfig.SharedInformerFactory.Core().V1().Services().Lister()},
			AuthResolverWrapper:  webhook.NewDefaultAuthenticationInfoResolverWrapper(nil, nil, serverConfig.LoopbackClientConfig, noopoteltrace.NewTracerProvider()),
		},
	}
	return config, nil
}

// NewCRDRESTOptionsGetter create a RESTOptionsGetter for CustomResources.
//
// Avoid messing with anything outside of changes to StorageConfig as that
// may lead to unexpected behavior when the options are applied.
func NewCRDRESTOptionsGetter(etcdOptions genericoptions.EtcdOptions, resourceTransformers storagevalue.ResourceTransformers, tracker flowcontrolrequest.StorageObjectCountTracker) genericregistry.RESTOptionsGetter {
	ucbor := cbor.NewSerializer(unstructuredscheme.NewUnstructuredCreator(), unstructuredscheme.NewUnstructuredObjectTyper())

	encoder := unstructured.UnstructuredJSONScheme
	if utilfeature.DefaultFeatureGate.Enabled(features.CBORServingAndStorage) {
		encoder = ucbor
	}

	etcdOptionsCopy := etcdOptions
	etcdOptionsCopy.StorageConfig.Codec = runtime.NewCodec(
		encoder,
		// Whether the feature gate is enabled or disabled, the decoder must be able to
		// recognize any resources stored using the CBOR encoder.
		recognizer.NewDecoder(
			ucbor,
			unstructured.UnstructuredJSONScheme,
		),
	)
	etcdOptionsCopy.StorageConfig.StorageObjectCountTracker = tracker
	etcdOptionsCopy.WatchCacheSizes = nil // this control is not provided for custom resources

	return etcdOptionsCopy.CreateRESTOptionsGetter(newCRDStorageFactory(etcdOptionsCopy.StorageConfig, etcdOptionsCopy.EtcdServersOverrides), resourceTransformers)
}

var _ serverstorage.StorageFactory = &crdStorageFactory{}

type crdStorageFactory struct {
	storageConfig storagebackend.Config
	overrides     map[schema.GroupResource]crdStorageOverride
}

type crdStorageOverride struct {
	servers []string
}

func (o crdStorageOverride) apply(config *storagebackend.Config) {
	if len(o.servers) > 0 {
		config.Transport.ServerList = o.servers
	}
}

func newCRDStorageFactory(storageConfig storagebackend.Config, etcdServersOverrides []string) *crdStorageFactory {
	factory := &crdStorageFactory{
		storageConfig: storageConfig,
		overrides:     map[schema.GroupResource]crdStorageOverride{},
	}

	overrides, err := genericoptions.ParseEtcdServersOverrides(etcdServersOverrides)
	if err != nil {
		klog.Errorf("failed to parse etcd-servers-overrides for custom resources, ignoring overrides: %v", err)
		return factory
	}
	for _, override := range overrides {
		factory.overrides[override.GroupResource] = crdStorageOverride{servers: override.Servers}
	}
	return factory
}

func (f *crdStorageFactory) NewConfig(resource schema.GroupResource, example runtime.Object) (*storagebackend.ConfigForResource, error) {
	storageConfig := f.storageConfig
	if override, ok := f.overrides[resource]; ok {
		override.apply(&storageConfig)
	}
	return storageConfig.ForResource(resource), nil
}

func (f *crdStorageFactory) ResourcePrefix(resource schema.GroupResource) string {
	return resource.Group + "/" + resource.Resource
}

func (f *crdStorageFactory) Configs() []storagebackend.Config {
	configs := []storagebackend.Config{f.storageConfig}
	seen := map[string]struct{}{serverListKey(f.storageConfig.Transport.ServerList): {}}

	resources := make([]schema.GroupResource, 0, len(f.overrides))
	for resource := range f.overrides {
		resources = append(resources, resource)
	}
	sort.Slice(resources, func(i, j int) bool {
		return resources[i].String() < resources[j].String()
	})

	for _, resource := range resources {
		config := f.storageConfig
		f.overrides[resource].apply(&config)
		key := serverListKey(config.Transport.ServerList)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}

		configs = append(configs, config)
	}
	return configs
}

func serverListKey(servers []string) string {
	sorted := make([]string, len(servers))
	copy(sorted, servers)
	sort.Strings(sorted)
	return strings.Join(sorted, ";")
}

func (f *crdStorageFactory) Backends() []serverstorage.Backend {
	backends := []serverstorage.Backend{}
	for _, config := range f.Configs() {
		backends = append(backends, serverstorage.Backends(config)...)
	}
	return backends
}

type serviceResolver struct {
	services corev1.ServiceLister
}

func (r *serviceResolver) ResolveEndpoint(namespace, name string, port int32) (*url.URL, error) {
	return proxy.ResolveCluster(r.services, namespace, name, port)
}

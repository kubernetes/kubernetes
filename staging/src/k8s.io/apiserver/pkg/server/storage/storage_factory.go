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

package storage

import (
	"crypto/tls"
	"crypto/x509"
	"os"
	"strings"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/klog/v2"
)

// Backend describes the storage servers, the information here should be enough
// for health validations.
type Backend struct {
	// the url of storage backend like: https://etcd.domain:2379
	Server string
	// the required tls config
	TLSConfig *tls.Config
}

// StorageFactory is the interface to locate the storage for a given GroupResource
type StorageFactory interface {
	// New finds the storage destination for the given group and resource. It will
	// return an error if the group has no storage destination configured.
	NewConfig(groupResource schema.GroupResource, example runtime.Object) (*storagebackend.ConfigForResource, error)

	// ResourcePrefix returns the overridden resource prefix for the GroupResource
	// This allows for cohabitation of resources with different native types and provides
	// centralized control over the shape of etcd directories
	ResourcePrefix(groupResource schema.GroupResource) string

	// Configs gets configurations for all of registered storage destinations.
	Configs() []storagebackend.Config

	// Backends gets all backends for all registered storage destinations.
	// Used for getting all instances for health validations.
	// Deprecated: Use Configs instead
	Backends() []Backend
}

// DefaultStorageFactory takes a GroupResource and returns back its storage interface.  This result includes:
// 1. Merged etcd config, including: auth, server locations, prefixes
// 2. Resource encodings for storage: group,version,kind to store as
// 3. Cohabitating default: some resources like hpa are exposed through multiple APIs.  They must agree on 1 and 2
type DefaultStorageFactory struct {
	// StorageConfig describes how to create a storage backend in general.
	// Its authentication information will be used for every storage.Interface returned.
	StorageConfig storagebackend.Config

	Overrides map[schema.GroupResource]groupResourceOverrides

	DefaultResourcePrefixes map[schema.GroupResource]string

	// DefaultMediaType is the media type used to store resources. If it is not set, "application/json" is used.
	DefaultMediaType string

	// DefaultSerializer is used to create encoders and decoders for the storage.Interface.
	DefaultSerializer runtime.StorageSerializer

	// ResourceEncodingConfig describes how to encode a particular GroupVersionResource
	ResourceEncodingConfig ResourceEncodingConfig

	// APIResourceConfigSource indicates whether the *storage* is enabled, NOT the API
	// This is discrete from resource enablement because those are separate concerns.  How this source is configured
	// is left to the caller.
	APIResourceConfigSource APIResourceConfigSource

	// newStorageCodecFn exists to be overwritten for unit testing.
	newStorageCodecFn func(opts StorageCodecConfig) (codec runtime.Codec, encodeVersioner runtime.GroupVersioner, err error)
}

type groupResourceOverrides struct {
	// etcdLocation contains the list of "special" locations that are used for particular GroupResources
	// These are merged on top of the StorageConfig when requesting the storage.Interface for a given GroupResource
	etcdLocation []string
	// etcdPrefix is the base location for a GroupResource.
	etcdPrefix string
	// etcdResourcePrefix is the location to use to store a particular type under the `etcdPrefix` location
	// If empty, the default mapping is used.  If the default mapping doesn't contain an entry, it will use
	// the ToLowered name of the resource, not including the group.
	etcdResourcePrefix string
	// mediaType is the desired serializer to choose. If empty, the default is chosen.
	mediaType string
	// serializer contains the list of "special" serializers for a GroupResource.  Resource=* means for the entire group
	serializer runtime.StorageSerializer
	// cohabitatingResources keeps track of which resources must be stored together.  This happens when we have multiple ways
	// of exposing one set of concepts.  autoscaling.HPA and extensions.HPA as a for instance
	// The order of the slice matters!  It is the priority order of lookup for finding a storage location
	cohabitatingResources []schema.GroupResource
	// encoderDecoratorFn is optional and may wrap the provided encoder prior to being serialized.
	encoderDecoratorFn func(runtime.Encoder) runtime.Encoder
	// decoderDecoratorFn is optional and may wrap the provided decoders (can add new decoders). The order of
	// returned decoders will be priority for attempt to decode.
	decoderDecoratorFn func([]runtime.Decoder) []runtime.Decoder
}

// Apply overrides the provided config and options if the override has a value in that position
func (o groupResourceOverrides) Apply(config *storagebackend.Config, options *StorageCodecConfig) {
	if len(o.etcdLocation) > 0 {
		config.Transport.ServerList = o.etcdLocation
	}
	if len(o.etcdPrefix) > 0 {
		config.Prefix = o.etcdPrefix
	}

	if len(o.mediaType) > 0 {
		options.StorageMediaType = o.mediaType
	}
	if o.serializer != nil {
		options.StorageSerializer = o.serializer
	}
	if o.encoderDecoratorFn != nil {
		options.EncoderDecoratorFn = o.encoderDecoratorFn
	}
	if o.decoderDecoratorFn != nil {
		options.DecoderDecoratorFn = o.decoderDecoratorFn
	}
}

var _ StorageFactory = &DefaultStorageFactory{}

const AllResources = "*"

func NewDefaultStorageFactory(
	config storagebackend.Config,
	defaultMediaType string,
	defaultSerializer runtime.StorageSerializer,
	resourceEncodingConfig ResourceEncodingConfig,
	resourceConfig APIResourceConfigSource,
	specialDefaultResourcePrefixes map[schema.GroupResource]string,
) *DefaultStorageFactory {
	if len(defaultMediaType) == 0 {
		defaultMediaType = runtime.ContentTypeJSON
	}
	return &DefaultStorageFactory{
		StorageConfig:           config,
		Overrides:               map[schema.GroupResource]groupResourceOverrides{},
		DefaultMediaType:        defaultMediaType,
		DefaultSerializer:       defaultSerializer,
		ResourceEncodingConfig:  resourceEncodingConfig,
		APIResourceConfigSource: resourceConfig,
		DefaultResourcePrefixes: specialDefaultResourcePrefixes,

		newStorageCodecFn: NewStorageCodec,
	}
}

func (s *DefaultStorageFactory) SetEtcdLocation(groupResource schema.GroupResource, location []string) {
	overrides := s.Overrides[groupResource]
	overrides.etcdLocation = location
	s.Overrides[groupResource] = overrides
}

func (s *DefaultStorageFactory) SetEtcdPrefix(groupResource schema.GroupResource, prefix string) {
	overrides := s.Overrides[groupResource]
	overrides.etcdPrefix = prefix
	s.Overrides[groupResource] = overrides
}

// SetResourceEtcdPrefix sets the prefix for a resource, but not the base-dir.  You'll end up in `etcdPrefix/resourceEtcdPrefix`.
func (s *DefaultStorageFactory) SetResourceEtcdPrefix(groupResource schema.GroupResource, prefix string) {
	overrides := s.Overrides[groupResource]
	overrides.etcdResourcePrefix = prefix
	s.Overrides[groupResource] = overrides
}

func (s *DefaultStorageFactory) SetSerializer(groupResource schema.GroupResource, mediaType string, serializer runtime.StorageSerializer) {
	overrides := s.Overrides[groupResource]
	overrides.mediaType = mediaType
	overrides.serializer = serializer
	s.Overrides[groupResource] = overrides
}

// AddCohabitatingResources links resources together the order of the slice matters!  its the priority order of lookup for finding a storage location
func (s *DefaultStorageFactory) AddCohabitatingResources(groupResources ...schema.GroupResource) {
	for _, groupResource := range groupResources {
		overrides := s.Overrides[groupResource]
		overrides.cohabitatingResources = groupResources
		s.Overrides[groupResource] = overrides
	}
}

func (s *DefaultStorageFactory) AddSerializationChains(encoderDecoratorFn func(runtime.Encoder) runtime.Encoder, decoderDecoratorFn func([]runtime.Decoder) []runtime.Decoder, groupResources ...schema.GroupResource) {
	for _, groupResource := range groupResources {
		overrides := s.Overrides[groupResource]
		overrides.encoderDecoratorFn = encoderDecoratorFn
		overrides.decoderDecoratorFn = decoderDecoratorFn
		s.Overrides[groupResource] = overrides
	}
}

func getAllResourcesAlias(resource schema.GroupResource) schema.GroupResource {
	return schema.GroupResource{Group: resource.Group, Resource: AllResources}
}

func (s *DefaultStorageFactory) getStorageGroupResource(groupResource schema.GroupResource) schema.GroupResource {
	for _, potentialStorageResource := range s.Overrides[groupResource].cohabitatingResources {
		// TODO deads2k or liggitt determine if have ever stored any of our cohabitating resources in a different location on new clusters
		if s.APIResourceConfigSource.AnyResourceForGroupEnabled(potentialStorageResource.Group) {
			return potentialStorageResource
		}
	}

	return groupResource
}

// New finds the storage destination for the given group and resource. It will
// return an error if the group has no storage destination configured.
func (s *DefaultStorageFactory) NewConfig(groupResource schema.GroupResource, example runtime.Object) (*storagebackend.ConfigForResource, error) {
	chosenStorageResource := s.getStorageGroupResource(groupResource)

	// operate on copy
	storageConfig := s.StorageConfig
	codecConfig := StorageCodecConfig{
		StorageMediaType:  s.DefaultMediaType,
		StorageSerializer: s.DefaultSerializer,
	}

	if override, ok := s.Overrides[getAllResourcesAlias(chosenStorageResource)]; ok {
		override.Apply(&storageConfig, &codecConfig)
	}
	if override, ok := s.Overrides[chosenStorageResource]; ok {
		override.Apply(&storageConfig, &codecConfig)
	}

	var err error
	if backwardCompatibleInterface, ok := s.ResourceEncodingConfig.(CompatibilityResourceEncodingConfig); ok {
		codecConfig.StorageVersion, err = backwardCompatibleInterface.BackwardCompatibileStorageEncodingFor(chosenStorageResource, example)
		if err != nil {
			return nil, err
		}
	} else {
		codecConfig.StorageVersion, err = s.ResourceEncodingConfig.StorageEncodingFor(chosenStorageResource)
		if err != nil {
			return nil, err
		}
	}

	codecConfig.MemoryVersion, err = s.ResourceEncodingConfig.InMemoryEncodingFor(groupResource)
	if err != nil {
		return nil, err
	}

	codecConfig.Config = storageConfig

	storageConfig.Codec, storageConfig.EncodeVersioner, err = s.newStorageCodecFn(codecConfig)
	if err != nil {
		return nil, err
	}
	klog.V(3).Infof("storing %v in %v, reading as %v from %#v", groupResource, codecConfig.StorageVersion, codecConfig.MemoryVersion, codecConfig.Config)

	return storageConfig.ForResource(groupResource), nil
}

// Configs implements StorageFactory.
func (s *DefaultStorageFactory) Configs() []storagebackend.Config {
	return configs(s.StorageConfig, s.Overrides)
}

// Configs gets configurations for all of registered storage destinations.
func Configs(storageConfig storagebackend.Config) []storagebackend.Config {
	return configs(storageConfig, nil)
}

// Returns all storage configurations including those for group resource overrides
func configs(storageConfig storagebackend.Config, grOverrides map[schema.GroupResource]groupResourceOverrides) []storagebackend.Config {
	configs := []storagebackend.Config{storageConfig}

	for _, override := range grOverrides {
		if len(override.etcdLocation) == 0 {
			continue
		}
		// copy
		newConfig := storageConfig
		override.Apply(&newConfig, &StorageCodecConfig{})
		newConfig.Transport.ServerList = override.etcdLocation
		configs = append(configs, newConfig)
	}
	return configs
}

// Backends implements StorageFactory.
func (s *DefaultStorageFactory) Backends() []Backend {
	return backends(s.StorageConfig, s.Overrides)
}

// Backends returns all backends for all registered storage destinations.
// Used for getting all instances for health validations.
// Deprecated: Validate health by passing storagebackend.Config directly to storagefactory.CreateProber.
func Backends(storageConfig storagebackend.Config) []Backend {
	return backends(storageConfig, nil)
}

func backends(storageConfig storagebackend.Config, grOverrides map[schema.GroupResource]groupResourceOverrides) []Backend {
	servers := sets.NewString(storageConfig.Transport.ServerList...)

	for _, overrides := range grOverrides {
		servers.Insert(overrides.etcdLocation...)
	}

	tlsConfig := &tls.Config{
		InsecureSkipVerify: true,
	}
	if len(storageConfig.Transport.CertFile) > 0 && len(storageConfig.Transport.KeyFile) > 0 {
		cert, err := tls.LoadX509KeyPair(storageConfig.Transport.CertFile, storageConfig.Transport.KeyFile)
		if err != nil {
			klog.Errorf("failed to load key pair while getting backends: %s", err)
		} else {
			tlsConfig.Certificates = []tls.Certificate{cert}
		}
	}
	if len(storageConfig.Transport.TrustedCAFile) > 0 {
		if caCert, err := os.ReadFile(storageConfig.Transport.TrustedCAFile); err != nil {
			klog.Errorf("failed to read ca file while getting backends: %s", err)
		} else {
			caPool := x509.NewCertPool()
			caPool.AppendCertsFromPEM(caCert)
			tlsConfig.RootCAs = caPool
			tlsConfig.InsecureSkipVerify = false
		}
	}

	backends := []Backend{}
	for server := range servers {
		backends = append(backends, Backend{
			Server: server,
			// We can't share TLSConfig across different backends to avoid races.
			// For more details see: https://pr.k8s.io/59338
			TLSConfig: tlsConfig.Clone(),
		})
	}
	return backends
}

func (s *DefaultStorageFactory) ResourcePrefix(groupResource schema.GroupResource) string {
	chosenStorageResource := s.getStorageGroupResource(groupResource)
	groupOverride := s.Overrides[getAllResourcesAlias(chosenStorageResource)]
	exactResourceOverride := s.Overrides[chosenStorageResource]

	etcdResourcePrefix := s.DefaultResourcePrefixes[chosenStorageResource]
	if len(groupOverride.etcdResourcePrefix) > 0 {
		etcdResourcePrefix = groupOverride.etcdResourcePrefix
	}
	if len(exactResourceOverride.etcdResourcePrefix) > 0 {
		etcdResourcePrefix = exactResourceOverride.etcdResourcePrefix
	}
	if len(etcdResourcePrefix) == 0 {
		etcdResourcePrefix = strings.ToLower(chosenStorageResource.Resource)
	}

	return etcdResourcePrefix
}

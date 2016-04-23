/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package genericapiserver

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/versioning"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/golang/glog"
)

// StorageFactory is the interface to locate the storage for a given GroupResource
type StorageFactory interface {
	// New finds the storage destination for the given group and resource. It will
	// return an error if the group has no storage destination configured.
	New(groupResource unversioned.GroupResource) (storage.Interface, error)
	// Backends gets all backends for all registered storage destinations.
	// Used for getting all instances for health validations.
	Backends() []string
}

// DefaultStorageFactory takes a GroupResource and returns back its storage interface.  This result includes:
// 1. Merged etcd config, including: auth, server locations, prefixes
// 2. Resource encodings for storage: group,version,kind to store as
// 3. Cohabitating default: some resources like hpa are exposed through multiple APIs.  They must agree on 1 and 2
type DefaultStorageFactory struct {
	// StorageConfig describes how to create a storage backend in general.
	// Its authentication information will be used for every storage.Interface returned.
	StorageConfig storagebackend.Config

	Overrides map[unversioned.GroupResource]groupResourceOverrides

	// DefaultSerializer is used to create encoders and decoders for the storage.Interface.
	DefaultSerializer runtime.NegotiatedSerializer

	// ResourceEncodingConfig describes how to encode a particular GroupVersionResource
	ResourceEncodingConfig ResourceEncodingConfig

	// APIResourceConfigSource indicates whether the *storage* is enabled, NOT the API
	// This is discrete from resource enablement because those are separate concerns.  How it is surfaced to the user via flags
	// or config is up to whoever is building this.
	APIResourceConfigSource APIResourceConfigSource

	// newEtcdFn exists to be overwritten for unit testing.  You should never set this in a normal world.
	newEtcdFn func(ns runtime.NegotiatedSerializer, storageVersion, memoryVersion unversioned.GroupVersion, config storagebackend.Config) (etcdStorage storage.Interface, err error)
}

type groupResourceOverrides struct {
	// etcdLocation contains the list of "special" locations that are used for particular GroupResources
	// These are merged on top of the StorageConfig when requesting the storage.Interface for a given GroupResource
	etcdLocation []string
	// etcdPrefix contains the list of "special" prefixes for a GroupResource.  Resource=* means for the entire group
	etcdPrefix string
	// serializer contains the list of "special" serializers for a GroupResource.  Resource=* means for the entire group
	serializer runtime.NegotiatedSerializer
	// cohabitatingResources keeps track of which resources must be stored together.  This happens when we have multiple ways
	// of exposing one set of concepts.  autoscaling.HPA and extensions.HPA as a for instance
	// The order of the slice matters!  It is the priority order of lookup for finding a storage location
	cohabitatingResources []unversioned.GroupResource
}

var _ StorageFactory = &DefaultStorageFactory{}

const AllResources = "*"

func NewDefaultStorageFactory(config storagebackend.Config, defaultSerializer runtime.NegotiatedSerializer, resourceEncodingConfig ResourceEncodingConfig, resourceConfig APIResourceConfigSource) *DefaultStorageFactory {
	return &DefaultStorageFactory{
		StorageConfig:           config,
		Overrides:               map[unversioned.GroupResource]groupResourceOverrides{},
		DefaultSerializer:       defaultSerializer,
		ResourceEncodingConfig:  resourceEncodingConfig,
		APIResourceConfigSource: resourceConfig,

		newEtcdFn: newEtcd,
	}
}

func (s *DefaultStorageFactory) SetEtcdLocation(groupResource unversioned.GroupResource, location []string) {
	overrides := s.Overrides[groupResource]
	overrides.etcdLocation = location
	s.Overrides[groupResource] = overrides
}

func (s *DefaultStorageFactory) SetEtcdPrefix(groupResource unversioned.GroupResource, prefix string) {
	overrides := s.Overrides[groupResource]
	overrides.etcdPrefix = prefix
	s.Overrides[groupResource] = overrides
}

func (s *DefaultStorageFactory) SetSerializer(groupResource unversioned.GroupResource, serializer runtime.NegotiatedSerializer) {
	overrides := s.Overrides[groupResource]
	overrides.serializer = serializer
	s.Overrides[groupResource] = overrides
}

// AddCohabitatingResources links resources together the order of the slice matters!  its the priority order of lookup for finding a storage location
func (s *DefaultStorageFactory) AddCohabitatingResources(groupResources ...unversioned.GroupResource) {
	for _, groupResource := range groupResources {
		overrides := s.Overrides[groupResource]
		overrides.cohabitatingResources = groupResources
		s.Overrides[groupResource] = overrides
	}
}

func getAllResourcesAlias(resource unversioned.GroupResource) unversioned.GroupResource {
	return unversioned.GroupResource{Group: resource.Group, Resource: AllResources}
}

func (s *DefaultStorageFactory) getStorageGroupResource(groupResource unversioned.GroupResource) unversioned.GroupResource {
	for _, potentialStorageResource := range s.Overrides[groupResource].cohabitatingResources {
		if s.APIResourceConfigSource.AnyVersionOfResourceEnabled(potentialStorageResource) {
			return potentialStorageResource
		}
	}

	return groupResource
}

// New finds the storage destination for the given group and resource. It will
// return an error if the group has no storage destination configured.
func (s *DefaultStorageFactory) New(groupResource unversioned.GroupResource) (storage.Interface, error) {
	chosenStorageResource := s.getStorageGroupResource(groupResource)

	groupOverride := s.Overrides[getAllResourcesAlias(chosenStorageResource)]
	exactResourceOverride := s.Overrides[chosenStorageResource]

	overriddenEtcdLocations := []string{}
	if len(groupOverride.etcdLocation) > 0 {
		overriddenEtcdLocations = groupOverride.etcdLocation
	}
	if len(exactResourceOverride.etcdLocation) > 0 {
		overriddenEtcdLocations = exactResourceOverride.etcdLocation
	}

	etcdPrefix := s.StorageConfig.Prefix
	if len(groupOverride.etcdPrefix) > 0 {
		etcdPrefix = groupOverride.etcdPrefix
	}
	if len(exactResourceOverride.etcdPrefix) > 0 {
		etcdPrefix = exactResourceOverride.etcdPrefix
	}

	etcdSerializer := s.DefaultSerializer
	if groupOverride.serializer != nil {
		etcdSerializer = groupOverride.serializer
	}
	if exactResourceOverride.serializer != nil {
		etcdSerializer = exactResourceOverride.serializer
	}
	// operate on copy
	config := s.StorageConfig
	config.Prefix = etcdPrefix
	if len(overriddenEtcdLocations) > 0 {
		config.ServerList = overriddenEtcdLocations
	}

	storageEncodingVersion, err := s.ResourceEncodingConfig.StorageEncodingFor(chosenStorageResource)
	if err != nil {
		return nil, err
	}
	internalVersion, err := s.ResourceEncodingConfig.InMemoryEncodingFor(groupResource)
	if err != nil {
		return nil, err
	}

	glog.V(3).Infof("storing %v in %v, reading as %v from %v", groupResource, storageEncodingVersion, internalVersion, config)
	return s.newEtcdFn(etcdSerializer, storageEncodingVersion, internalVersion, config)
}

func newEtcd(ns runtime.NegotiatedSerializer, storageVersion, memoryVersion unversioned.GroupVersion, config storagebackend.Config) (etcdStorage storage.Interface, err error) {
	s, ok := ns.SerializerForMediaType("application/json", nil)
	if !ok {
		return nil, fmt.Errorf("unable to find serializer for JSON")
	}
	encoder := ns.EncoderForVersion(s, storageVersion)
	decoder := ns.DecoderToVersion(s, memoryVersion)
	if memoryVersion.Group != storageVersion.Group {
		// Allow this codec to translate between groups.
		if err = versioning.EnableCrossGroupEncoding(encoder, memoryVersion.Group, storageVersion.Group); err != nil {
			return nil, fmt.Errorf("error setting up encoder from %v to %v: %v", memoryVersion, storageVersion, err)
		}
		if err = versioning.EnableCrossGroupDecoding(decoder, storageVersion.Group, memoryVersion.Group); err != nil {
			return nil, fmt.Errorf("error setting up decoder from %v to %v: %v", storageVersion, memoryVersion, err)
		}
	}
	config.Codec = runtime.NewCodec(encoder, decoder)
	return storagebackend.Create(config)
}

// Get all backends for all registered storage destinations.
// Used for getting all instances for health validations.
func (s *DefaultStorageFactory) Backends() []string {
	backends := sets.NewString(s.StorageConfig.ServerList...)

	for _, overrides := range s.Overrides {
		backends.Insert(overrides.etcdLocation...)
	}
	return backends.List()
}

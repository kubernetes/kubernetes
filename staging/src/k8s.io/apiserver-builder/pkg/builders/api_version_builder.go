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

package builders

import (
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"reflect"
)

type VersionedApiBuilder struct {
	Kinds         []*versionedResourceBuilder
	GroupVersion  schema.GroupVersion
	SchemaBuilder runtime.SchemeBuilder
}

func NewApiVersion(group, version string) *VersionedApiBuilder {
	b := &VersionedApiBuilder{
		GroupVersion: schema.GroupVersion{group, version},
	}
	b.SchemaBuilder = runtime.NewSchemeBuilder(b.registerTypes, b.registerDefaults, b.registerConversions)
	return b
}

// WithResources adds new resource types and subresources to the API versions
// resourceBuilders is a list of *versionedResourceBuilder
func (s *VersionedApiBuilder) WithResources(resourceBuilders ...*versionedResourceBuilder) *VersionedApiBuilder {
	s.Kinds = append(s.Kinds, resourceBuilders...)
	for _, b := range resourceBuilders {
		s.SchemaBuilder.Register(b.SchemeFns.Register)
	}
	return s
}

// registerVersionToScheme registers the version to scheme mapping
func (s *VersionedApiBuilder) registerVersionToScheme(to announced.VersionToSchemeFunc) {
	to[s.GroupVersion.Version] = s.SchemaBuilder.AddToScheme
}

// registerTypes registers all of the types in this API version
func (s *VersionedApiBuilder) registerTypes(scheme *runtime.Scheme) error {
	for _, k := range s.Kinds {
		// RegisterTypes type
		if t := k.New(); t != nil {
			scheme.AddKnownTypes(s.GroupVersion, t) // Register the versioned type
		}

		// RegisterTypes list type if it exists
		if l := k.NewList(); l != nil {
			scheme.AddKnownTypes(s.GroupVersion, l) // Register the versioned type
		}
	}
	metav1.AddToGroupVersion(scheme, s.GroupVersion)
	return nil
}

func (s *VersionedApiBuilder) registerDefaults(scheme *runtime.Scheme) error {
	for _, k := range s.Kinds {
		err := scheme.AddDefaultingFuncs(k.SchemeFns.GetDefaultingFunctions()...)
		if err != nil {
			glog.Errorf("Failed to add Defaulting functions %v", err)
			return err
		}
	}
	return nil
}

func (s *VersionedApiBuilder) registerConversions(scheme *runtime.Scheme) error {
	for _, k := range s.Kinds {
		err := scheme.AddDefaultingFuncs(k.SchemeFns.GetConversionFunctions()...)
		if err != nil {
			glog.Errorf("Failed to add conversion functions %v", err)
			return err
		}
	}
	return nil
}

// registerEndpoints registers the REST endpoints for all resources in this API group version
// group is the group to register the resources under
// optionsGetter is the RESTOptionsGetter provided by a server.Config
// registry is the server.APIGroupInfo VersionedResourcesStorageMap used to register REST endpoints
func (s *VersionedApiBuilder) registerEndpoints(
	optionsGetter generic.RESTOptionsGetter,
	registry map[string]map[string]rest.Storage) {

	// Register the endpoints for each kind
	for _, k := range s.Kinds {
		if _, found := registry[s.GroupVersion.Version]; !found {
			// Initialize the version if missing
			registry[s.GroupVersion.Version] = map[string]rest.Storage{}
		}
		// Register each of the endpoints in this version
		k.registerEndpoints(s.GroupVersion.Group, optionsGetter, registry[s.GroupVersion.Version])
	}
}

type UnVersionedApiBuilder struct {
	Kinds         []UnversionedResourceBuilder
	GroupVersion  schema.GroupVersion
	SchemaBuilder runtime.SchemeBuilder
}

func NewApiGroup(group string) *UnVersionedApiBuilder {
	b := &UnVersionedApiBuilder{
		GroupVersion: schema.GroupVersion{group, runtime.APIVersionInternal},
	}
	b.SchemaBuilder = runtime.NewSchemeBuilder(b.registerTypes)
	return b
}

func (s *UnVersionedApiBuilder) WithKinds(kinds ...UnversionedResourceBuilder) *UnVersionedApiBuilder {
	s.Kinds = append(s.Kinds, kinds...)
	return s
}

func (s *UnVersionedApiBuilder) registerTypes(scheme *runtime.Scheme) error {
	// Make sure we only each type once
	typesToRegister := map[reflect.Type]runtime.Object{}
	for _, k := range s.Kinds {
		// RegisterTypes type
		if t := k.New(); t != nil {
			// Register the unversioned type
			typesToRegister[reflect.TypeOf(t)] = t
		}

		// RegisterTypes list type if it exists
		if i, ok := k.(WithList); ok && i.NewList() != nil {
			// Register the unversioned type
			l := i.NewList()
			typesToRegister[reflect.TypeOf(l)] = l
		}
	}
	for _, t := range typesToRegister {
		scheme.AddKnownTypes(s.GroupVersion, t)
	}

	//fmt.Printf("Registering for group %v\n", s.GroupVersion)
	//metav1.AddToGroupVersion(scheme, s.GroupVersion)
	return nil
}

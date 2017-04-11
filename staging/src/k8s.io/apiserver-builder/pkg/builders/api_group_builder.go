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
	"k8s.io/apimachinery/pkg/apimachinery/announced"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/client-go/pkg/api"
)

// Global registry of API groups
var APIGroupBuilders = []*APIGroupBuilder{}

type APIGroupBuilder struct {
	UnVersioned  *UnVersionedApiBuilder
	Versions     []*VersionedApiBuilder
	Name         string
	ImportPrefix string
}

func NewApiGroupBuilder(name, prefix string) *APIGroupBuilder {
	g := &APIGroupBuilder{
		Name:         name,
		ImportPrefix: prefix,
	}
	return g
}

func (g *APIGroupBuilder) WithUnVersionedApi(unversioned *UnVersionedApiBuilder) *APIGroupBuilder {
	g.UnVersioned = unversioned
	return g
}

func (g *APIGroupBuilder) WithVersionedApis(versions ...*VersionedApiBuilder) *APIGroupBuilder {
	g.Versions = append(g.Versions, versions...)
	return g
}

// GetVersionPreferenceOrder returns the preferred ordering of versions for this api group
func (g *APIGroupBuilder) GetVersionPreferenceOrder() []string {
	order := []string{}
	for _, v := range g.Versions {
		order = append(order, v.GroupVersion.Version)
	}
	return order
}

// VersionToSchemeFunc returns a map of version to AddToScheme function for all versioned Schemes
func (g *APIGroupBuilder) VersionToSchemeFunc() announced.VersionToSchemeFunc {
	f := announced.VersionToSchemeFunc{}
	for _, v := range g.Versions {
		v.registerVersionToScheme(f)
	}
	return f
}

func (g *APIGroupBuilder) GetLegacyCodec() []schema.GroupVersion {
	versions := []schema.GroupVersion{}
	for _, v := range g.Versions {
		versions = append(versions, v.GroupVersion)
	}
	return versions
}

func (g *APIGroupBuilder) registerEndpoints(
	optionsGetter generic.RESTOptionsGetter,
	registry map[string]map[string]rest.Storage) {

	// Register the endpoints for each version
	for _, v := range g.Versions {
		v.registerEndpoints(optionsGetter, registry)
	}
}

// Build returns a new NewDefaultAPIGroupInfo to install into a GenericApiServer
func (g *APIGroupBuilder) Build(optionsGetter generic.RESTOptionsGetter) *genericapiserver.APIGroupInfo {

	// Build a new group
	i := genericapiserver.NewDefaultAPIGroupInfo(
		g.Name,
		api.Registry,
		api.Scheme,
		metav1.ParameterCodec,
		api.Codecs)

	// First group version is preferred
	i.GroupMeta.GroupVersion = g.Versions[0].GroupVersion

	// Register the endpoints with the group
	g.registerEndpoints(optionsGetter, i.VersionedResourcesStorageMap)

	return &i

}

// Announce installs the API group for an api server
func (g *APIGroupBuilder) Announce() {
	if err := announced.NewGroupMetaFactory(
		&announced.GroupMetaFactoryArgs{
			GroupName:                  g.Name,
			RootScopedKinds:            sets.NewString("APIService"),
			VersionPreferenceOrder:     g.GetVersionPreferenceOrder(),
			ImportPrefix:               g.ImportPrefix,
			AddInternalObjectsToScheme: g.UnVersioned.SchemaBuilder.AddToScheme,
		},
		g.VersionToSchemeFunc(),
	).Announce(api.GroupFactoryRegistry).RegisterAndEnable(api.Registry, api.Scheme); err != nil {
		panic(err)
	}

}

/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

// TODO: move everything in this file to pkg/api/rest
package meta

import (
	"fmt"
	"strings"

	"k8s.io/kubernetes/pkg/api/unversioned"
)

// Implements RESTScope interface
type restScope struct {
	name             RESTScopeName
	paramName        string
	argumentName     string
	paramDescription string
}

func (r *restScope) Name() RESTScopeName {
	return r.name
}
func (r *restScope) ParamName() string {
	return r.paramName
}
func (r *restScope) ArgumentName() string {
	return r.argumentName
}
func (r *restScope) ParamDescription() string {
	return r.paramDescription
}

var RESTScopeNamespace = &restScope{
	name:             RESTScopeNameNamespace,
	paramName:        "namespaces",
	argumentName:     "namespace",
	paramDescription: "object name and auth scope, such as for teams and projects",
}

var RESTScopeRoot = &restScope{
	name: RESTScopeNameRoot,
}

// DefaultRESTMapper exposes mappings between the types defined in a
// runtime.Scheme. It assumes that all types defined the provided scheme
// can be mapped with the provided MetadataAccessor and Codec interfaces.
//
// The resource name of a Kind is defined as the lowercase,
// English-plural version of the Kind string.
// When converting from resource to Kind, the singular version of the
// resource name is also accepted for convenience.
//
// TODO: Only accept plural for some operations for increased control?
// (`get pod bar` vs `get pods bar`)
// TODO these maps should be keyed based on GroupVersionKinds
type DefaultRESTMapper struct {
	defaultGroupVersions []unversioned.GroupVersion

	resourceToKind       map[string]unversioned.GroupVersionKind
	kindToPluralResource map[unversioned.GroupVersionKind]string
	kindToScope          map[unversioned.GroupVersionKind]RESTScope
	singularToPlural     map[string]string
	pluralToSingular     map[string]string

	interfacesFunc VersionInterfacesFunc
}

var _ RESTMapper = &DefaultRESTMapper{}

// VersionInterfacesFunc returns the appropriate codec, typer, and metadata accessor for a
// given api version, or an error if no such api version exists.
type VersionInterfacesFunc func(apiVersion string) (*VersionInterfaces, error)

// NewDefaultRESTMapper initializes a mapping between Kind and APIVersion
// to a resource name and back based on the objects in a runtime.Scheme
// and the Kubernetes API conventions. Takes a group name, a priority list of the versions
// to search when an object has no default version (set empty to return an error),
// and a function that retrieves the correct codec and metadata for a given version.
func NewDefaultRESTMapper(defaultGroupVersions []unversioned.GroupVersion, f VersionInterfacesFunc) *DefaultRESTMapper {
	resourceToKind := make(map[string]unversioned.GroupVersionKind)
	kindToPluralResource := make(map[unversioned.GroupVersionKind]string)
	kindToScope := make(map[unversioned.GroupVersionKind]RESTScope)
	singularToPlural := make(map[string]string)
	pluralToSingular := make(map[string]string)
	// TODO: verify name mappings work correctly when versions differ

	return &DefaultRESTMapper{
		resourceToKind:       resourceToKind,
		kindToPluralResource: kindToPluralResource,
		kindToScope:          kindToScope,
		defaultGroupVersions: defaultGroupVersions,
		singularToPlural:     singularToPlural,
		pluralToSingular:     pluralToSingular,
		interfacesFunc:       f,
	}
}

func (m *DefaultRESTMapper) Add(gvk unversioned.GroupVersionKind, scope RESTScope, mixedCase bool) {
	plural, singular := KindToResource(gvk.Kind, mixedCase)
	m.singularToPlural[singular] = plural
	m.pluralToSingular[plural] = singular
	_, ok1 := m.resourceToKind[plural]
	_, ok2 := m.resourceToKind[strings.ToLower(plural)]
	if !ok1 && !ok2 {
		m.resourceToKind[plural] = gvk
		m.resourceToKind[singular] = gvk
		if strings.ToLower(plural) != plural {
			m.resourceToKind[strings.ToLower(plural)] = gvk
			m.resourceToKind[strings.ToLower(singular)] = gvk
		}
	}
	m.kindToPluralResource[gvk] = plural
	m.kindToScope[gvk] = scope
}

// KindToResource converts Kind to a resource name.
func KindToResource(kind string, mixedCase bool) (plural, singular string) {
	if len(kind) == 0 {
		return
	}
	if mixedCase {
		// Legacy support for mixed case names
		singular = strings.ToLower(kind[:1]) + kind[1:]
	} else {
		singular = strings.ToLower(kind)
	}
	if strings.HasSuffix(singular, "endpoints") {
		plural = singular
	} else {
		switch string(singular[len(singular)-1]) {
		case "s":
			plural = singular + "es"
		case "y":
			plural = strings.TrimSuffix(singular, "y") + "ies"
		default:
			plural = singular + "s"
		}
	}
	return
}

// ResourceSingularizer implements RESTMapper
// It converts a resource name from plural to singular (e.g., from pods to pod)
func (m *DefaultRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	singular, ok := m.pluralToSingular[resource]
	if !ok {
		return resource, fmt.Errorf("no singular of resource %q has been defined", resource)
	}
	return singular, nil
}

// VersionAndKindForResource implements RESTMapper
func (m *DefaultRESTMapper) KindFor(resource string) (unversioned.GroupVersionKind, error) {
	gvk, ok := m.resourceToKind[strings.ToLower(resource)]
	if !ok {
		return gvk, fmt.Errorf("in version and kind for resource, no resource %q has been defined", resource)
	}
	return gvk, nil
}

// RESTMapping returns a struct representing the resource path and conversion interfaces a
// RESTClient should use to operate on the provided group/kind in order of versions. If a version search
// order is not provided, the search order provided to DefaultRESTMapper will be used to resolve which
// version should be used to access the named group/kind.
func (m *DefaultRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*RESTMapping, error) {
	// Pick an appropriate version
	var gvk *unversioned.GroupVersionKind
	hadVersion := false
	for _, version := range versions {
		if len(version) == 0 {
			continue
		}

		currGVK := gk.WithVersion(version)
		hadVersion = true
		if _, ok := m.kindToPluralResource[currGVK]; ok {
			gvk = &currGVK
			break
		}
	}
	// Use the default preferred versions
	if !hadVersion && (gvk == nil) {
		for _, gv := range m.defaultGroupVersions {
			if gv.Group != gk.Group {
				continue
			}

			currGVK := gk.WithVersion(gv.Version)
			if _, ok := m.kindToPluralResource[currGVK]; ok {
				gvk = &currGVK
				break
			}
		}
	}
	if gvk == nil {
		return nil, fmt.Errorf("no kind named %q is registered in versions %q", gk, versions)
	}

	// Ensure we have a REST mapping
	resource, ok := m.kindToPluralResource[*gvk]
	if !ok {
		found := []unversioned.GroupVersion{}
		for _, gv := range m.defaultGroupVersions {
			if _, ok := m.kindToPluralResource[*gvk]; ok {
				found = append(found, gv)
			}
		}
		if len(found) > 0 {
			return nil, fmt.Errorf("object with kind %q exists in versions %v, not %v", gvk.Kind, found, gvk.GroupVersion().String())
		}
		return nil, fmt.Errorf("the provided version %q and kind %q cannot be mapped to a supported object", gvk.GroupVersion().String(), gvk.Kind)
	}

	// Ensure we have a REST scope
	scope, ok := m.kindToScope[*gvk]
	if !ok {
		return nil, fmt.Errorf("the provided version %q and kind %q cannot be mapped to a supported scope", gvk.GroupVersion().String(), gvk.Kind)
	}

	interfaces, err := m.interfacesFunc(gvk.GroupVersion().String())
	if err != nil {
		return nil, fmt.Errorf("the provided version %q has no relevant versions", gvk.GroupVersion().String())
	}

	retVal := &RESTMapping{
		Resource:         resource,
		GroupVersionKind: *gvk,
		Scope:            scope,

		Codec:            interfaces.Codec,
		ObjectConvertor:  interfaces.ObjectConvertor,
		MetadataAccessor: interfaces.MetadataAccessor,
	}

	return retVal, nil
}

// aliasToResource is used for mapping aliases to resources
var aliasToResource = map[string][]string{}

// AddResourceAlias maps aliases to resources
func (m *DefaultRESTMapper) AddResourceAlias(alias string, resources ...string) {
	if len(resources) == 0 {
		return
	}
	aliasToResource[alias] = resources
}

// AliasesForResource returns whether a resource has an alias or not
func (m *DefaultRESTMapper) AliasesForResource(alias string) ([]string, bool) {
	if res, ok := aliasToResource[alias]; ok {
		return res, true
	}
	return nil, false
}

// ResourceIsValid takes a string (kind) and checks if it's a valid resource
func (m *DefaultRESTMapper) ResourceIsValid(resource string) bool {
	_, err := m.KindFor(resource)
	return err == nil
}

// MultiRESTMapper is a wrapper for multiple RESTMappers.
type MultiRESTMapper []RESTMapper

// ResourceSingularizer converts a REST resource name from plural to singular (e.g., from pods to pod)
// This implementation supports multiple REST schemas and return the first match.
func (m MultiRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	for _, t := range m {
		singular, err = t.ResourceSingularizer(resource)
		if err == nil {
			return
		}
	}
	return
}

// VersionAndKindForResource provides the Version and Kind  mappings for the
// REST resources. This implementation supports multiple REST schemas and return
// the first match.
func (m MultiRESTMapper) KindFor(resource string) (gvk unversioned.GroupVersionKind, err error) {
	for _, t := range m {
		gvk, err = t.KindFor(resource)
		if err == nil {
			return
		}
	}
	return
}

// RESTMapping provides the REST mapping for the resource based on the
// kind and version. This implementation supports multiple REST schemas and
// return the first match.
func (m MultiRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (mapping *RESTMapping, err error) {
	for _, t := range m {
		mapping, err = t.RESTMapping(gk, versions...)
		if err == nil {
			return
		}
	}
	return
}

// AliasesForResource finds the first alias response for the provided mappers.
func (m MultiRESTMapper) AliasesForResource(alias string) (aliases []string, ok bool) {
	for _, t := range m {
		if aliases, ok = t.AliasesForResource(alias); ok {
			return
		}
	}
	return nil, false
}

// ResourceIsValid takes a string (either group/kind or kind) and checks if it's a valid resource
func (m MultiRESTMapper) ResourceIsValid(resource string) bool {
	for _, t := range m {
		if t.ResourceIsValid(resource) {
			return true
		}
	}
	return false
}

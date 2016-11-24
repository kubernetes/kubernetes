/*
Copyright 2014 The Kubernetes Authors.

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
	"sort"
	"strings"

	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/runtime/schema"
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
type DefaultRESTMapper struct {
	defaultGroupVersions []schema.GroupVersion

	resourceToKind       map[schema.GroupVersionResource]schema.GroupVersionKind
	kindToPluralResource map[schema.GroupVersionKind]schema.GroupVersionResource
	kindToScope          map[schema.GroupVersionKind]RESTScope
	singularToPlural     map[schema.GroupVersionResource]schema.GroupVersionResource
	pluralToSingular     map[schema.GroupVersionResource]schema.GroupVersionResource

	interfacesFunc VersionInterfacesFunc

	// aliasToResource is used for mapping aliases to resources
	aliasToResource map[string][]string
}

func (m *DefaultRESTMapper) String() string {
	return fmt.Sprintf("DefaultRESTMapper{kindToPluralResource=%v}", m.kindToPluralResource)
}

var _ RESTMapper = &DefaultRESTMapper{}

// VersionInterfacesFunc returns the appropriate typer, and metadata accessor for a
// given api version, or an error if no such api version exists.
type VersionInterfacesFunc func(version schema.GroupVersion) (*VersionInterfaces, error)

// NewDefaultRESTMapper initializes a mapping between Kind and APIVersion
// to a resource name and back based on the objects in a runtime.Scheme
// and the Kubernetes API conventions. Takes a group name, a priority list of the versions
// to search when an object has no default version (set empty to return an error),
// and a function that retrieves the correct metadata for a given version.
func NewDefaultRESTMapper(defaultGroupVersions []schema.GroupVersion, f VersionInterfacesFunc) *DefaultRESTMapper {
	resourceToKind := make(map[schema.GroupVersionResource]schema.GroupVersionKind)
	kindToPluralResource := make(map[schema.GroupVersionKind]schema.GroupVersionResource)
	kindToScope := make(map[schema.GroupVersionKind]RESTScope)
	singularToPlural := make(map[schema.GroupVersionResource]schema.GroupVersionResource)
	pluralToSingular := make(map[schema.GroupVersionResource]schema.GroupVersionResource)
	aliasToResource := make(map[string][]string)
	// TODO: verify name mappings work correctly when versions differ

	return &DefaultRESTMapper{
		resourceToKind:       resourceToKind,
		kindToPluralResource: kindToPluralResource,
		kindToScope:          kindToScope,
		defaultGroupVersions: defaultGroupVersions,
		singularToPlural:     singularToPlural,
		pluralToSingular:     pluralToSingular,
		aliasToResource:      aliasToResource,
		interfacesFunc:       f,
	}
}

func (m *DefaultRESTMapper) Add(kind schema.GroupVersionKind, scope RESTScope) {
	plural, singular := KindToResource(kind)

	m.singularToPlural[singular] = plural
	m.pluralToSingular[plural] = singular

	m.resourceToKind[singular] = kind
	m.resourceToKind[plural] = kind

	m.kindToPluralResource[kind] = plural
	m.kindToScope[kind] = scope
}

// unpluralizedSuffixes is a list of resource suffixes that are the same plural and singular
// This is only is only necessary because some bits of code are lazy and don't actually use the RESTMapper like they should.
// TODO eliminate this so that different callers can correctly map to resources.  This probably means updating all
// callers to use the RESTMapper they mean.
var unpluralizedSuffixes = []string{
	"endpoints",
}

// KindToResource converts Kind to a resource name.
// Broken. This method only "sort of" works when used outside of this package.  It assumes that Kinds and Resources match
// and they aren't guaranteed to do so.
func KindToResource(kind schema.GroupVersionKind) ( /*plural*/ schema.GroupVersionResource /*singular*/, schema.GroupVersionResource) {
	kindName := kind.Kind
	if len(kindName) == 0 {
		return schema.GroupVersionResource{}, schema.GroupVersionResource{}
	}
	singularName := strings.ToLower(kindName)
	singular := kind.GroupVersion().WithResource(singularName)

	for _, skip := range unpluralizedSuffixes {
		if strings.HasSuffix(singularName, skip) {
			return singular, singular
		}
	}

	switch string(singularName[len(singularName)-1]) {
	case "s":
		return kind.GroupVersion().WithResource(singularName + "es"), singular
	case "y":
		return kind.GroupVersion().WithResource(strings.TrimSuffix(singularName, "y") + "ies"), singular
	}

	return kind.GroupVersion().WithResource(singularName + "s"), singular
}

// ResourceSingularizer implements RESTMapper
// It converts a resource name from plural to singular (e.g., from pods to pod)
func (m *DefaultRESTMapper) ResourceSingularizer(resourceType string) (string, error) {
	partialResource := schema.GroupVersionResource{Resource: resourceType}
	resources, err := m.ResourcesFor(partialResource)
	if err != nil {
		return resourceType, err
	}

	singular := schema.GroupVersionResource{}
	for _, curr := range resources {
		currSingular, ok := m.pluralToSingular[curr]
		if !ok {
			continue
		}
		if singular.Empty() {
			singular = currSingular
			continue
		}

		if currSingular.Resource != singular.Resource {
			return resourceType, fmt.Errorf("multiple possible singular resources (%v) found for %v", resources, resourceType)
		}
	}

	if singular.Empty() {
		return resourceType, fmt.Errorf("no singular of resource %v has been defined", resourceType)
	}

	return singular.Resource, nil
}

// coerceResourceForMatching makes the resource lower case and converts internal versions to unspecified (legacy behavior)
func coerceResourceForMatching(resource schema.GroupVersionResource) schema.GroupVersionResource {
	resource.Resource = strings.ToLower(resource.Resource)
	if resource.Version == runtime.APIVersionInternal {
		resource.Version = ""
	}

	return resource
}

func (m *DefaultRESTMapper) ResourcesFor(input schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	resource := coerceResourceForMatching(input)

	hasResource := len(resource.Resource) > 0
	hasGroup := len(resource.Group) > 0
	hasVersion := len(resource.Version) > 0

	if !hasResource {
		return nil, fmt.Errorf("a resource must be present, got: %v", resource)
	}

	ret := []schema.GroupVersionResource{}
	switch {
	case hasGroup && hasVersion:
		// fully qualified.  Find the exact match
		for plural, singular := range m.pluralToSingular {
			if singular == resource {
				ret = append(ret, plural)
				break
			}
			if plural == resource {
				ret = append(ret, plural)
				break
			}
		}

	case hasGroup:
		// given a group, prefer an exact match.  If you don't find one, resort to a prefix match on group
		foundExactMatch := false
		requestedGroupResource := resource.GroupResource()
		for plural, singular := range m.pluralToSingular {
			if singular.GroupResource() == requestedGroupResource {
				foundExactMatch = true
				ret = append(ret, plural)
			}
			if plural.GroupResource() == requestedGroupResource {
				foundExactMatch = true
				ret = append(ret, plural)
			}
		}

		// if you didn't find an exact match, match on group prefixing. This allows storageclass.storage to match
		// storageclass.storage.k8s.io
		if !foundExactMatch {
			for plural, singular := range m.pluralToSingular {
				if !strings.HasPrefix(plural.Group, requestedGroupResource.Group) {
					continue
				}
				if singular.Resource == requestedGroupResource.Resource {
					ret = append(ret, plural)
				}
				if plural.Resource == requestedGroupResource.Resource {
					ret = append(ret, plural)
				}
			}

		}

	case hasVersion:
		for plural, singular := range m.pluralToSingular {
			if singular.Version == resource.Version && singular.Resource == resource.Resource {
				ret = append(ret, plural)
			}
			if plural.Version == resource.Version && plural.Resource == resource.Resource {
				ret = append(ret, plural)
			}
		}

	default:
		for plural, singular := range m.pluralToSingular {
			if singular.Resource == resource.Resource {
				ret = append(ret, plural)
			}
			if plural.Resource == resource.Resource {
				ret = append(ret, plural)
			}
		}
	}

	if len(ret) == 0 {
		return nil, &NoResourceMatchError{PartialResource: resource}
	}

	sort.Sort(resourceByPreferredGroupVersion{ret, m.defaultGroupVersions})
	return ret, nil
}

func (m *DefaultRESTMapper) ResourceFor(resource schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	resources, err := m.ResourcesFor(resource)
	if err != nil {
		return schema.GroupVersionResource{}, err
	}
	if len(resources) == 1 {
		return resources[0], nil
	}

	return schema.GroupVersionResource{}, &AmbiguousResourceError{PartialResource: resource, MatchingResources: resources}
}

func (m *DefaultRESTMapper) KindsFor(input schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	resource := coerceResourceForMatching(input)

	hasResource := len(resource.Resource) > 0
	hasGroup := len(resource.Group) > 0
	hasVersion := len(resource.Version) > 0

	if !hasResource {
		return nil, fmt.Errorf("a resource must be present, got: %v", resource)
	}

	ret := []schema.GroupVersionKind{}
	switch {
	// fully qualified.  Find the exact match
	case hasGroup && hasVersion:
		kind, exists := m.resourceToKind[resource]
		if exists {
			ret = append(ret, kind)
		}

	case hasGroup:
		foundExactMatch := false
		requestedGroupResource := resource.GroupResource()
		for currResource, currKind := range m.resourceToKind {
			if currResource.GroupResource() == requestedGroupResource {
				foundExactMatch = true
				ret = append(ret, currKind)
			}
		}

		// if you didn't find an exact match, match on group prefixing. This allows storageclass.storage to match
		// storageclass.storage.k8s.io
		if !foundExactMatch {
			for currResource, currKind := range m.resourceToKind {
				if !strings.HasPrefix(currResource.Group, requestedGroupResource.Group) {
					continue
				}
				if currResource.Resource == requestedGroupResource.Resource {
					ret = append(ret, currKind)
				}
			}

		}

	case hasVersion:
		for currResource, currKind := range m.resourceToKind {
			if currResource.Version == resource.Version && currResource.Resource == resource.Resource {
				ret = append(ret, currKind)
			}
		}

	default:
		for currResource, currKind := range m.resourceToKind {
			if currResource.Resource == resource.Resource {
				ret = append(ret, currKind)
			}
		}
	}

	if len(ret) == 0 {
		return nil, &NoResourceMatchError{PartialResource: input}
	}

	sort.Sort(kindByPreferredGroupVersion{ret, m.defaultGroupVersions})
	return ret, nil
}

func (m *DefaultRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	kinds, err := m.KindsFor(resource)
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	if len(kinds) == 1 {
		return kinds[0], nil
	}

	return schema.GroupVersionKind{}, &AmbiguousResourceError{PartialResource: resource, MatchingKinds: kinds}
}

type kindByPreferredGroupVersion struct {
	list      []schema.GroupVersionKind
	sortOrder []schema.GroupVersion
}

func (o kindByPreferredGroupVersion) Len() int      { return len(o.list) }
func (o kindByPreferredGroupVersion) Swap(i, j int) { o.list[i], o.list[j] = o.list[j], o.list[i] }
func (o kindByPreferredGroupVersion) Less(i, j int) bool {
	lhs := o.list[i]
	rhs := o.list[j]
	if lhs == rhs {
		return false
	}

	if lhs.GroupVersion() == rhs.GroupVersion() {
		return lhs.Kind < rhs.Kind
	}

	// otherwise, the difference is in the GroupVersion, so we need to sort with respect to the preferred order
	lhsIndex := -1
	rhsIndex := -1

	for i := range o.sortOrder {
		if o.sortOrder[i] == lhs.GroupVersion() {
			lhsIndex = i
		}
		if o.sortOrder[i] == rhs.GroupVersion() {
			rhsIndex = i
		}
	}

	if rhsIndex == -1 {
		return true
	}

	return lhsIndex < rhsIndex
}

type resourceByPreferredGroupVersion struct {
	list      []schema.GroupVersionResource
	sortOrder []schema.GroupVersion
}

func (o resourceByPreferredGroupVersion) Len() int      { return len(o.list) }
func (o resourceByPreferredGroupVersion) Swap(i, j int) { o.list[i], o.list[j] = o.list[j], o.list[i] }
func (o resourceByPreferredGroupVersion) Less(i, j int) bool {
	lhs := o.list[i]
	rhs := o.list[j]
	if lhs == rhs {
		return false
	}

	if lhs.GroupVersion() == rhs.GroupVersion() {
		return lhs.Resource < rhs.Resource
	}

	// otherwise, the difference is in the GroupVersion, so we need to sort with respect to the preferred order
	lhsIndex := -1
	rhsIndex := -1

	for i := range o.sortOrder {
		if o.sortOrder[i] == lhs.GroupVersion() {
			lhsIndex = i
		}
		if o.sortOrder[i] == rhs.GroupVersion() {
			rhsIndex = i
		}
	}

	if rhsIndex == -1 {
		return true
	}

	return lhsIndex < rhsIndex
}

// RESTMapping returns a struct representing the resource path and conversion interfaces a
// RESTClient should use to operate on the provided group/kind in order of versions. If a version search
// order is not provided, the search order provided to DefaultRESTMapper will be used to resolve which
// version should be used to access the named group/kind.
// TODO: consider refactoring to use RESTMappings in a way that preserves version ordering and preference
func (m *DefaultRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*RESTMapping, error) {
	// Pick an appropriate version
	var gvk *schema.GroupVersionKind
	hadVersion := false
	for _, version := range versions {
		if len(version) == 0 || version == runtime.APIVersionInternal {
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
		return nil, &NoKindMatchError{PartialKind: gk.WithVersion("")}
	}

	// Ensure we have a REST mapping
	resource, ok := m.kindToPluralResource[*gvk]
	if !ok {
		found := []schema.GroupVersion{}
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

	interfaces, err := m.interfacesFunc(gvk.GroupVersion())
	if err != nil {
		return nil, fmt.Errorf("the provided version %q has no relevant versions: %v", gvk.GroupVersion().String(), err)
	}

	retVal := &RESTMapping{
		Resource:         resource.Resource,
		GroupVersionKind: *gvk,
		Scope:            scope,

		ObjectConvertor:  interfaces.ObjectConvertor,
		MetadataAccessor: interfaces.MetadataAccessor,
	}

	return retVal, nil
}

// RESTMappings returns the RESTMappings for the provided group kind in a rough internal preferred order. If no
// kind is found it will return a NoResourceMatchError.
func (m *DefaultRESTMapper) RESTMappings(gk schema.GroupKind) ([]*RESTMapping, error) {
	// Use the default preferred versions
	var mappings []*RESTMapping
	for _, gv := range m.defaultGroupVersions {
		if gv.Group != gk.Group {
			continue
		}

		gvk := gk.WithVersion(gv.Version)
		gvr, ok := m.kindToPluralResource[gvk]
		if !ok {
			continue
		}

		// Ensure we have a REST scope
		scope, ok := m.kindToScope[gvk]
		if !ok {
			return nil, fmt.Errorf("the provided version %q and kind %q cannot be mapped to a supported scope", gvk.GroupVersion(), gvk.Kind)
		}

		interfaces, err := m.interfacesFunc(gvk.GroupVersion())
		if err != nil {
			return nil, fmt.Errorf("the provided version %q has no relevant versions: %v", gvk.GroupVersion().String(), err)
		}

		mappings = append(mappings, &RESTMapping{
			Resource:         gvr.Resource,
			GroupVersionKind: gvk,
			Scope:            scope,

			ObjectConvertor:  interfaces.ObjectConvertor,
			MetadataAccessor: interfaces.MetadataAccessor,
		})
	}

	if len(mappings) == 0 {
		return nil, &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Group: gk.Group, Resource: gk.Kind}}
	}
	return mappings, nil
}

// AddResourceAlias maps aliases to resources
func (m *DefaultRESTMapper) AddResourceAlias(alias string, resources ...string) {
	if len(resources) == 0 {
		return
	}
	m.aliasToResource[alias] = resources
}

// AliasesForResource returns whether a resource has an alias or not
func (m *DefaultRESTMapper) AliasesForResource(alias string) ([]string, bool) {
	if res, ok := m.aliasToResource[alias]; ok {
		return res, true
	}
	return nil, false
}

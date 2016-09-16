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

package discovery

import (
	"sync"

	"k8s.io/client-go/1.4/pkg/api/errors"
	"k8s.io/client-go/1.4/pkg/api/meta"
	"k8s.io/client-go/1.4/pkg/api/unversioned"
)

// APIGroupResources is an API group with a mapping of versions to
// resources.
type APIGroupResources struct {
	Group unversioned.APIGroup
	// A mapping of version string to a slice of APIResources for
	// that version.
	VersionedResources map[string][]unversioned.APIResource
}

// NewRESTMapper returns a PriorityRESTMapper based on the discovered
// groups and resourced passed in.
func NewRESTMapper(groupResources []*APIGroupResources, versionInterfaces meta.VersionInterfacesFunc) meta.RESTMapper {
	unionMapper := meta.MultiRESTMapper{}

	var groupPriority []string
	var resourcePriority []unversioned.GroupVersionResource
	var kindPriority []unversioned.GroupVersionKind

	for _, group := range groupResources {
		groupPriority = append(groupPriority, group.Group.Name)

		if len(group.Group.PreferredVersion.Version) != 0 {
			preffered := group.Group.PreferredVersion.Version
			if _, ok := group.VersionedResources[preffered]; ok {
				resourcePriority = append(resourcePriority, unversioned.GroupVersionResource{
					Group:    group.Group.Name,
					Version:  group.Group.PreferredVersion.Version,
					Resource: meta.AnyResource,
				})

				kindPriority = append(kindPriority, unversioned.GroupVersionKind{
					Group:   group.Group.Name,
					Version: group.Group.PreferredVersion.Version,
					Kind:    meta.AnyKind,
				})
			}
		}

		for _, discoveryVersion := range group.Group.Versions {
			resources, ok := group.VersionedResources[discoveryVersion.Version]
			if !ok {
				continue
			}

			gv := unversioned.GroupVersion{Group: group.Group.Name, Version: discoveryVersion.Version}
			versionMapper := meta.NewDefaultRESTMapper([]unversioned.GroupVersion{gv}, versionInterfaces)

			for _, resource := range resources {
				scope := meta.RESTScopeNamespace
				if !resource.Namespaced {
					scope = meta.RESTScopeRoot
				}
				versionMapper.Add(gv.WithKind(resource.Kind), scope)
				// TODO only do this if it supports listing
				versionMapper.Add(gv.WithKind(resource.Kind+"List"), scope)
			}
			// TODO why is this type not in discovery (at least for "v1")
			versionMapper.Add(gv.WithKind("List"), meta.RESTScopeRoot)
			unionMapper = append(unionMapper, versionMapper)
		}
	}

	for _, group := range groupPriority {
		resourcePriority = append(resourcePriority, unversioned.GroupVersionResource{
			Group:    group,
			Version:  meta.AnyVersion,
			Resource: meta.AnyResource,
		})
		kindPriority = append(kindPriority, unversioned.GroupVersionKind{
			Group:   group,
			Version: meta.AnyVersion,
			Kind:    meta.AnyKind,
		})
	}

	return meta.PriorityRESTMapper{
		Delegate:         unionMapper,
		ResourcePriority: resourcePriority,
		KindPriority:     kindPriority,
	}
}

// GetAPIGroupResources uses the provided discovery client to gather
// discovery information and populate a slice of APIGroupResources.
func GetAPIGroupResources(cl DiscoveryInterface) ([]*APIGroupResources, error) {
	apiGroups, err := cl.ServerGroups()
	if err != nil {
		return nil, err
	}
	var result []*APIGroupResources
	for _, group := range apiGroups.Groups {
		groupResources := &APIGroupResources{
			Group:              group,
			VersionedResources: make(map[string][]unversioned.APIResource),
		}
		for _, version := range group.Versions {
			resources, err := cl.ServerResourcesForGroupVersion(version.GroupVersion)
			if err != nil {
				if errors.IsNotFound(err) {
					continue // ignore as this can race with deletion of 3rd party APIs
				}
				return nil, err
			}
			groupResources.VersionedResources[version.Version] = resources.APIResources
		}
		result = append(result, groupResources)
	}
	return result, nil
}

// DeferredDiscoveryRESTMapper is a RESTMapper that will defer
// initialization of the RESTMapper until the first mapping is
// requested.
type DeferredDiscoveryRESTMapper struct {
	initMu           sync.Mutex
	delegate         meta.RESTMapper
	cl               DiscoveryInterface
	versionInterface meta.VersionInterfacesFunc
}

// NewDeferredDiscoveryRESTMapper returns a
// DeferredDiscoveryRESTMapper that will lazily query the provided
// client for discovery information to do REST mappings.
func NewDeferredDiscoveryRESTMapper(cl DiscoveryInterface, versionInterface meta.VersionInterfacesFunc) *DeferredDiscoveryRESTMapper {
	return &DeferredDiscoveryRESTMapper{
		cl:               cl,
		versionInterface: versionInterface,
	}
}

func (d *DeferredDiscoveryRESTMapper) getDelegate() (meta.RESTMapper, error) {
	d.initMu.Lock()
	defer d.initMu.Unlock()

	if d.delegate != nil {
		return d.delegate, nil
	}

	groupResources, err := GetAPIGroupResources(d.cl)
	if err != nil {
		return nil, err
	}

	d.delegate = NewRESTMapper(groupResources, d.versionInterface)
	return d.delegate, err
}

// Reset resets the internally cached Discovery information and will
// cause the next mapping request to re-discover.
func (d *DeferredDiscoveryRESTMapper) Reset() {
	d.initMu.Lock()
	d.delegate = nil
	d.initMu.Unlock()
}

// KindFor takes a partial resource and returns back the single match.
// It returns an error if there are multiple matches.
func (d *DeferredDiscoveryRESTMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	del, err := d.getDelegate()
	if err != nil {
		return unversioned.GroupVersionKind{}, err
	}
	return del.KindFor(resource)
}

// KindsFor takes a partial resource and returns back the list of
// potential kinds in priority order.
func (d *DeferredDiscoveryRESTMapper) KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return del.KindsFor(resource)
}

// ResourceFor takes a partial resource and returns back the single
// match. It returns an error if there are multiple matches.
func (d *DeferredDiscoveryRESTMapper) ResourceFor(input unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	del, err := d.getDelegate()
	if err != nil {
		return unversioned.GroupVersionResource{}, err
	}
	return del.ResourceFor(input)
}

// ResourcesFor takes a partial resource and returns back the list of
// potential resource in priority order.
func (d *DeferredDiscoveryRESTMapper) ResourcesFor(input unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return del.ResourcesFor(input)
}

// RESTMapping identifies a preferred resource mapping for the
// provided group kind.
func (d *DeferredDiscoveryRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return del.RESTMapping(gk, versions...)
}

// RESTMappings returns the RESTMappings for the provided group kind
// in a rough internal preferred order. If no kind is found, it will
// return a NoResourceMatchError.
func (d *DeferredDiscoveryRESTMapper) RESTMappings(gk unversioned.GroupKind) ([]*meta.RESTMapping, error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return del.RESTMappings(gk)
}

// AliasesForResource returns whether a resource has an alias or not.
func (d *DeferredDiscoveryRESTMapper) AliasesForResource(resource string) ([]string, bool) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, false
	}
	return del.AliasesForResource(resource)
}

// ResourceSingularizer converts a resource name from plural to
// singular (e.g., from pods to pod).
func (d *DeferredDiscoveryRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return resource, err
	}
	return del.ResourceSingularizer(resource)
}

// Make sure it satisfies the interface
var _ meta.RESTMapper = &DeferredDiscoveryRESTMapper{}

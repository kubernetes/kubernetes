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

package restmapper

import (
	"fmt"
	"strings"
	"sync"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/discovery"

	"k8s.io/klog/v2"
)

// APIGroupResources is an API group with a mapping of versions to
// resources.
type APIGroupResources struct {
	Group metav1.APIGroup
	// A mapping of version string to a slice of APIResources for
	// that version.
	VersionedResources map[string][]metav1.APIResource
}

// NewDiscoveryRESTMapper returns a PriorityRESTMapper based on the discovered
// groups and resources passed in.
func NewDiscoveryRESTMapper(groupResources []*APIGroupResources) meta.RESTMapper {
	unionMapper := meta.MultiRESTMapper{}

	var groupPriority []string
	// /v1 is special.  It should always come first
	resourcePriority := []schema.GroupVersionResource{{Group: "", Version: "v1", Resource: meta.AnyResource}}
	kindPriority := []schema.GroupVersionKind{{Group: "", Version: "v1", Kind: meta.AnyKind}}

	for _, group := range groupResources {
		groupPriority = append(groupPriority, group.Group.Name)

		// Make sure the preferred version comes first
		if len(group.Group.PreferredVersion.Version) != 0 {
			preferred := group.Group.PreferredVersion.Version
			if _, ok := group.VersionedResources[preferred]; ok {
				resourcePriority = append(resourcePriority, schema.GroupVersionResource{
					Group:    group.Group.Name,
					Version:  group.Group.PreferredVersion.Version,
					Resource: meta.AnyResource,
				})

				kindPriority = append(kindPriority, schema.GroupVersionKind{
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

			// Add non-preferred versions after the preferred version, in case there are resources that only exist in those versions
			if discoveryVersion.Version != group.Group.PreferredVersion.Version {
				resourcePriority = append(resourcePriority, schema.GroupVersionResource{
					Group:    group.Group.Name,
					Version:  discoveryVersion.Version,
					Resource: meta.AnyResource,
				})

				kindPriority = append(kindPriority, schema.GroupVersionKind{
					Group:   group.Group.Name,
					Version: discoveryVersion.Version,
					Kind:    meta.AnyKind,
				})
			}

			gv := schema.GroupVersion{Group: group.Group.Name, Version: discoveryVersion.Version}
			versionMapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{gv})

			for _, resource := range resources {
				scope := meta.RESTScopeNamespace
				if !resource.Namespaced {
					scope = meta.RESTScopeRoot
				}

				// if we have a slash, then this is a subresource and we shouldn't create mappings for those.
				if strings.Contains(resource.Name, "/") {
					continue
				}

				plural := gv.WithResource(resource.Name)
				singular := gv.WithResource(resource.SingularName)
				// this is for legacy resources and servers which don't list singular forms.  For those we must still guess.
				if len(resource.SingularName) == 0 {
					_, singular = meta.UnsafeGuessKindToResource(gv.WithKind(resource.Kind))
				}

				versionMapper.AddSpecific(gv.WithKind(strings.ToLower(resource.Kind)), plural, singular, scope)
				versionMapper.AddSpecific(gv.WithKind(resource.Kind), plural, singular, scope)
				// TODO this is producing unsafe guesses that don't actually work, but it matches previous behavior
				versionMapper.Add(gv.WithKind(resource.Kind+"List"), scope)
			}
			// TODO why is this type not in discovery (at least for "v1")
			versionMapper.Add(gv.WithKind("List"), meta.RESTScopeRoot)
			unionMapper = append(unionMapper, versionMapper)
		}
	}

	for _, group := range groupPriority {
		resourcePriority = append(resourcePriority, schema.GroupVersionResource{
			Group:    group,
			Version:  meta.AnyVersion,
			Resource: meta.AnyResource,
		})
		kindPriority = append(kindPriority, schema.GroupVersionKind{
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
func GetAPIGroupResources(cl discovery.DiscoveryInterface) ([]*APIGroupResources, error) {
	gs, rs, err := cl.ServerGroupsAndResources()
	if rs == nil || gs == nil {
		return nil, err
		// TODO track the errors and update callers to handle partial errors.
	}
	rsm := map[string]*metav1.APIResourceList{}
	for _, r := range rs {
		rsm[r.GroupVersion] = r
	}

	var result []*APIGroupResources
	for _, group := range gs {
		groupResources := &APIGroupResources{
			Group:              *group,
			VersionedResources: make(map[string][]metav1.APIResource),
		}
		for _, version := range group.Versions {
			resources, ok := rsm[version.GroupVersion]
			if !ok {
				continue
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
	initMu   sync.Mutex
	delegate meta.RESTMapper
	cl       discovery.CachedDiscoveryInterface
}

// NewDeferredDiscoveryRESTMapper returns a
// DeferredDiscoveryRESTMapper that will lazily query the provided
// client for discovery information to do REST mappings.
func NewDeferredDiscoveryRESTMapper(cl discovery.CachedDiscoveryInterface) *DeferredDiscoveryRESTMapper {
	return &DeferredDiscoveryRESTMapper{
		cl: cl,
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

	d.delegate = NewDiscoveryRESTMapper(groupResources)
	return d.delegate, nil
}

// Reset resets the internally cached Discovery information and will
// cause the next mapping request to re-discover.
func (d *DeferredDiscoveryRESTMapper) Reset() {
	klog.V(5).Info("Invalidating discovery information")

	d.initMu.Lock()
	defer d.initMu.Unlock()

	d.cl.Invalidate()
	d.delegate = nil
}

// KindFor takes a partial resource and returns back the single match.
// It returns an error if there are multiple matches.
func (d *DeferredDiscoveryRESTMapper) KindFor(resource schema.GroupVersionResource) (gvk schema.GroupVersionKind, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return schema.GroupVersionKind{}, err
	}
	gvk, err = del.KindFor(resource)
	if err != nil && !d.cl.Fresh() {
		d.Reset()
		gvk, err = d.KindFor(resource)
	}
	return
}

// KindsFor takes a partial resource and returns back the list of
// potential kinds in priority order.
func (d *DeferredDiscoveryRESTMapper) KindsFor(resource schema.GroupVersionResource) (gvks []schema.GroupVersionKind, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	gvks, err = del.KindsFor(resource)
	if len(gvks) == 0 && !d.cl.Fresh() {
		d.Reset()
		gvks, err = d.KindsFor(resource)
	}
	return
}

// ResourceFor takes a partial resource and returns back the single
// match. It returns an error if there are multiple matches.
func (d *DeferredDiscoveryRESTMapper) ResourceFor(input schema.GroupVersionResource) (gvr schema.GroupVersionResource, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return schema.GroupVersionResource{}, err
	}
	gvr, err = del.ResourceFor(input)
	if err != nil && !d.cl.Fresh() {
		d.Reset()
		gvr, err = d.ResourceFor(input)
	}
	return
}

// ResourcesFor takes a partial resource and returns back the list of
// potential resource in priority order.
func (d *DeferredDiscoveryRESTMapper) ResourcesFor(input schema.GroupVersionResource) (gvrs []schema.GroupVersionResource, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	gvrs, err = del.ResourcesFor(input)
	if len(gvrs) == 0 && !d.cl.Fresh() {
		d.Reset()
		gvrs, err = d.ResourcesFor(input)
	}
	return
}

// RESTMapping identifies a preferred resource mapping for the
// provided group kind.
func (d *DeferredDiscoveryRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (m *meta.RESTMapping, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	m, err = del.RESTMapping(gk, versions...)
	if err != nil && !d.cl.Fresh() {
		d.Reset()
		m, err = d.RESTMapping(gk, versions...)
	}
	return
}

// RESTMappings returns the RESTMappings for the provided group kind
// in a rough internal preferred order. If no kind is found, it will
// return a NoResourceMatchError.
func (d *DeferredDiscoveryRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) (ms []*meta.RESTMapping, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	ms, err = del.RESTMappings(gk, versions...)
	if len(ms) == 0 && !d.cl.Fresh() {
		d.Reset()
		ms, err = d.RESTMappings(gk, versions...)
	}
	return
}

// ResourceSingularizer converts a resource name from plural to
// singular (e.g., from pods to pod).
func (d *DeferredDiscoveryRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	del, err := d.getDelegate()
	if err != nil {
		return resource, err
	}
	singular, err = del.ResourceSingularizer(resource)
	if err != nil && !d.cl.Fresh() {
		d.Reset()
		singular, err = d.ResourceSingularizer(resource)
	}
	return
}

func (d *DeferredDiscoveryRESTMapper) String() string {
	del, err := d.getDelegate()
	if err != nil {
		return fmt.Sprintf("DeferredDiscoveryRESTMapper{%v}", err)
	}
	return fmt.Sprintf("DeferredDiscoveryRESTMapper{\n\t%v\n}", del)
}

// Make sure it satisfies the interface
var _ meta.ResettableRESTMapper = &DeferredDiscoveryRESTMapper{}

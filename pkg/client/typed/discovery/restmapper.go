package discovery

import (
	"sync"

	kapi "k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
)

type discoveryRESTMapper struct {
	discoveryClient DiscoveryInterface

	delegate meta.RESTMapper

	initLock sync.Mutex
}

// NewRESTMapper that initializes using the discovery APIs, relying on group ordering and preferred versions
// to build its appropriate priorities.  Only versions are registered with API machinery are added now.
// TODO make this work with generic resources at some point.  For now, this handles enabled and disabled resources cleanly.
func NewRESTMapper(discoveryClient DiscoveryInterface) meta.RESTMapper {
	return &discoveryRESTMapper{discoveryClient: discoveryClient}
}

func (d *discoveryRESTMapper) getDelegate() (meta.RESTMapper, error) {
	d.initLock.Lock()
	defer d.initLock.Unlock()

	if d.delegate != nil {
		return d.delegate, nil
	}

	serverGroups, err := d.discoveryClient.ServerGroups()
	if err != nil {
		return nil, err
	}

	// always prefer our default group for now.  The version should be discovered from discovery, but this will hold us
	// for quite some time.
	resourcePriority := []unversioned.GroupVersionResource{
		{Group: kapi.GroupName, Version: meta.AnyVersion, Resource: meta.AnyResource},
	}
	kindPriority := []unversioned.GroupVersionKind{
		{Group: kapi.GroupName, Version: meta.AnyVersion, Kind: meta.AnyKind},
	}
	groupPriority := []string{}

	unionMapper := meta.MultiRESTMapper{}

	for _, group := range serverGroups.Groups {
		if len(group.Versions) == 0 {
			continue
		}
		groupPriority = append(groupPriority, group.Name)

		if len(group.PreferredVersion.Version) != 0 {
			preferredVersion := unversioned.GroupVersion{Group: group.Name, Version: group.PreferredVersion.Version}
			if registered.IsEnabledVersion(preferredVersion) {
				resourcePriority = append(resourcePriority, preferredVersion.WithResource(meta.AnyResource))
				kindPriority = append(kindPriority, preferredVersion.WithKind(meta.AnyKind))
			}
		}

		for _, discoveryVersion := range group.Versions {
			version := unversioned.GroupVersion{Group: group.Name, Version: discoveryVersion.Version}
			if !registered.IsEnabledVersion(version) {
				continue
			}
			groupMeta, err := registered.Group(group.Name)
			if err != nil {
				return nil, err
			}
			resources, err := d.discoveryClient.ServerResourcesForGroupVersion(version.String())
			if err != nil {
				return nil, err
			}

			versionMapper := meta.NewDefaultRESTMapper([]unversioned.GroupVersion{version}, groupMeta.InterfacesFor)
			for _, resource := range resources.APIResources {
				// TODO properly handle resource versus kind
				gvk := version.WithKind(resource.Kind)

				scope := meta.RESTScopeNamespace
				if !resource.Namespaced {
					scope = meta.RESTScopeRoot
				}
				versionMapper.Add(gvk, scope)

				// TODO formalize this by checking to see if they support listing
				versionMapper.Add(version.WithKind(resource.Kind+"List"), scope)
			}

			// we need to add List.  Its a special case of something we need that isn't in the discovery doc
			if group.Name == kapi.GroupName {
				versionMapper.Add(version.WithKind("List"), meta.RESTScopeNamespace)
			}

			unionMapper = append(unionMapper, versionMapper)
		}
	}

	for _, group := range groupPriority {
		resourcePriority = append(resourcePriority, unversioned.GroupVersionResource{Group: group, Version: meta.AnyVersion, Resource: meta.AnyResource})
		kindPriority = append(kindPriority, unversioned.GroupVersionKind{Group: group, Version: meta.AnyVersion, Kind: meta.AnyKind})
	}

	return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}, nil
}

func (d *discoveryRESTMapper) KindFor(resource unversioned.GroupVersionResource) (unversioned.GroupVersionKind, error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return unversioned.GroupVersionKind{}, err
	}
	return delegate.KindFor(resource)
}

func (d *discoveryRESTMapper) KindsFor(resource unversioned.GroupVersionResource) ([]unversioned.GroupVersionKind, error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return delegate.KindsFor(resource)
}

func (d *discoveryRESTMapper) ResourceFor(input unversioned.GroupVersionResource) (unversioned.GroupVersionResource, error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return unversioned.GroupVersionResource{}, err
	}
	return delegate.ResourceFor(input)
}

func (d *discoveryRESTMapper) ResourcesFor(input unversioned.GroupVersionResource) ([]unversioned.GroupVersionResource, error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return delegate.ResourcesFor(input)
}

func (d *discoveryRESTMapper) RESTMapping(gk unversioned.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return nil, err
	}
	return delegate.RESTMapping(gk, versions...)
}

func (d *discoveryRESTMapper) AliasesForResource(resource string) ([]string, bool) {
	delegate, err := d.getDelegate()
	if err != nil {
		return nil, false
	}
	return delegate.AliasesForResource(resource)
}

func (d *discoveryRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	delegate, err := d.getDelegate()
	if err != nil {
		return resource, err
	}
	return delegate.ResourceSingularizer(resource)
}

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
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
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

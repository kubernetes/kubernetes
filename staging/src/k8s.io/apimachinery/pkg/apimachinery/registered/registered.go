/*
Copyright 2015 The Kubernetes Authors.

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

// Package to keep track of API Versions that can be registered and are enabled in a Scheme.
package registered

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// APIRegistrationManager provides the concept of what API groups are enabled.
//
// TODO: currently, it also provides a "registered" concept. But it's wrong to
// have both concepts in the same object. Therefore the "announced" package is
// going to take over the registered concept. After all the install packages
// are switched to using the announce package instead of this package, then we
// can combine the registered/enabled concepts in this object. Simplifying this
// isn't easy right now because there are so many callers of this package.
type APIRegistrationManager struct {
	// registeredGroupVersions stores all API group versions for which RegisterGroup is called.
	registeredVersions map[schema.GroupVersion]struct{}

	// map of group meta for all groups.
	groupMetaMap map[string]*apimachinery.GroupMeta
}

// NewAPIRegistrationManager constructs a new manager.
func NewAPIRegistrationManager() *APIRegistrationManager {
	m := &APIRegistrationManager{
		registeredVersions: map[schema.GroupVersion]struct{}{},
		groupMetaMap:       map[string]*apimachinery.GroupMeta{},
	}

	return m
}

// RegisterVersions adds the given group versions to the list of registered group versions.
func (m *APIRegistrationManager) RegisterVersions(availableVersions []schema.GroupVersion) {
	for _, v := range availableVersions {
		m.registeredVersions[v] = struct{}{}
	}
}

// RegisterGroup adds the given group to the list of registered groups.
func (m *APIRegistrationManager) RegisterGroup(groupMeta apimachinery.GroupMeta) error {
	groupName := groupMeta.GroupVersions[0].Group
	if _, found := m.groupMetaMap[groupName]; found {
		return fmt.Errorf("group %q is already registered in groupsMap: %v", groupName, m.groupMetaMap)
	}
	m.groupMetaMap[groupName] = &groupMeta
	return nil
}

// Group returns the metadata of a group if the group is registered, otherwise
// an error is returned.
func (m *APIRegistrationManager) Group(group string) (*apimachinery.GroupMeta, error) {
	groupMeta, found := m.groupMetaMap[group]
	if !found {
		return nil, fmt.Errorf("group %v has not been registered", group)
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy, nil
}

// IsRegistered takes a string and determines if it's one of the registered groups
func (m *APIRegistrationManager) IsRegistered(group string) bool {
	_, found := m.groupMetaMap[group]
	return found
}

// IsRegisteredVersion returns if a version is registered.
func (m *APIRegistrationManager) IsRegisteredVersion(v schema.GroupVersion) bool {
	_, found := m.registeredVersions[v]
	return found
}

// RegisteredGroupVersions returns all registered group versions.  Groups are randomly ordered, but versions within groups
// are priority order from best to worst
func (m *APIRegistrationManager) RegisteredGroupVersions() []schema.GroupVersion {
	ret := []schema.GroupVersion{}
	for _, groupMeta := range m.groupMetaMap {
		for _, version := range groupMeta.GroupVersions {
			if m.IsRegisteredVersion(version) {
				ret = append(ret, version)
			}
		}
	}
	return ret
}

// RegisteredVersionsForGroup returns all enabled versions for a group in order of best to worst
func (m *APIRegistrationManager) RegisteredVersionsForGroup(group string) []schema.GroupVersion {
	groupMeta, ok := m.groupMetaMap[group]
	if !ok {
		return []schema.GroupVersion{}
	}

	ret := []schema.GroupVersion{}
	for _, version := range groupMeta.GroupVersions {
		if m.IsRegisteredVersion(version) {
			ret = append(ret, version)
		}
	}
	return ret
}

// InterfacesFor is a union meta.VersionInterfacesFunc func for all registered types
func (m *APIRegistrationManager) InterfacesFor(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	groupMeta, err := m.Group(version.Group)
	if err != nil {
		return nil, err
	}
	return groupMeta.InterfacesFor(version)
}

// TODO: This is an expedient function, because we don't check if a Group is
// supported throughout the code base. We will abandon this function and
// checking the error returned by the Group() function.
func (m *APIRegistrationManager) GroupOrDie(group string) *apimachinery.GroupMeta {
	groupMeta, found := m.groupMetaMap[group]
	if !found {
		if group == "" {
			panic("The legacy v1 API is not registered.")
		} else {
			panic(fmt.Sprintf("Group %s is not registered.", group))
		}
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy
}

// RESTMapper returns a union RESTMapper of all known types with priorities chosen in the following order:
//  1. legacy kube group preferred version, extensions preferred version, metrics perferred version, legacy
//     kube any version, extensions any version, metrics any version, all other groups alphabetical preferred version,
//     all other groups alphabetical.
func (m *APIRegistrationManager) RESTMapper(versionPatterns ...schema.GroupVersion) meta.RESTMapper {
	unionMapper := meta.MultiRESTMapper{}
	unionedGroups := sets.NewString()
	for enabledVersion := range m.registeredVersions {
		if !unionedGroups.Has(enabledVersion.Group) {
			unionedGroups.Insert(enabledVersion.Group)
			groupMeta := m.groupMetaMap[enabledVersion.Group]
			if groupMeta != nil {
				unionMapper = append(unionMapper, groupMeta.RESTMapper)
			}
		}
	}

	if len(versionPatterns) != 0 {
		resourcePriority := []schema.GroupVersionResource{}
		kindPriority := []schema.GroupVersionKind{}
		for _, versionPriority := range versionPatterns {
			resourcePriority = append(resourcePriority, versionPriority.WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, versionPriority.WithKind(meta.AnyKind))
		}

		return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
	}

	prioritizedGroups := []string{"", "extensions", "metrics"}
	resourcePriority, kindPriority := m.prioritiesForGroups(prioritizedGroups...)

	prioritizedGroupsSet := sets.NewString(prioritizedGroups...)
	remainingGroups := sets.String{}
	for enabledVersion := range m.registeredVersions {
		if !prioritizedGroupsSet.Has(enabledVersion.Group) {
			remainingGroups.Insert(enabledVersion.Group)
		}
	}

	remainingResourcePriority, remainingKindPriority := m.prioritiesForGroups(remainingGroups.List()...)
	resourcePriority = append(resourcePriority, remainingResourcePriority...)
	kindPriority = append(kindPriority, remainingKindPriority...)

	return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
}

// prioritiesForGroups returns the resource and kind priorities for a PriorityRESTMapper, preferring the preferred version of each group first,
// then any non-preferred version of the group second.
func (m *APIRegistrationManager) prioritiesForGroups(groups ...string) ([]schema.GroupVersionResource, []schema.GroupVersionKind) {
	resourcePriority := []schema.GroupVersionResource{}
	kindPriority := []schema.GroupVersionKind{}

	for _, group := range groups {
		availableVersions := m.RegisteredVersionsForGroup(group)
		if len(availableVersions) > 0 {
			resourcePriority = append(resourcePriority, availableVersions[0].WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, availableVersions[0].WithKind(meta.AnyKind))
		}
	}
	for _, group := range groups {
		resourcePriority = append(resourcePriority, schema.GroupVersionResource{Group: group, Version: meta.AnyVersion, Resource: meta.AnyResource})
		kindPriority = append(kindPriority, schema.GroupVersionKind{Group: group, Version: meta.AnyVersion, Kind: meta.AnyKind})
	}

	return resourcePriority, kindPriority
}

// AllPreferredGroupVersions returns the preferred versions of all registered
// groups in the form of "group1/version1,group2/version2,..."
func (m *APIRegistrationManager) AllPreferredGroupVersions() string {
	if len(m.groupMetaMap) == 0 {
		return ""
	}
	var defaults []string
	for _, groupMeta := range m.groupMetaMap {
		defaults = append(defaults, groupMeta.GroupVersions[0].String())
	}
	sort.Strings(defaults)
	return strings.Join(defaults, ",")
}

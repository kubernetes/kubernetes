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

	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/runtime/schema"
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

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

package latest

import (
	"fmt"
	"sort"
	"strings"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	allGroups = GroupMetaMap{}
	// Group is a shortcut to allGroups.Group.
	Group = allGroups.Group
	// RegisterGroup is a shortcut to allGroups.RegisterGroup.
	RegisterGroup = allGroups.RegisterGroup
	// GroupOrDie is a shortcut to allGroups.GroupOrDie.
	GroupOrDie = allGroups.GroupOrDie
	// AllPreferredGroupVersions returns the preferred versions of all
	// registered groups in the form of "group1/version1,group2/version2,..."
	AllPreferredGroupVersions = allGroups.AllPreferredGroupVersions
	// IsRegistered is a shortcut to allGroups.IsRegistered.
	IsRegistered = allGroups.IsRegistered
)

// ExternalVersions is a list of all external versions for this API group in order of
// most preferred to least preferred
var ExternalVersions = []unversioned.GroupVersion{
	{Group: "", Version: "v1"},
}

// GroupMetaMap is a map between group names and their metadata.
type GroupMetaMap map[string]*GroupMeta

// RegisterGroup registers a group to GroupMetaMap.
func (g GroupMetaMap) RegisterGroup(groupMeta GroupMeta) error {
	groupName := groupMeta.GroupVersion.Group
	if _, found := g[groupName]; found {
		return fmt.Errorf("group %v is already registered", g)
	}

	g[groupName] = &groupMeta
	return nil
}

// Group returns the metadata of a group if the gruop is registered, otherwise
// an erorr is returned.
func (g GroupMetaMap) Group(group string) (*GroupMeta, error) {
	groupMeta, found := g[group]
	if !found {
		return nil, fmt.Errorf("no version is registered for group %v", group)
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy, nil
}

// IsRegistered takes a string and determines if it's one of the registered groups
func (g GroupMetaMap) IsRegistered(group string) bool {
	_, found := g[group]
	return found
}

// TODO: This is an expedient function, because we don't check if a Group is
// supported throughout the code base. We will abandon this function and
// checking the error returned by the Group() function.
func (g GroupMetaMap) GroupOrDie(group string) *GroupMeta {
	groupMeta, found := g[group]
	if !found {
		const msg = "Please check the KUBE_API_VERSIONS environment variable."
		if group == "" {
			panic("The legacy v1 API is not registered. " + msg)
		} else {
			panic(fmt.Sprintf("No version is registered for group %s. ", group) + msg)
		}
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy
}

// AllPreferredGroupVersions returns the preferred versions of all registered
// groups in the form of "group1/version1,group2/version2,..."
func (g GroupMetaMap) AllPreferredGroupVersions() string {
	if len(g) == 0 {
		return ""
	}
	var defaults []string
	for _, groupMeta := range g {
		defaults = append(defaults, groupMeta.GroupVersion.String())
	}
	sort.Strings(defaults)
	return strings.Join(defaults, ",")
}

// GroupMeta stores the metadata of a group, such as the latest supported version.
type GroupMeta struct {
	// GroupVersion represents the current external default version of the group.
	GroupVersion unversioned.GroupVersion

	// GroupVersions is Group + Versions. This is to avoid string concatenation
	// in many places.
	GroupVersions []unversioned.GroupVersion

	// Codec is the default codec for serializing output that should use
	// the latest supported version.  Use this Codec when writing to
	// disk, a data store that is not dynamically versioned, or in tests.
	// This codec can decode any object that Kubernetes is aware of.
	Codec runtime.Codec

	// SelfLinker can set or get the SelfLink field of all API types.
	// TODO: when versioning changes, make this part of each API definition.
	// TODO(lavalamp): Combine SelfLinker & ResourceVersioner interfaces, force all uses
	// to go through the InterfacesFor method below.
	SelfLinker runtime.SelfLinker

	// RESTMapper provides the default mapping between REST paths and the objects declared in api.Scheme and all known
	// Kubernetes versions.
	RESTMapper meta.RESTMapper

	// InterfacesFor returns the default Codec and ResourceVersioner for a given version
	// or an error if the version is not known.
	InterfacesFor func(version unversioned.GroupVersion) (*meta.VersionInterfaces, error)
}

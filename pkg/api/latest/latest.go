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

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/registered"
	"k8s.io/kubernetes/pkg/runtime"
)

var (
	allGroups     = GroupMetaMap{}
	Group         = allGroups.Group
	RegisterGroup = allGroups.RegisterGroup
	GroupOrDie    = allGroups.GroupOrDie
)

type GroupMetaMap map[string]*GroupMeta

func (g GroupMetaMap) RegisterGroup(group string) (*GroupMeta, error) {
	_, found := g[group]
	if found {
		return nil, fmt.Errorf("group %v is already registered", g)
	}
	if len(registered.GroupVersionsForGroup(group)) == 0 {
		return nil, fmt.Errorf("No version is registered for group %v", group)
	}
	g[group] = &GroupMeta{}
	return g[group], nil
}

func (g GroupMetaMap) Group(group string) (*GroupMeta, error) {
	groupMeta, found := g[group]
	if !found {
		return nil, fmt.Errorf("no version is registered for group %v", group)
	}
	return groupMeta, nil
}

// TODO: This is an expedient function, because we don't check if a Group is
// supported throughout the code base. We will abandon this function and
// checking the error returned by the Group() function.
func (g GroupMetaMap) GroupOrDie(group string) *GroupMeta {
	groupMeta, found := g[group]
	if !found {
		panic(fmt.Sprintf("no version is registered for group %v", group))
	}
	return groupMeta
}

// GroupMeta stores the metadata of a group, such as the latest supported version.
type GroupMeta struct {
	// GroupVersion represents the current external default version of the group. It
	// is in the form of "group/version".
	GroupVersion string

	// Version represents the current external default version of the group.
	// It equals to the "version" part of GroupVersion.
	Version string

	// Group represents the name of the group
	Group string

	// Versions is the list of versions that are recognized in code. The order
	// provided is assumed to be from the oldest to the newest, e.g.,
	// Versions[0] == oldest and Versions[N-1] == newest.
	// Clients may choose to prefer the latter items in the list over the former
	// items when presented with a set of versions to choose.
	Versions []string

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

	InterfacesFor func(version string) (*meta.VersionInterfaces, error)
}

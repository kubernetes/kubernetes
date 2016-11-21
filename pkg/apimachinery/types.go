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

package apimachinery

import (
	"fmt"

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/schema"
)

// GroupMeta stores the metadata of a group.
type GroupMeta struct {
	// GroupVersion represents the preferred version of the group.
	GroupVersion schema.GroupVersion

	// GroupVersions is Group + all versions in that group.
	GroupVersions []schema.GroupVersion

	// Codec is the default codec for serializing output that should use
	// the preferred version.  Use this Codec when writing to
	// disk, a data store that is not dynamically versioned, or in tests.
	// This codec can decode any object that the schema is aware of.
	Codec runtime.Codec

	// SelfLinker can set or get the SelfLink field of all API types.
	// TODO: when versioning changes, make this part of each API definition.
	// TODO(lavalamp): Combine SelfLinker & ResourceVersioner interfaces, force all uses
	// to go through the InterfacesFor method below.
	SelfLinker runtime.SelfLinker

	// RESTMapper provides the default mapping between REST paths and the objects declared in api.Scheme and all known
	// versions.
	RESTMapper meta.RESTMapper

	// InterfacesFor returns the default Codec and ResourceVersioner for a given version
	// string, or an error if the version is not known.
	// TODO: make this stop being a func pointer and always use the default
	// function provided below once every place that populates this field has been changed.
	InterfacesFor func(version schema.GroupVersion) (*meta.VersionInterfaces, error)

	// InterfacesByVersion stores the per-version interfaces.
	InterfacesByVersion map[schema.GroupVersion]*meta.VersionInterfaces
}

// DefaultInterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
// TODO: Remove the "Default" prefix.
func (gm *GroupMeta) DefaultInterfacesFor(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	if v, ok := gm.InterfacesByVersion[version]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, gm.GroupVersions)
}

// AddVersionInterfaces adds the given version to the group. Only call during
// init, after that GroupMeta objects should be immutable. Not thread safe.
// (If you use this, be sure to set .InterfacesFor = .DefaultInterfacesFor)
// TODO: remove the "Interfaces" suffix and make this also maintain the
// .GroupVersions member.
func (gm *GroupMeta) AddVersionInterfaces(version schema.GroupVersion, interfaces *meta.VersionInterfaces) error {
	if e, a := gm.GroupVersion.Group, version.Group; a != e {
		return fmt.Errorf("got a version in group %v, but am in group %v", a, e)
	}
	if gm.InterfacesByVersion == nil {
		gm.InterfacesByVersion = make(map[schema.GroupVersion]*meta.VersionInterfaces)
	}
	gm.InterfacesByVersion[version] = interfaces

	// TODO: refactor to make the below error not possible, this function
	// should *set* GroupVersions rather than depend on it.
	for _, v := range gm.GroupVersions {
		if v == version {
			return nil
		}
	}
	return fmt.Errorf("added a version interface without the corresponding version %v being in the list %#v", version, gm.GroupVersions)
}

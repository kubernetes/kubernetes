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

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
)

// GroupMeta stores the metadata of a group.
type GroupMeta struct {
	// GroupVersions is Group + all versions in that group.
	GroupVersions []schema.GroupVersion

	RootScopedKinds sets.String

	// SelfLinker can set or get the SelfLink field of all API types.
	// TODO: when versioning changes, make this part of each API definition.
	// TODO(lavalamp): Combine SelfLinker & ResourceVersioner interfaces, force all uses
	// to go through the InterfacesFor method below.
	SelfLinker runtime.SelfLinker

	// RESTMapper provides the default mapping between REST paths and the objects declared in a Scheme and all known
	// versions.
	RESTMapper meta.RESTMapper

	// versionInterfaces stores the per-version interfaces.
	versionInterfaces map[schema.GroupVersion]*meta.VersionInterfaces
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func (gm *GroupMeta) InterfacesFor(version schema.GroupVersion) (*meta.VersionInterfaces, error) {
	if v, ok := gm.versionInterfaces[version]; ok {
		return v, nil
	}
	return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", version, gm.GroupVersions)
}

// AddVersion adds the given version to the group. Only call during
// init, after that GroupMeta objects should be immutable. Not thread safe.
// (If you use this, be sure to set .InterfacesFor = .InterfacesFor)
func (gm *GroupMeta) AddVersion(version schema.GroupVersion, interfaces *meta.VersionInterfaces) error {
	if gm.GroupVersions[0].Group != version.Group {
		return fmt.Errorf("got a version in group %v, but am in group %v", version.Group, gm.GroupVersions[0].Group)
	}

	if gm.versionInterfaces == nil {
		gm.versionInterfaces = make(map[schema.GroupVersion]*meta.VersionInterfaces)
	}
	gm.versionInterfaces[version] = interfaces

	for _, v := range gm.GroupVersions {
		if v == version {
			return nil
		}
	}

	gm.GroupVersions = append(gm.GroupVersions, version)

	return nil
}

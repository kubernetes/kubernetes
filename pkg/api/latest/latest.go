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
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta2"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta3"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Version is the string that represents the current external default version.
const Version = "v1beta3"

// OldestVersion is the string that represents the oldest server version supported,
// for client code that wants to hardcode the lowest common denominator.
const OldestVersion = "v1beta1"

// Versions is the list of versions that are recognized in code. The order provided
// may be assumed to be least feature rich to most feature rich, and clients may
// choose to prefer the latter items in the list over the former items when presented
// with a set of versions to choose.
var Versions = []string{"v1beta1", "v1beta2", "v1beta3", "v1"}

// Codec is the default codec for serializing output that should use
// the latest supported version.  Use this Codec when writing to
// disk, a data store that is not dynamically versioned, or in tests.
// This codec can decode any object that Kubernetes is aware of.
var Codec = v1beta3.Codec

// accessor is the shared static metadata accessor for the API.
var accessor = meta.NewAccessor()

// SelfLinker can set or get the SelfLink field of all API types.
// TODO: when versioning changes, make this part of each API definition.
// TODO(lavalamp): Combine SelfLinker & ResourceVersioner interfaces, force all uses
// to go through the InterfacesFor method below.
var SelfLinker = runtime.SelfLinker(accessor)

// RESTMapper provides the default mapping between REST paths and the objects declared in api.Scheme and all known
// Kubernetes versions.
var RESTMapper meta.RESTMapper

// userResources is a group of resources mostly used by a kubectl user
var userResources = []string{"rc", "svc", "pods", "pvc"}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(version string) (*meta.VersionInterfaces, error) {
	switch version {
	case "v1beta1":
		return &meta.VersionInterfaces{
			Codec:            v1beta1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	case "v1beta2":
		return &meta.VersionInterfaces{
			Codec:            v1beta2.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	case "v1beta3":
		return &meta.VersionInterfaces{
			Codec:            v1beta3.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	case "v1":
		return &meta.VersionInterfaces{
			Codec:            v1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %s)", version, strings.Join(Versions, ", "))
	}
}

func init() {
	mapper := meta.NewDefaultRESTMapper(
		Versions,
		func(version string) (*meta.VersionInterfaces, bool) {
			interfaces, err := InterfacesFor(version)
			if err != nil {
				return nil, false
			}
			return interfaces, true
		},
	)
	// list of versions we support on the server
	versions := []string{"v1beta1", "v1beta2", "v1beta3", "v1"}

	// versions that used mixed case URL formats
	versionMixedCase := map[string]bool{
		"v1beta1": true,
		"v1beta2": true,
	}

	// backwards compatibility, prior to v1beta3, we identified the namespace as a query parameter
	versionToNamespaceScope := map[string]meta.RESTScope{
		"v1beta1": meta.RESTScopeNamespaceLegacy,
		"v1beta2": meta.RESTScopeNamespaceLegacy,
		"v1beta3": meta.RESTScopeNamespace,
		"v1":      meta.RESTScopeNamespace,
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	kindToRootScope := map[string]bool{
		"Node":             true,
		"Minion":           true,
		"Namespace":        true,
		"PersistentVolume": true,
	}

	// setup aliases for groups of resources
	mapper.AddResourceAlias("all", userResources...)

	// these kinds should be excluded from the list of resources
	ignoredKinds := util.NewStringSet(
		"ListOptions",
		"DeleteOptions",
		"Status",
		"ContainerManifest",
		"PodLogOptions",
		"PodExecOptions",
		"PodProxyOptions")

	// enumerate all supported versions, get the kinds, and register with the mapper how to address our resources
	for _, version := range versions {
		for kind := range api.Scheme.KnownTypes(version) {
			if ignoredKinds.Has(kind) {
				continue
			}
			mixedCase, found := versionMixedCase[version]
			if !found {
				mixedCase = false
			}
			scope := versionToNamespaceScope[version]
			_, found = kindToRootScope[kind]
			if found {
				scope = meta.RESTScopeRoot
			}
			mapper.Add(scope, kind, version, mixedCase)
		}
	}
	RESTMapper = mapper
}

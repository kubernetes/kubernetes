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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/registered"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
)

// GroupVersion is the string that represents the current external default groupVersion.
var GroupVersion string

//
var Group string

//Version is the string that represents the current external default Version.
var Version string

// OldestVersion is the string that represents the oldest server version supported,
// for client code that wants to hardcode the lowest common denominator.
var OldestVersion string

// GroupVersions is the list of groupVersions that are recognized in code. The
// order provided may be assumed to be least feature rich to most feature rich,
// and clients may choose to prefer the latter items in the list over the former
// items when presented with a set of versions to choose.
var GroupVersions []string

var Versions []string

// Codec is the default codec for serializing output that should use
// the latest supported version.  Use this Codec when writing to
// disk, a data store that is not dynamically versioned, or in tests.
// This codec can decode any object that Kubernetes is aware of.
var Codec runtime.Codec

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

const importPrefix = "k8s.io/kubernetes/pkg/api"

func init() {
	// Native v1 object has an empty group name.
	apiGroupVersions := registered.GroupVersionsForGroup("")
	// Use the first API version in the list of registered versions as the latest.
	GroupVersion = apiGroupVersions[0]
	Group = util.GetGroup(GroupVersion)
	Version = util.GetVersion(GroupVersion)
	OldestVersion = apiGroupVersions[len(apiGroupVersions)-1]
	Codec = runtime.CodecFor(api.Scheme, GroupVersion)
	// Put the registered groupVersions in GroupVersions in reverse order.
	GroupVersions = []string{}
	for i := len(apiGroupVersions) - 1; i >= 0; i-- {
		GroupVersions = append(GroupVersions, apiGroupVersions[i])
		Versions = append(Versions, util.GetVersion(apiGroupVersions[i]))
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := util.NewStringSet(
		"Node",
		"Minion",
		"Namespace",
		"PersistentVolume",
	)

	// these kinds should be excluded from the list of resources
	ignoredKinds := util.NewStringSet(
		"ListOptions",
		"DeleteOptions",
		"Status",
		"PodLogOptions",
		"PodExecOptions",
		"PodAttachOptions",
		"PodProxyOptions",
		"Daemon",
		"ThirdPartyResource",
		"ThirdPartyResourceData",
		"ThirdPartyResourceList")

	mapper := api.NewDefaultRESTMapper(Group, apiGroupVersions, InterfacesFor, importPrefix, ignoredKinds, rootScoped)
	// setup aliases for groups of resources
	mapper.AddResourceAlias("all", userResources...)
	RESTMapper = mapper
	api.RegisterRESTMapper(RESTMapper)
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(version string) (*meta.VersionInterfaces, error) {
	switch version {
	case "v1":
		return &meta.VersionInterfaces{
			Codec:            v1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: accessor,
		}, nil
	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %s)", version, strings.Join(GroupVersions, ", "))
	}
}

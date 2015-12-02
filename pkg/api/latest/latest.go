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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/registered"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// ExternalVersions lists all allowed external versions for this group from most preferred to least preferred
var ExternalVersions = []unversioned.GroupVersion{}

// PreferredExternalVersion is the most preferred external version (ExternalVersions[0])
var PreferredExternalVersion = unversioned.GroupVersion{}

// Codec is the Codec for the most preferred ExternalVersion
var Codec runtime.Codec

// RESTMapper is a RESTMapper for all versions of the API group
var RESTMapper meta.RESTMapper

var Accessor = meta.NewAccessor()

// availableVersions lists all known external versions for this group from most preferred to least preferred
var availableVersions = []unversioned.GroupVersion{v1.SchemeGroupVersion}

func init() {
	finalExternalVersions := []unversioned.GroupVersion{}

	for _, allowedVersion := range registered.GroupVersionsForGroup(api.SchemeGroupVersion.Group) {
		for _, externalVersion := range availableVersions {
			if externalVersion == allowedVersion {
				finalExternalVersions = append(finalExternalVersions, externalVersion)
			}
		}
	}

	if len(finalExternalVersions) == 0 {
		return
	}

	ExternalVersions = finalExternalVersions
	PreferredExternalVersion = ExternalVersions[0]
	Codec = runtime.CodecFor(api.Scheme, PreferredExternalVersion.String())
	RESTMapper = newRESTMapper()
}

var userResources = []string{"rc", "svc", "pods", "pvc"}

const importPrefix = "k8s.io/kubernetes/pkg/api"

func newRESTMapper() meta.RESTMapper {
	worstToBestGroupVersions := []unversioned.GroupVersion{}
	for i := len(ExternalVersions) - 1; i >= 0; i-- {
		worstToBestGroupVersions = append(worstToBestGroupVersions, ExternalVersions[i])
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString(
		"Node",
		"Namespace",
		"PersistentVolume",
		"ComponentStatus",
	)

	// these kinds should be excluded from the list of resources
	ignoredKinds := sets.NewString(
		"ListOptions",
		"DeleteOptions",
		"Status",
		"PodLogOptions",
		"PodExecOptions",
		"PodAttachOptions",
		"PodProxyOptions",
		"ThirdPartyResource",
		"ThirdPartyResourceData",
		"ThirdPartyResourceList")

	mapper := api.NewDefaultRESTMapper(worstToBestGroupVersions, InterfacesFor, importPrefix, ignoredKinds, rootScoped)
	// setup aliases for groups of resources
	mapper.AddResourceAlias("all", userResources...)
	return mapper
}

// InterfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(groupVersionString string) (*meta.VersionInterfaces, error) {
	switch groupVersionString {
	case v1.SchemeGroupVersion.String():
		return &meta.VersionInterfaces{
			Codec:            v1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: Accessor,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %v)", groupVersionString, ExternalVersions)
	}
}

func IsEnabled() bool {
	return len(ExternalVersions) > 0
}

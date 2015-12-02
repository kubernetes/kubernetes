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
	"k8s.io/kubernetes/pkg/apis/componentconfig"
	"k8s.io/kubernetes/pkg/apis/componentconfig/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
)

// ExternalVersions lists all known external versions for this group from most preferred to least preferred
var ExternalVersions = []unversioned.GroupVersion{v1alpha1.SchemeGroupVersion}

// Codec is the Codec for the most preferred ExternalVersion
var Codec = v1alpha1.Codec

// RESTMapper is a RESTMapper for all versions of the API group
var RESTMapper meta.RESTMapper

var Accessor = meta.NewAccessor()

func init() {
	finalExternalVersions := []unversioned.GroupVersion{}

	for _, allowedVersion := range registered.GroupVersionsForGroup(componentconfig.SchemeGroupVersion.Group) {
		for _, externalVersion := range ExternalVersions {
			if externalVersion == allowedVersion {
				finalExternalVersions = append(finalExternalVersions, externalVersion)
			}
		}
	}

	ExternalVersions = finalExternalVersions
	Codec = nil
	if len(ExternalVersions) > 0 {
		Codec = runtime.CodecFor(api.Scheme, ExternalVersions[0].String())
		RESTMapper = newRESTMapper()
	}
}

const importPrefix = "k8s.io/kubernetes/pkg/apis/componentconfig"

func newRESTMapper() meta.RESTMapper {
	worstToBestGroupVersions := []unversioned.GroupVersion{}
	for i := len(ExternalVersions) - 1; i >= 0; i-- {
		worstToBestGroupVersions = append(worstToBestGroupVersions, ExternalVersions[i])
	}

	// the list of kinds that are scoped at the root of the api hierarchy
	// if a kind is not enumerated here, it is assumed to have a namespace scope
	rootScoped := sets.NewString()

	ignoredKinds := sets.NewString()

	return api.NewDefaultRESTMapper(worstToBestGroupVersions, InterfacesFor, importPrefix, ignoredKinds, rootScoped)
}

// interfacesFor returns the default Codec and ResourceVersioner for a given version
// string, or an error if the version is not known.
func InterfacesFor(version string) (*meta.VersionInterfaces, error) {
	switch version {
	case v1alpha1.SchemeGroupVersion.String():
		return &meta.VersionInterfaces{
			Codec:            v1alpha1.Codec,
			ObjectConvertor:  api.Scheme,
			MetadataAccessor: Accessor,
		}, nil

	default:
		return nil, fmt.Errorf("unsupported storage version: %s (valid: %s)", version, ExternalVersions)
	}
}

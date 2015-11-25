/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

// Package to keep track of API Versions that should be registered in api.Scheme.
package registered

import (
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

// List of registered API versions.
// The list is in the order of most preferred to the least.
var RegisteredGroupVersions []unversioned.GroupVersion

func init() {
	validGroupVersions := map[unversioned.GroupVersion]bool{
		unversioned.GroupVersion{Group: "", Version: "v1"}:                      true,
		unversioned.GroupVersion{Group: "extensions", Version: "v1beta1"}:       true,
		unversioned.GroupVersion{Group: "componentconfig", Version: "v1alpha1"}: true,
		unversioned.GroupVersion{Group: "metrics", Version: "v1alpha1"}:         true,
	}

	// The default list of supported api versions, in order of most preferred to the least.
	supportedVersions := []unversioned.GroupVersion{
		{Group: "", Version: "v1"},
		{Group: "extensions", Version: "v1beta1"},
		{Group: "componentconfig", Version: "v1alpha1"},
	}

	// Env var KUBE_API_VERSIONS is a comma separated list of API versions that should be registered in the scheme.
	// The versions should be in the order of most preferred to the least.
	userRequestedVersions := os.Getenv("KUBE_API_VERSIONS")
	if len(userRequestedVersions) != 0 {
		// reset the supported versions
		supportedVersions = []unversioned.GroupVersion{}
		for _, version := range strings.Split(userRequestedVersions, ",") {
			gv, err := unversioned.ParseGroupVersion(version)
			if err != nil {
				glog.Fatalf("invalid api version: %s in KUBE_API_VERSIONS: %s. List of valid API versions: %v",
					version, os.Getenv("KUBE_API_VERSIONS"), validGroupVersions)
			}

			// Verify that the version is valid.
			valid, ok := validGroupVersions[gv]
			if !ok || !valid {
				// Not a valid API version.
				glog.Fatalf("invalid api version: %s in KUBE_API_VERSIONS: %s. List of valid API versions: %v",
					version, os.Getenv("KUBE_API_VERSIONS"), validGroupVersions)
			}

			supportedVersions = append(supportedVersions, gv)
		}
	}

	RegisteredGroupVersions = supportedVersions
}

// Returns true if the given api version is one of the registered api versions.
func IsRegisteredAPIGroupVersion(gv unversioned.GroupVersion) bool {
	for _, currGV := range RegisteredGroupVersions {
		if currGV == gv {
			return true
		}
	}
	return false
}

// GroupVersionsForGroup returns the registered versions of a group in the form
// of "group/version".
func GroupVersionsForGroup(group string) []unversioned.GroupVersion {
	ret := []unversioned.GroupVersion{}
	for _, v := range RegisteredGroupVersions {
		if v.Group == group {
			ret = append(ret, v)
		}
	}
	return ret
}

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

// Package to keep track of API GroupVersions that should be registered in api.Scheme.
package registered

import (
	"os"
	"strings"

	"github.com/golang/glog"
	apiutil "k8s.io/kubernetes/pkg/api/util"
)

// List of registered API versions.
// The list is in the order of most preferred to the least.
var RegisteredGroupVersions []string

func init() {
	validAPIGroupVersions := map[string]bool{
		"v1": true,
		"experimental/v1alpha1": true,
	}

	// The default list of supported api versions, in order of most preferred to the least.
	defaultSupportedGroupVersions := "v1"
	// Env var KUBE_API_VERSIONS is a comma separated list of API versions that should be registered in the scheme.
	// The versions should be in the order of most preferred to the least.
	supportedGroupVersions := os.Getenv("KUBE_API_VERSIONS")
	if supportedGroupVersions == "" {
		supportedGroupVersions = defaultSupportedGroupVersions
	}
	versions := strings.Split(supportedGroupVersions, ",")
	for _, version := range versions {
		// Verify that the version is valid.
		valid, ok := validAPIGroupVersions[version]
		if !ok || !valid {
			// Not a valid API version.
			glog.Fatalf("invalid api version: %s in KUBE_API_VERSIONS: %s. List of valid API versions: %v",
				version, os.Getenv("KUBE_API_VERSIONS"), validAPIGroupVersions)
		}
		RegisteredGroupVersions = append(RegisteredGroupVersions, version)
	}
}

// Returns true if the given groupVersion is one of the registered groupVersions.
func IsRegisteredAPIVersion(groupVersion string) bool {
	for _, apiGroupVersion := range RegisteredGroupVersions {
		if apiGroupVersion == groupVersion {
			return true
		}
	}
	return false
}

func GroupVersionsForGroup(group string) []string {
	ret := []string{}
	for _, v := range RegisteredGroupVersions {
		if apiutil.GetGroup(v) == group {
			ret = append(ret, v)
		}
	}
	return ret
}

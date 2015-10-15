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
	apiutil "k8s.io/kubernetes/pkg/api/util"
)

// List of registered API versions.
// The list is in the order of most preferred to the least.
var RegisteredVersions []string

func init() {
	// TODO: caesarxuchao: rename this variable to validGroupVersions
	validAPIVersions := map[string]bool{
		"v1":                 true,
		"extensions/v1beta1": true,
	}

	// The default list of supported api versions, in order of most preferred to the least.
	defaultSupportedVersions := "v1,extensions/v1beta1"
	// Env var KUBE_API_VERSIONS is a comma separated list of API versions that should be registered in the scheme.
	// The versions should be in the order of most preferred to the least.
	supportedVersions := os.Getenv("KUBE_API_VERSIONS")
	if supportedVersions == "" {
		supportedVersions = defaultSupportedVersions
	}
	versions := strings.Split(supportedVersions, ",")
	for _, version := range versions {
		// Verify that the version is valid.
		valid, ok := validAPIVersions[version]
		if !ok || !valid {
			// Not a valid API version.
			glog.Fatalf("invalid api version: %s in KUBE_API_VERSIONS: %s. List of valid API versions: %v",
				version, os.Getenv("KUBE_API_VERSIONS"), validAPIVersions)
		}
		RegisteredVersions = append(RegisteredVersions, version)
	}
}

// Returns true if the given api version is one of the registered api versions.
func IsRegisteredAPIVersion(version string) bool {
	for _, apiVersion := range RegisteredVersions {
		if apiVersion == version {
			return true
		}
	}
	return false
}

// GroupVersionsForGroup returns the registered versions of a group in the form
// of "group/version".
func GroupVersionsForGroup(group string) []string {
	ret := []string{}
	for _, v := range RegisteredVersions {
		if apiutil.GetGroup(v) == group {
			ret = append(ret, v)
		}
	}
	return ret
}

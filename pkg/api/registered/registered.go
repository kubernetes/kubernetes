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
	"fmt"
	"os"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
)

var (
	// registeredVersions represents all registered API versions. Please call
	// RegisterVersions() to add registered versions.
	registeredVersions = map[unversioned.GroupVersion]struct{}{}

	// enabledVersions represents all enabled API versions. It should be a
	// subset of registeredVersions. Please call EnableVersions() to add
	// enabled versions.
	enabledVersions = map[unversioned.GroupVersion]struct{}{}

	// envRequestedVersions represents the versions requested via the
	// KUBE_API_VERSIONS environment variable. The install package of each group
	// checks this list before add their versions to the latest package and
	// Scheme.
	envRequestedVersions = map[unversioned.GroupVersion]struct{}{}
)

func init() {
	// Env var KUBE_API_VERSIONS is a comma separated list of API versions that
	// should be registered in the scheme.
	kubeAPIVersions := os.Getenv("KUBE_API_VERSIONS")
	if len(kubeAPIVersions) != 0 {
		for _, version := range strings.Split(kubeAPIVersions, ",") {
			gv, err := unversioned.ParseGroupVersion(version)
			if err != nil {
				glog.Fatalf("invalid api version: %s in KUBE_API_VERSIONS: %s.",
					version, os.Getenv("KUBE_API_VERSIONS"))
			}
			envRequestedVersions[gv] = struct{}{}
		}
	}
}

// RegisterVersions add the versions the registeredVersions.
func RegisterVersions(versions ...unversioned.GroupVersion) {
	for _, v := range versions {
		registeredVersions[v] = struct{}{}
	}
}

// EnableVersions add the versions to the enabledVersions. The caller of this
// function is responsible to add the version to 'latest' and 'Scheme'.
func EnableVersions(versions ...unversioned.GroupVersion) error {
	var unregisteredVersions []unversioned.GroupVersion
	for _, v := range versions {
		if _, found := registeredVersions[v]; !found {
			unregisteredVersions = append(unregisteredVersions, v)
		}
		enabledVersions[v] = struct{}{}
	}
	if len(unregisteredVersions) != 0 {
		return fmt.Errorf("Please register versions before enabling them: %v", unregisteredVersions)
	}
	return nil
}

// IsAllowedVersion returns if the version is allowed by the KUBE_API_VERSIONS
// environment variable. If the environment variable is empty, then it always
// returns true.
func IsAllowedVersion(v unversioned.GroupVersion) bool {
	if len(envRequestedVersions) == 0 {
		return true
	}
	_, found := envRequestedVersions[v]
	return found
}

// IsEnabledVersion returns if a version is enabled.
func IsEnabledVersion(v unversioned.GroupVersion) bool {
	_, found := enabledVersions[v]
	return found
}

// IsRegisteredVersion returns if a version is registered.
func IsRegisteredVersion(v unversioned.GroupVersion) bool {
	_, found := registeredVersions[v]
	return found
}

// EnabledVersions returns all enabled versions.
func EnabledVersions() (ret []unversioned.GroupVersion) {
	for v := range enabledVersions {
		ret = append(ret, v)
	}
	return
}

// RegisteredVersions returns all registered versions.
func RegisteredVersions() (ret []unversioned.GroupVersion) {
	for v := range registeredVersions {
		ret = append(ret, v)
	}
	return
}

// EnabledVersionsForGroup returns all enabled versions for a group.
func EnabledVersionsForGroup(group string) (ret []unversioned.GroupVersion) {
	for v := range enabledVersions {
		if v.Group == group {
			ret = append(ret, v)
		}
	}
	return
}

// RegisteredVersionsForGroup returns all registered versions for a group.
func RegisteredVersionsForGroup(group string) (ret []unversioned.GroupVersion) {
	for v := range registeredVersions {
		if v.Group == group {
			ret = append(ret, v)
		}
	}
	return
}

// ValidateEnvRequestedVersions returns a list of versions that are requested in
// the KUBE_API_VERSIONS environment variable, but not enabled.
func ValidateEnvRequestedVersions() []unversioned.GroupVersion {
	var missingVersions []unversioned.GroupVersion
	for v := range envRequestedVersions {
		if _, found := enabledVersions[v]; !found {
			missingVersions = append(missingVersions, v)
		}
	}
	return missingVersions
}

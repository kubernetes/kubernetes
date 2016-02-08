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

// Package to keep track of API Versions that can be registered and are enabled in api.Scheme.
package registered

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
)

var (
	// registeredGroupVersions stores all API group versions for which RegisterGroup is called.
	registeredVersions = map[unversioned.GroupVersion]struct{}{}

	// enabledVersions represents all enabled API versions. It should be a
	// subset of registeredVersions. Please call EnableVersions() to add
	// enabled versions.
	enabledVersions = map[unversioned.GroupVersion]struct{}{}

	// map of group meta for all groups.
	groupMetaMap = map[string]*apimachinery.GroupMeta{}

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

// RegisterVersions adds the given group versions to the list of registered group versions.
func RegisterVersions(availableVersions []unversioned.GroupVersion) {
	for _, v := range availableVersions {
		registeredVersions[v] = struct{}{}
	}
}

// RegisterGroup adds the given group to the list of registered groups.
func RegisterGroup(groupMeta apimachinery.GroupMeta) error {
	groupName := groupMeta.GroupVersion.Group
	if _, found := groupMetaMap[groupName]; found {
		return fmt.Errorf("group %v is already registered", groupMetaMap)
	}
	groupMetaMap[groupName] = &groupMeta
	return nil
}

// EnableVersions adds the versions for the given group to the list of enabled versions.
// Note that the caller should call RegisterGroup before calling this method.
// The caller of this function is responsible to add the versions to scheme and RESTMapper.
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

// EnabledVersions returns all enabled versions.  Groups are randomly ordered, but versions within groups
// are priority order from best to worst
func EnabledVersions() []unversioned.GroupVersion {
	ret := []unversioned.GroupVersion{}
	for _, groupMeta := range groupMetaMap {
		ret = append(ret, groupMeta.GroupVersions...)
	}
	return ret
}

// EnabledVersionsForGroup returns all enabled versions for a group in order of best to worst
func EnabledVersionsForGroup(group string) []unversioned.GroupVersion {
	groupMeta, ok := groupMetaMap[group]
	if !ok {
		return []unversioned.GroupVersion{}
	}

	return append([]unversioned.GroupVersion{}, groupMeta.GroupVersions...)
}

// Group returns the metadata of a group if the gruop is registered, otherwise
// an erorr is returned.
func Group(group string) (*apimachinery.GroupMeta, error) {
	groupMeta, found := groupMetaMap[group]
	if !found {
		return nil, fmt.Errorf("group %v has not been registered", group)
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy, nil
}

// IsRegistered takes a string and determines if it's one of the registered groups
func IsRegistered(group string) bool {
	_, found := groupMetaMap[group]
	return found
}

// TODO: This is an expedient function, because we don't check if a Group is
// supported throughout the code base. We will abandon this function and
// checking the error returned by the Group() function.
func GroupOrDie(group string) *apimachinery.GroupMeta {
	groupMeta, found := groupMetaMap[group]
	if !found {
		if group == "" {
			panic("The legacy v1 API is not registered.")
		} else {
			panic(fmt.Sprintf("Group %s is not registered.", group))
		}
	}
	groupMetaCopy := *groupMeta
	return &groupMetaCopy
}

// AllPreferredGroupVersions returns the preferred versions of all registered
// groups in the form of "group1/version1,group2/version2,..."
func AllPreferredGroupVersions() string {
	if len(groupMetaMap) == 0 {
		return ""
	}
	var defaults []string
	for _, groupMeta := range groupMetaMap {
		defaults = append(defaults, groupMeta.GroupVersion.String())
	}
	sort.Strings(defaults)
	return strings.Join(defaults, ",")
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

// Resets the state.
// Should not be used by anyone else than tests.
func reset() {
	registeredVersions = map[unversioned.GroupVersion]struct{}{}
	enabledVersions = map[unversioned.GroupVersion]struct{}{}
	groupMetaMap = map[string]*apimachinery.GroupMeta{}

}

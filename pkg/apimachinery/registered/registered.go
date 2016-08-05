/*
Copyright 2015 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apimachinery"
	"k8s.io/kubernetes/pkg/util/sets"
)

var (
	// registeredGroupVersions stores all API group versions for which RegisterGroup is called.
	registeredVersions = map[unversioned.GroupVersion]struct{}{}

	// thirdPartyGroupVersions are API versions which are dynamically
	// registered (and unregistered) via API calls to the apiserver
	thirdPartyGroupVersions []unversioned.GroupVersion

	// enabledVersions represents all enabled API versions. It should be a
	// subset of registeredVersions. Please call EnableVersions() to add
	// enabled versions.
	enabledVersions = map[unversioned.GroupVersion]struct{}{}

	// map of group meta for all groups.
	groupMetaMap = map[string]*apimachinery.GroupMeta{}

	// envRequestedVersions represents the versions requested via the
	// KUBE_API_VERSIONS environment variable. The install package of each group
	// checks this list before add their versions to the latest package and
	// Scheme.  This list is small and order matters, so represent as a slice
	envRequestedVersions = []unversioned.GroupVersion{}
)

func init() {
	loadKubeAPIVersions()
}

func loadKubeAPIVersions() {
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
			envRequestedVersions = append(envRequestedVersions, gv)
		}
	}
}

// Resets everything to clean room for the start of a test
func clearForTesting() {
	registeredVersions = map[unversioned.GroupVersion]struct{}{}
	thirdPartyGroupVersions = []unversioned.GroupVersion{}
	enabledVersions = map[unversioned.GroupVersion]struct{}{}
	groupMetaMap = map[string]*apimachinery.GroupMeta{}
	envRequestedVersions = []unversioned.GroupVersion{}
	loadKubeAPIVersions()
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
	for _, envGV := range envRequestedVersions {
		if v == envGV {
			return true
		}
	}
	return false
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

// IsRegisteredVersion returns if a version is registered.
func IsRegisteredVersion(v unversioned.GroupVersion) bool {
	_, found := registeredVersions[v]
	return found
}

// RegisteredGroupVersions returns all registered group versions.
func RegisteredGroupVersions() []unversioned.GroupVersion {
	ret := []unversioned.GroupVersion{}
	for groupVersion := range registeredVersions {
		ret = append(ret, groupVersion)
	}
	return ret
}

// IsThirdPartyAPIGroupVersion returns true if the api version is a user-registered group/version.
func IsThirdPartyAPIGroupVersion(gv unversioned.GroupVersion) bool {
	for ix := range thirdPartyGroupVersions {
		if thirdPartyGroupVersions[ix] == gv {
			return true
		}
	}
	return false
}

// AddThirdPartyAPIGroupVersions sets the list of third party versions,
// registers them in the API machinery and enables them.
// Skips GroupVersions that are already registered.
// Returns the list of GroupVersions that were skipped.
func AddThirdPartyAPIGroupVersions(gvs ...unversioned.GroupVersion) []unversioned.GroupVersion {
	filteredGVs := []unversioned.GroupVersion{}
	skippedGVs := []unversioned.GroupVersion{}
	for ix := range gvs {
		if !IsRegisteredVersion(gvs[ix]) {
			filteredGVs = append(filteredGVs, gvs[ix])
		} else {
			glog.V(3).Infof("Skipping %s, because its already registered", gvs[ix].String())
			skippedGVs = append(skippedGVs, gvs[ix])
		}
	}
	if len(filteredGVs) == 0 {
		return skippedGVs
	}
	RegisterVersions(filteredGVs)
	EnableVersions(filteredGVs...)
	thirdPartyGroupVersions = append(thirdPartyGroupVersions, filteredGVs...)
	return skippedGVs
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

// RESTMapper returns a union RESTMapper of all known types with priorities chosen in the following order:
//  1. if KUBE_API_VERSIONS is specified, then KUBE_API_VERSIONS in order, OR
//  1. legacy kube group preferred version, extensions preferred version, metrics perferred version, legacy
//     kube any version, extensions any version, metrics any version, all other groups alphabetical preferred version,
//     all other groups alphabetical.
func RESTMapper(versionPatterns ...unversioned.GroupVersion) meta.RESTMapper {
	unionMapper := meta.MultiRESTMapper{}
	unionedGroups := sets.NewString()
	for enabledVersion := range enabledVersions {
		if !unionedGroups.Has(enabledVersion.Group) {
			unionedGroups.Insert(enabledVersion.Group)
			groupMeta := groupMetaMap[enabledVersion.Group]
			unionMapper = append(unionMapper, groupMeta.RESTMapper)
		}
	}

	if len(versionPatterns) != 0 {
		resourcePriority := []unversioned.GroupVersionResource{}
		kindPriority := []unversioned.GroupVersionKind{}
		for _, versionPriority := range versionPatterns {
			resourcePriority = append(resourcePriority, versionPriority.WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, versionPriority.WithKind(meta.AnyKind))
		}

		return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
	}

	if len(envRequestedVersions) != 0 {
		resourcePriority := []unversioned.GroupVersionResource{}
		kindPriority := []unversioned.GroupVersionKind{}

		for _, versionPriority := range envRequestedVersions {
			resourcePriority = append(resourcePriority, versionPriority.WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, versionPriority.WithKind(meta.AnyKind))
		}

		return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
	}

	prioritizedGroups := []string{"", "extensions", "metrics"}
	resourcePriority, kindPriority := prioritiesForGroups(prioritizedGroups...)

	prioritizedGroupsSet := sets.NewString(prioritizedGroups...)
	remainingGroups := sets.String{}
	for enabledVersion := range enabledVersions {
		if !prioritizedGroupsSet.Has(enabledVersion.Group) {
			remainingGroups.Insert(enabledVersion.Group)
		}
	}

	remainingResourcePriority, remainingKindPriority := prioritiesForGroups(remainingGroups.List()...)
	resourcePriority = append(resourcePriority, remainingResourcePriority...)
	kindPriority = append(kindPriority, remainingKindPriority...)

	return meta.PriorityRESTMapper{Delegate: unionMapper, ResourcePriority: resourcePriority, KindPriority: kindPriority}
}

// prioritiesForGroups returns the resource and kind priorities for a PriorityRESTMapper, preferring the preferred version of each group first,
// then any non-preferred version of the group second.
func prioritiesForGroups(groups ...string) ([]unversioned.GroupVersionResource, []unversioned.GroupVersionKind) {
	resourcePriority := []unversioned.GroupVersionResource{}
	kindPriority := []unversioned.GroupVersionKind{}

	for _, group := range groups {
		availableVersions := EnabledVersionsForGroup(group)
		if len(availableVersions) > 0 {
			resourcePriority = append(resourcePriority, availableVersions[0].WithResource(meta.AnyResource))
			kindPriority = append(kindPriority, availableVersions[0].WithKind(meta.AnyKind))
		}
	}
	for _, group := range groups {
		resourcePriority = append(resourcePriority, unversioned.GroupVersionResource{Group: group, Version: meta.AnyVersion, Resource: meta.AnyResource})
		kindPriority = append(kindPriority, unversioned.GroupVersionKind{Group: group, Version: meta.AnyVersion, Kind: meta.AnyKind})
	}

	return resourcePriority, kindPriority
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
	for _, v := range envRequestedVersions {
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

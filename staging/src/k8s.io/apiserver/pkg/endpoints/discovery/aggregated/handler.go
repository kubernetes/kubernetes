/*
Copyright 2022 The Kubernetes Authors.

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

package aggregated

import (
	"net/http"
	"reflect"
	"sort"
	"sync"

	apidiscoveryv2beta1 "k8s.io/api/apidiscovery/v2beta1"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"

	"sync/atomic"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
)

// This handler serves the /apis endpoint for an aggregated list of
// api resources indexed by their group version.
type ResourceManager interface {
	// Adds knowledge of the given groupversion to the discovery document
	// If it was already being tracked, updates the stored APIVersionDiscovery
	// Thread-safe
	AddGroupVersion(groupName string, value apidiscoveryv2beta1.APIVersionDiscovery)

	// Sets a priority to be used while sorting a specific group and
	// group-version. If two versions report different priorities for
	// the group, the higher one will be used. If the group is not
	// known, the priority is ignored. The priority for this version
	// is forgotten once the group-version is forgotten
	SetGroupVersionPriority(gv metav1.GroupVersion, grouppriority, versionpriority int)

	// Removes all group versions for a given group
	// Thread-safe
	RemoveGroup(groupName string)

	// Removes a specific groupversion. If all versions of a group have been
	// removed, then the entire group is unlisted.
	// Thread-safe
	RemoveGroupVersion(gv metav1.GroupVersion)

	// Resets the manager's known list of group-versions and replaces them
	// with the given groups
	// Thread-Safe
	SetGroups([]apidiscoveryv2beta1.APIGroupDiscovery)

	http.Handler
}

type resourceDiscoveryManager struct {
	serializer runtime.NegotiatedSerializer
	// cache is an atomic pointer to avoid the use of locks
	cache atomic.Pointer[cachedGroupList]

	// Writes protected by the lock.
	// List of all apigroups & resources indexed by the resource manager
	lock              sync.RWMutex
	apiGroups         map[string]*apidiscoveryv2beta1.APIGroupDiscovery
	versionPriorities map[metav1.GroupVersion]priorityInfo
}

type priorityInfo struct {
	GroupPriorityMinimum int
	VersionPriority      int
}

func NewResourceManager() ResourceManager {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	utilruntime.Must(apidiscoveryv2beta1.AddToScheme(scheme))
	return &resourceDiscoveryManager{serializer: codecs, versionPriorities: make(map[metav1.GroupVersion]priorityInfo)}
}

func (rdm *resourceDiscoveryManager) SetGroupVersionPriority(gv metav1.GroupVersion, groupPriorityMinimum, versionPriority int) {
	rdm.lock.Lock()
	defer rdm.lock.Unlock()

	rdm.versionPriorities[gv] = priorityInfo{
		GroupPriorityMinimum: groupPriorityMinimum,
		VersionPriority:      versionPriority,
	}
	rdm.cache.Store(nil)
}

func (rdm *resourceDiscoveryManager) SetGroups(groups []apidiscoveryv2beta1.APIGroupDiscovery) {
	rdm.lock.Lock()
	defer rdm.lock.Unlock()

	rdm.apiGroups = nil
	rdm.cache.Store(nil)

	for _, group := range groups {
		for _, version := range group.Versions {
			rdm.addGroupVersionLocked(group.Name, version)
		}
	}

	// Filter unused out priority entries
	for gv := range rdm.versionPriorities {
		entry, exists := rdm.apiGroups[gv.Group]
		if !exists {
			delete(rdm.versionPriorities, gv)
			continue
		}

		containsVersion := false

		for _, v := range entry.Versions {
			if v.Version == gv.Version {
				containsVersion = true
				break
			}
		}

		if !containsVersion {
			delete(rdm.versionPriorities, gv)
		}
	}
}

func (rdm *resourceDiscoveryManager) AddGroupVersion(groupName string, value apidiscoveryv2beta1.APIVersionDiscovery) {
	rdm.lock.Lock()
	defer rdm.lock.Unlock()

	rdm.addGroupVersionLocked(groupName, value)
}

func (rdm *resourceDiscoveryManager) addGroupVersionLocked(groupName string, value apidiscoveryv2beta1.APIVersionDiscovery) {
	klog.Infof("Adding GroupVersion %s %s to ResourceManager", groupName, value.Version)

	if rdm.apiGroups == nil {
		rdm.apiGroups = make(map[string]*apidiscoveryv2beta1.APIGroupDiscovery)
	}

	if existing, groupExists := rdm.apiGroups[groupName]; groupExists {
		// If this version already exists, replace it
		versionExists := false

		// Not very efficient, but in practice there are generally not many versions
		for i := range existing.Versions {
			if existing.Versions[i].Version == value.Version {
				// The new gv is the exact same as what is already in
				// the map. This is a noop and cache should not be
				// invalidated.
				if reflect.DeepEqual(existing.Versions[i], value) {
					return
				}
				existing.Versions[i] = value
				versionExists = true
				break
			}
		}

		if !versionExists {
			existing.Versions = append(existing.Versions, value)
		}

	} else {
		group := &apidiscoveryv2beta1.APIGroupDiscovery{
			ObjectMeta: metav1.ObjectMeta{
				Name: groupName,
			},
			Versions: []apidiscoveryv2beta1.APIVersionDiscovery{value},
		}
		rdm.apiGroups[groupName] = group
	}

	gv := metav1.GroupVersion{Group: groupName, Version: value.Version}
	if _, ok := rdm.versionPriorities[gv]; !ok {
		rdm.versionPriorities[gv] = priorityInfo{
			GroupPriorityMinimum: 1000,
			VersionPriority:      15,
		}
	}

	// Reset response document so it is recreated lazily
	rdm.cache.Store(nil)
}

func (rdm *resourceDiscoveryManager) RemoveGroupVersion(apiGroup metav1.GroupVersion) {
	rdm.lock.Lock()
	defer rdm.lock.Unlock()
	group, exists := rdm.apiGroups[apiGroup.Group]
	if !exists {
		return
	}

	modified := false
	for i := range group.Versions {
		if group.Versions[i].Version == apiGroup.Version {
			group.Versions = append(group.Versions[:i], group.Versions[i+1:]...)
			modified = true
			break
		}
	}
	// If no modification was done, cache does not need to be cleared
	if !modified {
		return
	}

	delete(rdm.versionPriorities, apiGroup)
	if len(group.Versions) == 0 {
		delete(rdm.apiGroups, group.Name)
	}

	// Reset response document so it is recreated lazily
	rdm.cache.Store(nil)
}

func (rdm *resourceDiscoveryManager) RemoveGroup(groupName string) {
	rdm.lock.Lock()
	defer rdm.lock.Unlock()

	delete(rdm.apiGroups, groupName)

	for k := range rdm.versionPriorities {
		if k.Group == groupName {
			delete(rdm.versionPriorities, k)
		}
	}

	// Reset response document so it is recreated lazily
	rdm.cache.Store(nil)
}

// Prepares the api group list for serving by converting them from map into
// list and sorting them according to insertion order
func (rdm *resourceDiscoveryManager) calculateAPIGroupsLocked() []apidiscoveryv2beta1.APIGroupDiscovery {
	// Re-order the apiGroups by their priority.
	groups := []apidiscoveryv2beta1.APIGroupDiscovery{}
	for _, group := range rdm.apiGroups {
		copied := *group.DeepCopy()

		// Re-order versions based on their priority. Use kube-aware string
		// comparison as a tie breaker
		sort.SliceStable(copied.Versions, func(i, j int) bool {
			iVersion := copied.Versions[i].Version
			jVersion := copied.Versions[j].Version

			iPriority := rdm.versionPriorities[metav1.GroupVersion{Group: group.Name, Version: iVersion}].VersionPriority
			jPriority := rdm.versionPriorities[metav1.GroupVersion{Group: group.Name, Version: jVersion}].VersionPriority

			// Sort by version string comparator if priority is equal
			if iPriority == jPriority {
				return version.CompareKubeAwareVersionStrings(iVersion, jVersion) > 0
			}

			// i sorts before j if it has a higher priority
			return iPriority > jPriority
		})

		groups = append(groups, *copied.DeepCopy())

	}

	// For each group, determine the highest minimum group priority and use that
	priorities := map[string]int{}
	for gv, info := range rdm.versionPriorities {
		if existing, exists := priorities[gv.Group]; exists {
			if existing < info.GroupPriorityMinimum {
				priorities[gv.Group] = info.GroupPriorityMinimum
			}
		} else {
			priorities[gv.Group] = info.GroupPriorityMinimum
		}
	}

	sort.SliceStable(groups, func(i, j int) bool {
		iName := groups[i].Name
		jName := groups[j].Name

		// Default to 0 priority by default
		iPriority := priorities[iName]
		jPriority := priorities[jName]

		// Sort discovery based on apiservice priority.
		// Duplicated from staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helpers.go
		if iPriority == jPriority {
			// Equal priority uses name to break ties
			return iName < jName
		}

		// i sorts before j if it has a higher priority
		return iPriority > jPriority
	})

	return groups
}

// Fetches from cache if it exists. If cache is empty, create it.
func (rdm *resourceDiscoveryManager) fetchFromCache() *cachedGroupList {
	rdm.lock.RLock()
	defer rdm.lock.RUnlock()

	cacheLoad := rdm.cache.Load()
	if cacheLoad != nil {
		return cacheLoad
	}
	response := apidiscoveryv2beta1.APIGroupDiscoveryList{
		Items: rdm.calculateAPIGroupsLocked(),
	}
	etag, err := calculateETag(response)
	if err != nil {
		klog.Errorf("failed to calculate etag for discovery document: %s", etag)
		etag = ""
	}
	cached := &cachedGroupList{
		cachedResponse:     response,
		cachedResponseETag: etag,
	}
	rdm.cache.Store(cached)
	return cached
}

type cachedGroupList struct {
	cachedResponse     apidiscoveryv2beta1.APIGroupDiscoveryList
	cachedResponseETag string
}

func (rdm *resourceDiscoveryManager) ServeHTTP(resp http.ResponseWriter, req *http.Request) {
	cache := rdm.fetchFromCache()
	response := cache.cachedResponse
	etag := cache.cachedResponseETag

	if len(etag) > 0 {
		// Use proper e-tag headers if one is available
		ServeHTTPWithETag(
			&response,
			etag,
			rdm.serializer,
			resp,
			req,
		)
	} else {
		// Default to normal response in rare case etag is
		// not cached with the object for some reason.
		responsewriters.WriteObjectNegotiated(
			rdm.serializer,
			DiscoveryEndpointRestrictions,
			AggregatedDiscoveryGV,
			resp,
			req,
			http.StatusOK,
			&response,
			true,
		)
	}
}

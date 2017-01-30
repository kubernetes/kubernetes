/*
Copyright 2016 The Kubernetes Authors.

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

package discovery

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/version"
	// Import solely to initialize client auth plugins.
	_ "k8s.io/kubernetes/plugin/pkg/client/auth"
)

// MatchesServerVersion queries the server to compares the build version
// (git hash) of the client with the server's build version. It returns an error
// if it failed to contact the server or if the versions are not an exact match.
func MatchesServerVersion(client DiscoveryInterface) error {
	cVer := version.Get()
	sVer, err := client.ServerVersion()
	if err != nil {
		return fmt.Errorf("couldn't read version from server: %v\n", err)
	}
	// GitVersion includes GitCommit and GitTreeState, but best to be safe?
	if cVer.GitVersion != sVer.GitVersion || cVer.GitCommit != sVer.GitCommit || cVer.GitTreeState != sVer.GitTreeState {
		return fmt.Errorf("server version (%#v) differs from client version (%#v)!\n", sVer, cVer)
	}

	return nil
}

// NegotiateVersion queries the server's supported api versions to find
// a version that both client and server support.
// - If no version is provided, try registered client versions in order of
//   preference.
// - If version is provided and the server does not support it,
//   return an error.
func NegotiateVersion(client DiscoveryInterface, requiredGV *schema.GroupVersion, clientRegisteredGVs []schema.GroupVersion) (*schema.GroupVersion, error) {
	clientVersions := sets.String{}
	for _, gv := range clientRegisteredGVs {
		clientVersions.Insert(gv.String())
	}
	groups, err := client.ServerGroups()
	if err != nil {
		// This is almost always a connection error, and higher level code should treat this as a generic error,
		// not a negotiation specific error.
		return nil, err
	}
	versions := metav1.ExtractGroupVersions(groups)
	serverVersions := sets.String{}
	for _, v := range versions {
		serverVersions.Insert(v)
	}

	// If version explicitly requested verify that both client and server support it.
	// If server does not support warn, but try to negotiate a lower version.
	if requiredGV != nil {
		if !clientVersions.Has(requiredGV.String()) {
			return nil, fmt.Errorf("client does not support API version %q; client supported API versions: %v", requiredGV, clientVersions)

		}
		// If the server supports no versions, then we should just use the preferredGV
		// This can happen because discovery fails due to 403 Forbidden errors
		if len(serverVersions) == 0 {
			return requiredGV, nil
		}
		if serverVersions.Has(requiredGV.String()) {
			return requiredGV, nil
		}
		// If we are using an explicit config version the server does not support, fail.
		return nil, fmt.Errorf("server does not support API version %q", requiredGV)
	}

	for _, clientGV := range clientRegisteredGVs {
		if serverVersions.Has(clientGV.String()) {
			// Version was not explicitly requested in command config (--api-version).
			// Ok to fall back to a supported version with a warning.
			// TODO: caesarxuchao: enable the warning message when we have
			// proper fix. Please refer to issue #14895.
			// if len(version) != 0 {
			// 	glog.Warningf("Server does not support API version '%s'. Falling back to '%s'.", version, clientVersion)
			// }
			t := clientGV
			return &t, nil
		}
	}

	// if we have no server versions and we have no required version, choose the first clientRegisteredVersion
	if len(serverVersions) == 0 && len(clientRegisteredGVs) > 0 {
		return &clientRegisteredGVs[0], nil
	}

	return nil, fmt.Errorf("failed to negotiate an api version; server supports: %v, client supports: %v",
		serverVersions, clientVersions)
}

// GroupVersionResources converts APIResourceLists to the GroupVersionResources.
func GroupVersionResources(rls []*metav1.APIResourceList) (map[schema.GroupVersionResource]struct{}, error) {
	gvrs := map[schema.GroupVersionResource]struct{}{}
	for _, rl := range rls {
		gv, err := schema.ParseGroupVersion(rl.GroupVersion)
		if err != nil {
			return nil, err
		}
		for i := range rl.APIResources {
			gvrs[schema.GroupVersionResource{Group: gv.Group, Version: gv.Version, Resource: rl.APIResources[i].Name}] = struct{}{}
		}
	}
	return gvrs, nil
}

// FilteredBy filters by the given predicate. Empty APIResourceLists are dropped.
func FilteredBy(pred ResourcePredicate, rls []*metav1.APIResourceList) []*metav1.APIResourceList {
	result := []*metav1.APIResourceList{}
	for _, rl := range rls {
		filtered := *rl
		filtered.APIResources = nil
		for i := range rl.APIResources {
			if pred.Match(rl.GroupVersion, &rl.APIResources[i]) {
				filtered.APIResources = append(filtered.APIResources, rl.APIResources[i])
			}
		}
		if filtered.APIResources != nil {
			result = append(result, &filtered)
		}
	}
	return result
}

type ResourcePredicate interface {
	Match(groupVersion string, r *metav1.APIResource) bool
}

type ResourcePredicateFunc func(groupVersion string, r *metav1.APIResource) bool

func (fn ResourcePredicateFunc) Match(groupVersion string, r *metav1.APIResource) bool {
	return fn(groupVersion, r)
}

// SupportsAllVerbs is a predicate matching a resource iff all given verbs are supported.
type SupportsAllVerbs struct {
	Verbs []string
}

func (p SupportsAllVerbs) Match(groupVersion string, r *metav1.APIResource) bool {
	return sets.NewString([]string(r.Verbs)...).HasAll(p.Verbs...)
}

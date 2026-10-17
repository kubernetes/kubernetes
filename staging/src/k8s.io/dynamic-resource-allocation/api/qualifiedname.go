/*
Copyright The Kubernetes Authors.

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

package api

import (
	"strings"

	resourceapi "k8s.io/api/resource/v1"
)

// LookupByQualifiedName looks up key in m. The default domain
// qualifies keys which do not already have an explicit domain.
//
// The map may contain entries with or without a domain or (worse!) both for the same identifier.
// The entry with a fully-qualified name is preferred in case of such an ambiguity.
func LookupByQualifiedName[T any](m map[resourceapi.QualifiedName]T, key resourceapi.QualifiedName, defaultDomain string) (T, bool) {
	domain, name, hasDomain := strings.Cut(string(key), "/")
	if hasDomain {
		// Check with domain first.
		if v, ok := m[key]; ok {
			return v, true
		}
		// Then without, but only if the explicit domain matches.
		// Entries in the map without a domain are no match
		// because they would be qualified with the defaultDomain.
		if domain == defaultDomain {
			if v, ok := m[resourceapi.QualifiedName(name)]; ok {
				return v, true
			}
		}
	} else {
		// Check with domain first.
		if v, ok := m[resourceapi.QualifiedName(defaultDomain+"/"+string(key))]; ok {
			return v, true
		}
		// Then without.
		if v, ok := m[key]; ok {
			return v, true
		}
	}

	// Not found.
	var zero T
	return zero, false
}

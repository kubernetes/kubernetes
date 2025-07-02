// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/http"
	"slices"
	"strings"
)

// VaryMatcher defines the interface implemented by types that can match
// request headers nominated by the a cached response's Vary header against
// the headers of an incoming request.
type VaryMatcher interface {
	VaryHeadersMatch(cachedHdrs ResponseRefs, reqHdr http.Header) (int, bool)
}

func NewVaryMatcher(hvn HeaderValueNormalizer) *varyMatcher {
	return &varyMatcher{hvn: hvn}
}

type varyMatcher struct {
	hvn HeaderValueNormalizer
}

func (vm *varyMatcher) VaryHeadersMatch(entries ResponseRefs, reqHdr http.Header) (int, bool) {
	slices.SortFunc(entries, func(a, b *ResponseRef) int {
		aVary := strings.TrimSpace(a.Vary)
		bVary := strings.TrimSpace(b.Vary)

		// Responses with Vary: "*" are least preferred
		aIsStar := aVary == "*"
		bIsStar := bVary == "*"
		if aIsStar && !bIsStar {
			return 1 // b preferred
		}
		if bIsStar && !aIsStar {
			return -1 // a preferred
		}

		// Responses with Vary headers are preferred over those without
		aHasVary := aVary != ""
		bHasVary := bVary != ""
		if aHasVary && !bHasVary {
			return -1 // a preferred
		}
		if !aHasVary && bHasVary {
			return 1 // b preferred
		}

		// If both have Vary headers, sort by Date or ResponseTime
		return a.ReceivedAt.Compare(b.ReceivedAt)
	})

	for i, entry := range entries {
		if vm.varyHeadersMatchOne(entry, reqHdr) {
			return i, true // Found a match
		}
	}

	return -1, false // No match found
}

func (vm *varyMatcher) varyHeadersMatchOne(entry *ResponseRef, reqHeader http.Header) bool {
	if entry.Vary == "*" {
		return false // Vary: "*" never matches
	}
	for field, value := range entry.VaryResolved {
		reqValues := reqHeader[field]
		// an empty value is comparable and means "no variation"
		reqValue := ""
		if len(reqValues) > 0 {
			// NOTE: The policy of this cache is to use just the first header line
			reqValue = vm.hvn.NormalizeHeaderValue(field, reqValues[0])
		}
		if reqValue != value {
			return false
		}
	}
	return true
}

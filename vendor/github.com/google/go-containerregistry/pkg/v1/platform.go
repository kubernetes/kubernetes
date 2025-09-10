// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package v1

import (
	"fmt"
	"sort"
	"strings"
)

// Platform represents the target os/arch for an image.
type Platform struct {
	Architecture string   `json:"architecture"`
	OS           string   `json:"os"`
	OSVersion    string   `json:"os.version,omitempty"`
	OSFeatures   []string `json:"os.features,omitempty"`
	Variant      string   `json:"variant,omitempty"`
	Features     []string `json:"features,omitempty"`
}

func (p Platform) String() string {
	if p.OS == "" {
		return ""
	}
	var b strings.Builder
	b.WriteString(p.OS)
	if p.Architecture != "" {
		b.WriteString("/")
		b.WriteString(p.Architecture)
	}
	if p.Variant != "" {
		b.WriteString("/")
		b.WriteString(p.Variant)
	}
	if p.OSVersion != "" {
		b.WriteString(":")
		b.WriteString(p.OSVersion)
	}
	return b.String()
}

// ParsePlatform parses a string representing a Platform, if possible.
func ParsePlatform(s string) (*Platform, error) {
	var p Platform
	parts := strings.Split(strings.TrimSpace(s), ":")
	if len(parts) == 2 {
		p.OSVersion = parts[1]
	}
	parts = strings.Split(parts[0], "/")
	if len(parts) > 0 {
		p.OS = parts[0]
	}
	if len(parts) > 1 {
		p.Architecture = parts[1]
	}
	if len(parts) > 2 {
		p.Variant = parts[2]
	}
	if len(parts) > 3 {
		return nil, fmt.Errorf("too many slashes in platform spec: %s", s)
	}
	return &p, nil
}

// Equals returns true if the given platform is semantically equivalent to this one.
// The order of Features and OSFeatures is not important.
func (p Platform) Equals(o Platform) bool {
	return p.OS == o.OS &&
		p.Architecture == o.Architecture &&
		p.Variant == o.Variant &&
		p.OSVersion == o.OSVersion &&
		stringSliceEqualIgnoreOrder(p.OSFeatures, o.OSFeatures) &&
		stringSliceEqualIgnoreOrder(p.Features, o.Features)
}

// Satisfies returns true if this Platform "satisfies" the given spec Platform.
//
// Note that this is different from Equals and that Satisfies is not reflexive.
//
// The given spec represents "requirements" such that any missing values in the
// spec are not compared.
//
// For OSFeatures and Features, Satisfies will return true if this Platform's
// fields contain a superset of the values in the spec's fields (order ignored).
func (p Platform) Satisfies(spec Platform) bool {
	return satisfies(spec.OS, p.OS) &&
		satisfies(spec.Architecture, p.Architecture) &&
		satisfies(spec.Variant, p.Variant) &&
		satisfies(spec.OSVersion, p.OSVersion) &&
		satisfiesList(spec.OSFeatures, p.OSFeatures) &&
		satisfiesList(spec.Features, p.Features)
}

func satisfies(want, have string) bool {
	return want == "" || want == have
}

func satisfiesList(want, have []string) bool {
	if len(want) == 0 {
		return true
	}

	set := map[string]struct{}{}
	for _, h := range have {
		set[h] = struct{}{}
	}

	for _, w := range want {
		if _, ok := set[w]; !ok {
			return false
		}
	}

	return true
}

// stringSliceEqual compares 2 string slices and returns if their contents are identical.
func stringSliceEqual(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i, elm := range a {
		if elm != b[i] {
			return false
		}
	}
	return true
}

// stringSliceEqualIgnoreOrder compares 2 string slices and returns if their contents are identical, ignoring order
func stringSliceEqualIgnoreOrder(a, b []string) bool {
	if a != nil && b != nil {
		sort.Strings(a)
		sort.Strings(b)
	}
	return stringSliceEqual(a, b)
}

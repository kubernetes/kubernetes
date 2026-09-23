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
	"testing"

	resourceapi "k8s.io/api/resource/v1"
)

func TestLookupByQualifiedName(t *testing.T) {
	const defaultDomain = "example.com"

	for name, tc := range map[string]struct {
		m         map[resourceapi.QualifiedName]int
		key       resourceapi.QualifiedName
		wantValue int
		wantOK    bool
	}{
		"unqualified-key-map-has-unqualified-only": {
			m:         map[resourceapi.QualifiedName]int{"foo": 1},
			key:       "foo",
			wantValue: 1,
			wantOK:    true,
		},
		"unqualified-key-map-has-qualified-only": {
			m:         map[resourceapi.QualifiedName]int{"example.com/foo": 2},
			key:       "foo",
			wantValue: 2,
			wantOK:    true,
		},
		"unqualified-key-map-has-both-prefers-qualified": {
			m:         map[resourceapi.QualifiedName]int{"foo": 1, "example.com/foo": 2},
			key:       "foo",
			wantValue: 2,
			wantOK:    true,
		},
		"unqualified-key-map-has-neither": {
			m:      map[resourceapi.QualifiedName]int{},
			key:    "foo",
			wantOK: false,
		},
		"qualified-key-matching-domain-map-has-qualified-only": {
			m:         map[resourceapi.QualifiedName]int{"example.com/foo": 2},
			key:       "example.com/foo",
			wantValue: 2,
			wantOK:    true,
		},
		"qualified-key-matching-domain-map-has-unqualified-only": {
			m:         map[resourceapi.QualifiedName]int{"foo": 1},
			key:       "example.com/foo",
			wantValue: 1,
			wantOK:    true,
		},
		"qualified-key-matching-domain-map-has-both-prefers-qualified": {
			m:         map[resourceapi.QualifiedName]int{"foo": 1, "example.com/foo": 2},
			key:       "example.com/foo",
			wantValue: 2,
			wantOK:    true,
		},
		"qualified-key-different-domain-map-has-unqualified-only-not-found": {
			m:      map[resourceapi.QualifiedName]int{"foo": 1},
			key:    "other.com/foo",
			wantOK: false,
		},
		"qualified-key-different-domain-map-has-matching-qualified": {
			m:         map[resourceapi.QualifiedName]int{"other.com/foo": 3},
			key:       "other.com/foo",
			wantValue: 3,
			wantOK:    true,
		},
		"empty-map": {
			m:      nil,
			key:    "foo",
			wantOK: false,
		},
	} {
		t.Run(name, func(t *testing.T) {
			value, ok := LookupByQualifiedName(tc.m, tc.key, defaultDomain)
			if ok != tc.wantOK {
				t.Fatalf("got ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && value != tc.wantValue {
				t.Fatalf("got value = %v, want %v", value, tc.wantValue)
			}
		})
	}
}

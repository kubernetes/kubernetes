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

package meta

import (
	"errors"
	"fmt"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestErrorMatching(t *testing.T) {
	testCases := []struct {
		name string
		// input should contain an error that is _not_ empty, otherwise the naive reflectlite.DeepEqual matching of
		// the errors lib will always succeed, but for all of these we want to verify that the matching is based on
		// type.
		input       error
		new         func() error
		matcherFunc func(error) bool
	}{
		{
			name:        "AmbiguousResourceError",
			input:       &AmbiguousResourceError{MatchingResources: []schema.GroupVersionResource{{}}},
			new:         func() error { return &AmbiguousResourceError{} },
			matcherFunc: IsAmbiguousError,
		},
		{
			name:        "AmbiguousKindError",
			input:       &AmbiguousKindError{MatchingResources: []schema.GroupVersionResource{{}}},
			new:         func() error { return &AmbiguousKindError{} },
			matcherFunc: IsAmbiguousError,
		},
		{
			name:        "NoResourceMatchError",
			input:       &NoResourceMatchError{PartialResource: schema.GroupVersionResource{Group: "foo"}},
			new:         func() error { return &NoResourceMatchError{} },
			matcherFunc: IsNoMatchError,
		},
		{
			name:        "NoKindMatchError",
			input:       &NoKindMatchError{SearchedVersions: []string{"foo"}},
			new:         func() error { return &NoKindMatchError{} },
			matcherFunc: IsNoMatchError,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			if !errors.Is(tc.input, tc.new()) {
				t.Error("error doesn't match itself directly")
			}
			if !errors.Is(fmt.Errorf("wrapepd: %w", tc.input), tc.new()) {
				t.Error("error doesn't match itself when wrapped")
			}
			if !tc.matcherFunc(tc.input) {
				t.Errorf("error doesn't get matched by matcherfunc")
			}
			if errors.Is(tc.input, errors.New("foo")) {
				t.Error("error incorrectly matches other error")
			}
		})
	}
}

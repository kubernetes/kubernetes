/*
Copyright 2025 The Kubernetes Authors.

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

package kuberc

import (
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestDecodePreference(t *testing.T) {
	testCases := map[string]struct {
		kuberc           string
		expectedAliases  []string
		expectedDefaults []string
		expectedError    string
	}{
		"v1alpha1": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "v1alpha1.kuberc"),
			expectedDefaults: []string{"v1alpha1-apply", "v1alpha1-delete"},
		},
		"v1beta1": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "v1beta1.kuberc"),
			expectedDefaults: []string{"v1beta1-apply", "v1beta1-delete"},
		},
		"first known version (v1beta1) with all versions": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "allversions.kuberc"),
			expectedAliases:  []string{"getn", "runx"},
			expectedDefaults: []string{"v1beta1-apply", "v1beta1-delete"},
		},
		"first known (v1beta1) with multiple versions (unknown, v1beta1, v1alpha1)": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "multiple1.kuberc"),
			expectedDefaults: []string{"v1beta1-apply", "v1beta1-delete"},
		},
		"first known (v1beta1) with multiple versions (unknown, v1beta1, v1beta1, v1alpha1)": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "multiple2.kuberc"),
			expectedDefaults: []string{"v1beta1-apply-first", "v1beta1-delete-first"},
		},
		"first known older (v1alpha1) with multiple versions (unknown, v1alpha1)": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "multiple3.kuberc"),
			expectedDefaults: []string{"v1alpha1-apply-first", "v1alpha1-delete-first"},
		},
		"first v1alpha1 with multiple versions (unknown, v1alpha1, v1beta1)": {
			kuberc:           filepath.Join("..", "..", "testdata", "kuberc", "multiple4.kuberc"),
			expectedDefaults: []string{"v1alpha1-apply-first", "v1alpha1-delete-first"},
		},
		"single unknown version": {
			kuberc:        filepath.Join("..", "..", "testdata", "kuberc", "unknown.kuberc"),
			expectedError: "no valid preferences found",
		},
		"multiple unknown version": {
			kuberc:        filepath.Join("..", "..", "testdata", "kuberc", "unknown.kuberc"),
			expectedError: "no valid preferences found",
		},
		"non-existent file": {
			kuberc:        filepath.Join("..", "..", "testdata", "kuberc", "non-existent"),
			expectedError: "no such file or directory",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			actual, err := decodePreference(tc.kuberc)
			if len(tc.expectedError) != 0 {
				require.ErrorContains(t, err, tc.expectedError, "wrong expected error")
				return
			}
			require.NoError(t, err, "unexpected error")
			require.NotNil(t, actual, "missing preferences when decoding")
			defaults := []string{}
			for _, o := range actual.Defaults {
				defaults = append(defaults, o.Command)
			}
			require.ElementsMatch(t, defaults, tc.expectedDefaults, "defaults mismatch")
			aliases := []string{}
			for _, o := range actual.Aliases {
				aliases = append(aliases, o.Name)
			}
			require.ElementsMatch(t, aliases, tc.expectedAliases, "aliases mismatch")
		})
	}
}

func TestDecodeEmptyPreference(t *testing.T) {
	actual, err := decodePreference(filepath.Join("..", "..", "testdata", "kuberc", "empty.kuberc"))
	require.NoError(t, err, "unexpected error")
	require.Nil(t, actual, "unexpected preferences")
}

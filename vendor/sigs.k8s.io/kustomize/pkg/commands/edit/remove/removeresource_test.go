/*
Copyright 2019 The Kubernetes Authors.

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

package remove

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestRemoveResources(t *testing.T) {
	type given struct {
		resources  []string
		removeArgs []string
	}
	type expected struct {
		resources []string
		deleted   []string
		err       error
	}
	testCases := []struct {
		description string
		given       given
		expected    expected
	}{
		{
			description: "remove resource",
			given: given{
				resources: []string{
					"resource1.yaml",
					"resource2.yaml",
					"resource3.yaml",
				},
				removeArgs: []string{"resource1.yaml"},
			},
			expected: expected{
				resources: []string{
					"resource2.yaml",
					"resource3.yaml",
				},
				deleted: []string{
					"resource1.yaml",
				},
			},
		},
		{
			description: "remove resources with pattern",
			given: given{
				resources: []string{
					"foo/resource1.yaml",
					"foo/resource2.yaml",
					"foo/resource3.yaml",
					"do/not/deleteme/please.yaml",
				},
				removeArgs: []string{"foo/resource*.yaml"},
			},
			expected: expected{
				resources: []string{
					"do/not/deleteme/please.yaml",
				},
				deleted: []string{
					"foo/resource1.yaml",
					"foo/resource2.yaml",
					"foo/resource3.yaml",
				},
			},
		},
		{
			description: "nothing found to remove",
			given: given{
				resources: []string{
					"resource1.yaml",
					"resource2.yaml",
					"resource3.yaml",
				},
				removeArgs: []string{"foo"},
			},
			expected: expected{
				resources: []string{
					"resource2.yaml",
					"resource3.yaml",
					"resource1.yaml",
				},
			},
		},
		{
			description: "no arguments",
			given:       given{},
			expected: expected{
				err: errors.New("must specify a resource file"),
			},
		},
		{
			description: "remove with multiple pattern arguments",
			given: given{
				resources: []string{
					"foo/foo.yaml",
					"bar/bar.yaml",
					"resource3.yaml",
					"do/not/deleteme/please.yaml",
				},
				removeArgs: []string{
					"foo/*.*",
					"bar/*.*",
					"res*.yaml",
				},
			},
			expected: expected{
				resources: []string{
					"do/not/deleteme/please.yaml",
				},
				deleted: []string{
					"foo/foo.yaml",
					"bar/bar.yaml",
					"resource3.yaml",
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			fakeFS := fs.MakeFakeFS()
			fakeFS.WriteTestKustomizationWith([]byte(fmt.Sprintf("resources:\n  - %s", strings.Join(tc.given.resources, "\n  - "))))
			cmd := newCmdRemoveResource(fakeFS)
			err := cmd.RunE(cmd, tc.given.removeArgs)
			if err != nil && tc.expected.err == nil {

				t.Errorf("unexpected cmd error: %v", err)
			} else if tc.expected.err != nil {
				if err.Error() != tc.expected.err.Error() {
					t.Errorf("expected error did not occurred. Expected: %v. Actual: %v", tc.expected.err, err)
				}
				return
			}
			content, err := fakeFS.ReadTestKustomization()
			if err != nil {
				t.Errorf("unexpected read error: %v", err)
			}

			for _, resourceFileName := range tc.expected.resources {
				if !strings.Contains(string(content), resourceFileName) {
					t.Errorf("expected resource not found in kustomization file.\nResource: %s\nKustomization file:\n%s", resourceFileName, content)
				}
			}
			for _, resourceFileName := range tc.expected.deleted {
				if strings.Contains(string(content), resourceFileName) {
					t.Errorf("expected deleted resource found in kustomization file. Resource: %s\nKustomization file:\n%s", resourceFileName, content)
				}
			}

		})
	}
}

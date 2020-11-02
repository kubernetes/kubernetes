/*
Copyright 2020 The Kubernetes Authors.

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

package gkeautopilot

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/apimachinery/pkg/util/sets"
)

func TestTreeNodeAddPath(t *testing.T) {
	samplePath := []string{"s1", "s2", "s3", "s4", "s5"}
	sampleVerbSet := sets.NewString([]string{"verb1", "verb2"}...)
	type input struct {
		path     []string
		verbList sets.String
	}

	type expected struct {
		assertFn func(*treeNode) error
	}

	testCases := map[string]struct {
		input    input
		expected expected
	}{
		"NormalPath_AddsToExpectedNode": {
			input{
				path:     samplePath,
				verbList: sampleVerbSet,
			},
			expected{
				assertFn: func(root *treeNode) error {
					node := root
					for _, p := range samplePath {
						node = node.children[p]

						if node == nil {
							return fmt.Errorf("node is nil")
						}
					}

					if diff := cmp.Diff(sampleVerbSet, node.allowedVerbs); diff != "" {
						return fmt.Errorf("-want verbList, +got verbList:\n%s", diff)
					}

					return nil
				},
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {

			root := newTreeNode()
			root.addPath(tc.input.path, tc.input.verbList)

			observedErr := tc.expected.assertFn(root)
			if observedErr != nil {
				t.Errorf("AddNode assert failed: %s", observedErr)
			}
		})
	}
}

func TestValidateConfig(t *testing.T) {
	type input struct {
		conf *config
	}
	type expected struct {
		err error
	}

	testCases := map[string]struct {
		input    input
		expected expected
	}{
		"ManagedNamespaceNameEmpty_ReturnsErr": {
			input{
				conf: &config{
					ManagedNamespaces: []ManagedNamespace{
						{
							Name: "",
						},
					},
				},
			},
			expected{
				err: errManagedNamespaceEmptyName,
			},
		},
		"ManagedNamespaceDeniedVerbsHasEmptyItem_ReturnsErr": {
			input{
				conf: &config{
					ManagedNamespaces: []ManagedNamespace{
						{
							Name:        "ns1",
							DeniedVerbs: []string{"verb1", ""},
						},
					},
				},
			},
			expected{
				err: errManagedNamespaceEmptyVerbs,
			},
		},
		"ManagedNamespaceDeniedResroucesHasItemWithEmptyResource_ReturnsErr": {
			input{
				conf: &config{
					ManagedNamespaces: []ManagedNamespace{
						{
							Name: "ns1",
							DeniedResources: []ResourceSubresource{
								{Resource: "", Subresource: "exec"},
							},
						},
					},
				},
			},
			expected{
				err: errManagedNamespaceEmptyResource,
			},
		},
		"ManagedResourceResourceEmpty_ReturnsErr": {
			input{
				conf: &config{
					ManagedResources: []ManagedResources{
						{
							Resources: []ManagedResource{
								{
									// group.foo.bar/""/gadgets/pillar-obj
									APIGroup:  "group.foo.bar",
									Namespace: "",
									Resource:  "", // should not be empty
									Name:      "pillar-obj",
								},
							},
						},
					},
				},
			},
			expected{
				err: errManagedResourceEmptyResource,
			},
		},
		"ManagedResourceNameEmpty_ReturnsErr": {
			input{
				conf: &config{
					ManagedResources: []ManagedResources{
						{
							Resources: []ManagedResource{
								{
									// group.foo.bar/""/gadgets/pillar-obj
									APIGroup:  "group.foo.bar",
									Namespace: "",
									Resource:  "gadget",
									Name:      "", //should not be empty
								},
							},
						},
					},
				},
			},
			expected{
				err: errManagedResourceEmptyName,
			},
		},
		"ManagedResourceSubresourceNameEmpty_ReturnsErr": {
			input{
				conf: &config{
					ManagedResources: []ManagedResources{
						{
							Resources: []ManagedResource{
								{
									// group.foo.bar/""/gadgets/pillar-obj
									APIGroup:  "group.foo.bar",
									Namespace: "",
									Resource:  "gadget",
									Name:      "pillar-obj", //should not be empty
									Subresources: []Subresource{
										{
											Name: "",
										},
									},
								},
							},
						},
					},
				},
			},
			expected{
				err: errManagedResourceEmptySubresourceName,
			},
		},
		"ValidConfig_ReturnsNil": {
			input{
				conf: &config{
					ManagedNamespaces: []ManagedNamespace{
						{
							Name: "ns1",
						},
					},
					ManagedResources: []ManagedResources{
						{
							Resources: []ManagedResource{
								{
									// group.foo.bar/""/gadgets/pillar-obj
									APIGroup:  "group.foo.bar",
									Namespace: "",
									Resource:  "gadget",
									Name:      "pillar-obj", //should not be empty
									Subresources: []Subresource{
										{
											Name: "exec",
										},
									},
								},
							},
						},
					},
				},
			},
			expected{
				err: nil,
			},
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {

			observedErr := tc.input.conf.validate()
			if diff := cmp.Diff(tc.expected.err, observedErr); diff != "" {
				t.Errorf("validate config (...): -want error, +got error:\n%s", diff)
			}
		})
	}
}

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

package set

import (
	"fmt"
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/fs"
)

func TestSetImage(t *testing.T) {
	type given struct {
		args         []string
		infileImages []string
	}
	type expected struct {
		fileOutput []string
		err        error
	}
	testCases := []struct {
		description string
		given       given
		expected    expected
	}{
		{
			given: given{
				args: []string{"image1=my-image1:my-tag"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: image1",
					"  newName: my-image1",
					"  newTag: my-tag",
				}},
		},
		{
			given: given{
				args: []string{"image1=my-image1@sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3",
					"  name: image1",
					"  newName: my-image1",
				}},
		},
		{
			given: given{
				args: []string{"image1:my-tag"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: image1",
					"  newTag: my-tag",
				}},
		},
		{
			given: given{
				args: []string{"image1@sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3",
					"  name: image1",
				}},
		},
		{
			description: "<image>=<image>",
			given: given{
				args: []string{"ngnix=localhost:5000/my-project/ngnix"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: ngnix",
					"  newName: localhost:5000/my-project/ngnix",
				}},
		},
		{
			given: given{
				args: []string{"ngnix=localhost:5000/my-project/ngnix:dev-01"},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: ngnix",
					"  newName: localhost:5000/my-project/ngnix",
					"  newTag: dev-01",
				}},
		},
		{
			description: "override file",
			given: given{
				args: []string{"image1=foo.bar.foo:8800/foo/image1:foo-bar"},
				infileImages: []string{
					"images:",
					"- name: image1",
					"  newName: my-image1",
					"  newTag: my-tag",
					"- name: image2",
					"  newName: my-image2",
					"  newTag: my-tag2",
				},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: image1",
					"  newName: foo.bar.foo:8800/foo/image1",
					"  newTag: foo-bar",
					"- name: image2",
					"  newName: my-image2",
					"  newTag: my-tag2",
				}},
		},
		{
			description: "override new tag and new name with just a new tag",
			given: given{
				args: []string{"image1:v1"},
				infileImages: []string{
					"images:",
					"- name: image1",
					"  newTag: my-tag",
				},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: image1",
					"  newTag: v1",
				}},
		},
		{
			description: "multiple args with multiple overrides",
			given: given{
				args: []string{
					"image1=foo.bar.foo:8800/foo/image1:foo-bar",
					"image2=my-image2@sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3",
					"image3:my-tag",
				},
				infileImages: []string{
					"images:",
					"- name: image1",
					"  newName: my-image1",
					"  newTag: my-tag1",
					"- name: image2",
					"  newName: my-image2",
					"  newTag: my-tag2",
					"- name: image3",
					"  newTag: my-tag",
				},
			},
			expected: expected{
				fileOutput: []string{
					"images:",
					"- name: image1",
					"  newName: foo.bar.foo:8800/foo/image1",
					"  newTag: foo-bar",
					"- digest: sha256:24a0c4b4a4c0eb97a1aabb8e29f18e917d05abfe1b7a7c07857230879ce7d3d3",
					"  name: image2",
					"  newName: my-image2",
					"- name: image3",
					"  newTag: my-tag",
				}},
		},
		{
			description: "error: no args",
			expected: expected{
				err: errImageNoArgs,
			},
		},
		{
			description: "error: invalid args",
			given: given{
				args: []string{"bad", "args"},
			},
			expected: expected{
				err: errImageInvalidArgs,
			},
		},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s%v", tc.description, tc.given.args), func(t *testing.T) {
			// arrange
			fakeFS := fs.MakeFakeFS()
			cmd := newCmdSetImage(fakeFS)

			if len(tc.given.infileImages) > 0 {
				// write file with infileImages
				fakeFS.WriteTestKustomizationWith([]byte(strings.Join(tc.given.infileImages, "\n")))
			} else {
				// writes default kustomization file
				fakeFS.WriteTestKustomization()
			}

			// act
			err := cmd.RunE(cmd, tc.given.args)

			// assert
			if err != tc.expected.err {
				t.Errorf("Unexpedted error from set image command. Actual: %v\nExpected: %v", err, tc.expected.err)
				t.FailNow()
			}

			content, err := fakeFS.ReadTestKustomization()
			if err != nil {
				t.Errorf("unexpected read error: %v", err)
				t.FailNow()
			}
			expectedStr := strings.Join(tc.expected.fileOutput, "\n")
			if !strings.Contains(string(content), expectedStr) {
				t.Errorf("unexpected image in kustomization file. \nActual:\n%s\nExpected:\n%s", content, expectedStr)
			}
		})
	}
}

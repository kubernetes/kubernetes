/*
Copyright 2017 The Kubernetes Authors.

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

package azure

import (
	"testing"

	"k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsMasterNode(t *testing.T) {
	if isMasterNode(&v1.Node{}) {
		t.Errorf("Empty node should not be master!")
	}
	if isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "worker",
			},
		},
	}) {
		t.Errorf("Node labelled 'worker' should not be master!")
	}
	if !isMasterNode(&v1.Node{
		ObjectMeta: meta.ObjectMeta{
			Labels: map[string]string{
				nodeLabelRole: "master",
			},
		},
	}) {
		t.Errorf("Node should be master!")
	}
}

func TestGetLastSegment(t *testing.T) {
	tests := []struct {
		ID        string
		expected  string
		expectErr bool
	}{
		{
			ID:        "",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/",
			expected:  "",
			expectErr: true,
		},
		{
			ID:        "foo/bar",
			expected:  "bar",
			expectErr: false,
		},
		{
			ID:        "foo/bar/baz",
			expected:  "baz",
			expectErr: false,
		},
	}

	for _, test := range tests {
		s, e := getLastSegment(test.ID)
		if test.expectErr && e == nil {
			t.Errorf("Expected err, but it was nil")
			continue
		}
		if !test.expectErr && e != nil {
			t.Errorf("Unexpected error: %v", e)
			continue
		}
		if s != test.expected {
			t.Errorf("expected: %s, got %s", test.expected, s)
		}
	}
}

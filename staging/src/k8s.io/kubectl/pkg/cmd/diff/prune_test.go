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

package diff

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type testRESTMapper struct {
	meta.RESTMapper
	scope meta.RESTScope
}

func (m *testRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group:    gk.Group,
			Version:  "",
			Resource: "",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group:   gk.Group,
			Version: "",
			Kind:    gk.Kind,
		},
		Scope: m.scope,
	}, nil
}

func TestParsePruneResources(t *testing.T) {
	tests := []struct {
		mapper   *testRESTMapper
		gvks     []string
		expected []pruneResource
		err      bool
	}{
		{
			mapper: &testRESTMapper{
				scope: meta.RESTScopeNamespace,
			},
			gvks:     nil,
			expected: []pruneResource{},
			err:      false,
		},
		{
			mapper: &testRESTMapper{
				scope: meta.RESTScopeNamespace,
			},
			gvks:     []string{"group/kind/version/test"},
			expected: []pruneResource{},
			err:      true,
		},
		{
			mapper: &testRESTMapper{
				scope: meta.RESTScopeNamespace,
			},
			gvks:     []string{"group/kind/version"},
			expected: []pruneResource{{group: "group", version: "kind", kind: "version", namespaced: true}},
			err:      false,
		},
		{
			mapper: &testRESTMapper{
				scope: meta.RESTScopeRoot,
			},
			gvks:     []string{"group/kind/version"},
			expected: []pruneResource{{group: "group", version: "kind", kind: "version", namespaced: false}},
			err:      false,
		},
	}

	for _, tc := range tests {
		actual, err := parsePruneResources(tc.mapper, tc.gvks)
		if tc.err {
			assert.NotEmptyf(t, err, "parsePruneResources error expected but not fired")
		} else {
			assert.Equal(t, actual, tc.expected, "parsePruneResources failed expected %v actual %v", tc.expected, actual)
		}
	}
}

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

package runtime_test

import (
	"io"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimetesting "k8s.io/apimachinery/pkg/runtime/testing"
)

func gv(group, version string) schema.GroupVersion {
	return schema.GroupVersion{Group: group, Version: version}
}
func gvk(group, version, kind string) schema.GroupVersionKind {
	return schema.GroupVersionKind{Group: group, Version: version, Kind: kind}
}
func gk(group, kind string) schema.GroupKind {
	return schema.GroupKind{Group: group, Kind: kind}
}

func TestCoercingMultiGroupVersioner(t *testing.T) {
	testcases := []struct {
		name           string
		target         schema.GroupVersion
		preferredKinds []schema.GroupKind
		kinds          []schema.GroupVersionKind
		expectKind     schema.GroupVersionKind
		expectedId     string
	}{
		{
			name:           "matched preferred group/kind",
			target:         gv("mygroup", "__internal"),
			preferredKinds: []schema.GroupKind{gk("mygroup", "Foo"), gk("anothergroup", "Bar")},
			kinds:          []schema.GroupVersionKind{gvk("yetanother", "v1", "Baz"), gvk("anothergroup", "v1", "Bar")},
			expectKind:     gvk("mygroup", "__internal", "Bar"),
			expectedId:     "{\"accepted\":\"Foo.mygroup,Bar.anothergroup\",\"coerce\":\"true\",\"name\":\"multi\",\"target\":\"mygroup/__internal\"}",
		},
		{
			name:           "matched preferred group",
			target:         gv("mygroup", "__internal"),
			preferredKinds: []schema.GroupKind{gk("mygroup", ""), gk("anothergroup", "")},
			kinds:          []schema.GroupVersionKind{gvk("yetanother", "v1", "Baz"), gvk("anothergroup", "v1", "Bar")},
			expectKind:     gvk("mygroup", "__internal", "Bar"),
			expectedId:     "{\"accepted\":\".mygroup,.anothergroup\",\"coerce\":\"true\",\"name\":\"multi\",\"target\":\"mygroup/__internal\"}",
		},
		{
			name:           "no preferred group/kind match, uses first kind in list",
			target:         gv("mygroup", "__internal"),
			preferredKinds: []schema.GroupKind{gk("mygroup", ""), gk("anothergroup", "")},
			kinds:          []schema.GroupVersionKind{gvk("yetanother", "v1", "Baz"), gvk("yetanother", "v1", "Bar")},
			expectKind:     gvk("mygroup", "__internal", "Baz"),
			expectedId:     "{\"accepted\":\".mygroup,.anothergroup\",\"coerce\":\"true\",\"name\":\"multi\",\"target\":\"mygroup/__internal\"}",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			v := runtime.NewCoercingMultiGroupVersioner(tc.target, tc.preferredKinds...)
			kind, ok := v.KindForGroupVersionKinds(tc.kinds)
			if !ok {
				t.Error("got no kind")
			}
			if kind != tc.expectKind {
				t.Errorf("expected %#v, got %#v", tc.expectKind, kind)
			}
			if e, a := tc.expectedId, v.Identifier(); e != a {
				t.Errorf("unexpected identifier: %s, expected: %s", a, e)
			}
		})
	}
}

type mockEncoder struct{}

func (m *mockEncoder) Encode(obj runtime.Object, w io.Writer) error {
	_, err := w.Write([]byte("mock-result"))
	return err
}

func (m *mockEncoder) Identifier() runtime.Identifier {
	return runtime.Identifier("mock-identifier")
}

func TestCacheableObject(t *testing.T) {
	serializer := runtime.NewBase64Serializer(&mockEncoder{}, nil)
	runtimetesting.CacheableObjectTest(t, serializer)
}

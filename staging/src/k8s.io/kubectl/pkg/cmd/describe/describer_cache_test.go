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

package describe

import (
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"
	fuzz "github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubectl/pkg/describe"
)

type fakeDescriber struct {
	name string
	describe.ResourceDescriber
}

func (d fakeDescriber) String() string { return fmt.Sprintf("fakeDescriber{%s}", d.name) }

type fakeScope string

func (s fakeScope) Name() meta.RESTScopeName { return meta.RESTScopeName(s) }

var (
	podDescriber = fakeDescriber{name: "pods"}
	podMapping   = &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "pods",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "",
			Version: "v1",
			Kind:    "Pod",
		},
		Scope: meta.RESTScopeNamespace,
	}

	ingressv1beta1Describer = fakeDescriber{name: "ingressv1beta1"}
	ingressv1beta1Mapping   = &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group:    "networking.k8s.io",
			Version:  "v1beta1",
			Resource: "ingresses",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "networking.k8s.io",
			Version: "v1beta1",
			Kind:    "Ingress",
		},
		Scope: meta.RESTScopeNamespace,
	}

	ingressv1Describer = fakeDescriber{name: "ingressv1"}
	ingressv1Mapping   = &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group:    "networking.k8s.io",
			Version:  "v1",
			Resource: "ingresses",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "networking.k8s.io",
			Version: "v1",
			Kind:    "Ingress",
		},
		Scope: meta.RESTScopeNamespace,
	}

	namespaceDescriber = fakeDescriber{name: "namespace"}
	namespaceMapping   = &meta.RESTMapping{
		Resource: schema.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "namespaces",
		},
		GroupVersionKind: schema.GroupVersionKind{
			Group:   "",
			Version: "v1",
			Kind:    "Namespace",
		},
		Scope: meta.RESTScopeRoot,
	}
)

func TestDescriberCache(t *testing.T) {
	t.Parallel()

	dc := newDescriberCache()
	dc.put(podMapping, podDescriber)
	dc.put(ingressv1beta1Mapping, ingressv1beta1Describer)
	assertCacheGet(t, dc, podMapping, podDescriber)
	assertCacheGet(t, dc, ingressv1beta1Mapping, ingressv1beta1Describer)
	assertCacheGet(t, dc, ingressv1Mapping, nil)
	assertCacheGet(t, dc, namespaceMapping, nil)

	dc.put(ingressv1Mapping, ingressv1Describer)
	dc.put(namespaceMapping, namespaceDescriber)
	assertCacheGet(t, dc, podMapping, podDescriber)
	assertCacheGet(t, dc, ingressv1beta1Mapping, ingressv1beta1Describer)
	assertCacheGet(t, dc, ingressv1Mapping, ingressv1Describer)
	assertCacheGet(t, dc, namespaceMapping, namespaceDescriber)
}

// TestDescriberCacheKey attempts to serve as a reminder to update the describer cache key
// contents if a new field is added to meta.RESTMapping.
func TestDescriberCacheKey(t *testing.T) {
	t.Parallel()

	f := fuzz.New().NilChance(0).Funcs(
		func(s *meta.RESTScope, f fuzz.Continue) {
			var scope string
			f.Fuzz(&scope)
			*s = fakeScope(scope)
		},
	)
	for i := 0; i < 100; i++ {
		actual := meta.RESTMapping{}
		f.Fuzz(&actual)

		// Set fields that we are using in the cache key. If new fields are added to meta.RESTMapping,
		// then we should most likely add them to the cache key and here as well.
		actual.Resource = podMapping.Resource
		actual.GroupVersionKind = podMapping.GroupVersionKind
		actual.Scope = podMapping.Scope

		expected := *podMapping
		cmpScope := func(a, b meta.RESTScope) bool {
			return a.Name() == b.Name()
		}
		if diff := cmp.Diff(actual, expected, cmp.Comparer(cmpScope)); diff != "" {
			t.Fatalf("we are missing some RESTMapping fields from our cache key, -got, +want:\n %s", diff)
		}
	}
}

func assertCacheGet(
	t *testing.T,
	dc *describerCache,
	mapping *meta.RESTMapping,
	want describe.ResourceDescriber,
) {
	t.Helper()
	got := dc.get(mapping)
	if want != got {
		t.Errorf("mapping %#v returned %s, wanted %s", mapping, got, want)
	}
}

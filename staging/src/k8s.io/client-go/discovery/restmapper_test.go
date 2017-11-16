/*
Copyright 2016 The Kubernetes Authors.

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

package discovery_test

import (
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	. "k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"

	"github.com/googleapis/gnostic/OpenAPIv2"
	"github.com/stretchr/testify/assert"
)

func TestRESTMapper(t *testing.T) {
	resources := []*APIGroupResources{
		{
			Group: metav1.APIGroup{
				Name: "extensions",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1beta"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1beta"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1beta": {
					{Name: "jobs", Namespaced: true, Kind: "Job"},
					{Name: "pods", Namespaced: true, Kind: "Pod"},
				},
			},
		},
		{
			Group: metav1.APIGroup{
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
					{Version: "v2"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
				},
				"v2": {
					{Name: "pods", Namespaced: true, Kind: "Pod"},
				},
			},
		},

		// This group tests finding and prioritizing resources that only exist in non-preferred versions
		{
			Group: metav1.APIGroup{
				Name: "unpreferred",
				Versions: []metav1.GroupVersionForDiscovery{
					{Version: "v1"},
					{Version: "v2beta1"},
					{Version: "v2alpha1"},
				},
				PreferredVersion: metav1.GroupVersionForDiscovery{Version: "v1"},
			},
			VersionedResources: map[string][]metav1.APIResource{
				"v1": {
					{Name: "broccoli", Namespaced: true, Kind: "Broccoli"},
				},
				"v2beta1": {
					{Name: "broccoli", Namespaced: true, Kind: "Broccoli"},
					{Name: "peas", Namespaced: true, Kind: "Pea"},
				},
				"v2alpha1": {
					{Name: "broccoli", Namespaced: true, Kind: "Broccoli"},
					{Name: "peas", Namespaced: true, Kind: "Pea"},
				},
			},
		},
	}

	restMapper := NewRESTMapper(resources, nil)

	kindTCs := []struct {
		input schema.GroupVersionResource
		want  schema.GroupVersionKind
	}{
		{
			input: schema.GroupVersionResource{
				Resource: "pods",
			},
			want: schema.GroupVersionKind{
				Version: "v1",
				Kind:    "Pod",
			},
		},
		{
			input: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
			want: schema.GroupVersionKind{
				Version: "v1",
				Kind:    "Pod",
			},
		},
		{
			input: schema.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
			want: schema.GroupVersionKind{
				Version: "v2",
				Kind:    "Pod",
			},
		},
		{
			input: schema.GroupVersionResource{
				Resource: "pods",
			},
			want: schema.GroupVersionKind{
				Version: "v1",
				Kind:    "Pod",
			},
		},
		{
			input: schema.GroupVersionResource{
				Resource: "jobs",
			},
			want: schema.GroupVersionKind{
				Group:   "extensions",
				Version: "v1beta",
				Kind:    "Job",
			},
		},
		{
			input: schema.GroupVersionResource{
				Resource: "peas",
			},
			want: schema.GroupVersionKind{
				Group:   "unpreferred",
				Version: "v2beta1",
				Kind:    "Pea",
			},
		},
	}

	for _, tc := range kindTCs {
		got, err := restMapper.KindFor(tc.input)
		if err != nil {
			t.Errorf("KindFor(%#v) unexpected error: %v", tc.input, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("KindFor(%#v) = %#v, want %#v", tc.input, got, tc.want)
		}
	}

	resourceTCs := []struct {
		input schema.GroupVersionResource
		want  schema.GroupVersionResource
	}{
		{
			input: schema.GroupVersionResource{
				Resource: "pods",
			},
			want: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			input: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
			want: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			input: schema.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
			want: schema.GroupVersionResource{
				Version:  "v2",
				Resource: "pods",
			},
		},
		{
			input: schema.GroupVersionResource{
				Resource: "pods",
			},
			want: schema.GroupVersionResource{
				Version:  "v1",
				Resource: "pods",
			},
		},
		{
			input: schema.GroupVersionResource{
				Resource: "jobs",
			},
			want: schema.GroupVersionResource{
				Group:    "extensions",
				Version:  "v1beta",
				Resource: "jobs",
			},
		},
	}

	for _, tc := range resourceTCs {
		got, err := restMapper.ResourceFor(tc.input)
		if err != nil {
			t.Errorf("ResourceFor(%#v) unexpected error: %v", tc.input, err)
			continue
		}

		if !reflect.DeepEqual(got, tc.want) {
			t.Errorf("ResourceFor(%#v) = %#v, want %#v", tc.input, got, tc.want)
		}
	}
}

func TestDeferredDiscoveryRESTMapper_CacheMiss(t *testing.T) {
	assert := assert.New(t)

	cdc := fakeCachedDiscoveryInterface{fresh: false}
	m := NewDeferredDiscoveryRESTMapper(&cdc, nil)
	assert.False(cdc.fresh, "should NOT be fresh after instantiation")
	assert.Zero(cdc.invalidateCalls, "should not have called Invalidate()")

	gvk, err := m.KindFor(schema.GroupVersionResource{
		Group:    "a",
		Version:  "v1",
		Resource: "foo",
	})
	assert.NoError(err)
	assert.True(cdc.fresh, "should be fresh after a cache-miss")
	assert.Equal(cdc.invalidateCalls, 1, "should have called Invalidate() once")
	assert.Equal(gvk.Kind, "Foo")

	gvk, err = m.KindFor(schema.GroupVersionResource{
		Group:    "a",
		Version:  "v1",
		Resource: "foo",
	})
	assert.NoError(err)
	assert.Equal(cdc.invalidateCalls, 1, "should NOT have called Invalidate() again")

	gvk, err = m.KindFor(schema.GroupVersionResource{
		Group:    "a",
		Version:  "v1",
		Resource: "bar",
	})
	assert.Error(err)
	assert.Equal(cdc.invalidateCalls, 1, "should NOT have called Invalidate() again after another cache-miss, but with fresh==true")

	cdc.fresh = false
	gvk, err = m.KindFor(schema.GroupVersionResource{
		Group:    "a",
		Version:  "v1",
		Resource: "bar",
	})
	assert.Error(err)
	assert.Equal(cdc.invalidateCalls, 2, "should HAVE called Invalidate() again after another cache-miss, but with fresh==false")
}

type fakeCachedDiscoveryInterface struct {
	invalidateCalls int
	fresh           bool
	enabledA        bool
}

var _ CachedDiscoveryInterface = &fakeCachedDiscoveryInterface{}

func (c *fakeCachedDiscoveryInterface) Fresh() bool {
	return c.fresh
}

func (c *fakeCachedDiscoveryInterface) Invalidate() {
	c.invalidateCalls = c.invalidateCalls + 1
	c.fresh = true
	c.enabledA = true
}

func (c *fakeCachedDiscoveryInterface) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeCachedDiscoveryInterface) ServerGroups() (*metav1.APIGroupList, error) {
	if c.enabledA {
		return &metav1.APIGroupList{
			Groups: []metav1.APIGroup{
				{
					Name: "a",
					Versions: []metav1.GroupVersionForDiscovery{
						{
							GroupVersion: "a/v1",
							Version:      "v1",
						},
					},
					PreferredVersion: metav1.GroupVersionForDiscovery{
						GroupVersion: "a/v1",
						Version:      "v1",
					},
				},
			},
		}, nil
	}
	return &metav1.APIGroupList{}, nil
}

func (c *fakeCachedDiscoveryInterface) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	if c.enabledA && groupVersion == "a/v1" {
		return &metav1.APIResourceList{
			GroupVersion: "a/v1",
			APIResources: []metav1.APIResource{
				{
					Name:       "foo",
					Kind:       "Foo",
					Namespaced: false,
				},
			},
		}, nil
	}

	return nil, errors.NewNotFound(schema.GroupResource{}, "")
}

func (c *fakeCachedDiscoveryInterface) ServerResources() ([]*metav1.APIResourceList, error) {
	if c.enabledA {
		av1, _ := c.ServerResourcesForGroupVersion("a/v1")
		return []*metav1.APIResourceList{av1}, nil
	}
	return []*metav1.APIResourceList{}, nil
}

func (c *fakeCachedDiscoveryInterface) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	if c.enabledA {
		return []*metav1.APIResourceList{
			{
				GroupVersion: "a/v1",
				APIResources: []metav1.APIResource{
					{
						Name:  "foo",
						Kind:  "Foo",
						Verbs: []string{},
					},
				},
			},
		}, nil
	}
	return nil, nil
}

func (c *fakeCachedDiscoveryInterface) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (c *fakeCachedDiscoveryInterface) ServerVersion() (*version.Info, error) {
	return &version.Info{}, nil
}

func (c *fakeCachedDiscoveryInterface) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

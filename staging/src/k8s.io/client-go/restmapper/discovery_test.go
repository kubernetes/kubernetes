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

package restmapper

import (
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/dump"
	"k8s.io/apimachinery/pkg/version"
	. "k8s.io/client-go/discovery"
	"k8s.io/client-go/openapi"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"

	openapi_v2 "github.com/google/gnostic-models/openapiv2"
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

	restMapper := NewDiscoveryRESTMapper(resources)

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
	m := NewDeferredDiscoveryRESTMapper(&cdc)
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

func TestGetAPIGroupResources(t *testing.T) {
	type Test struct {
		name string

		discovery DiscoveryInterface

		expected      []*APIGroupResources
		expectedError error
	}

	for _, test := range []Test{
		{"nil", &fakeFailingDiscovery{nil, nil, nil, nil}, nil, nil},
		{"normal",
			&fakeFailingDiscovery{
				[]metav1.APIGroup{aGroup, bGroup}, nil,
				map[string]*metav1.APIResourceList{"a/v1": &aResources, "b/v1": &bResources}, nil,
			},
			[]*APIGroupResources{
				{aGroup, map[string][]metav1.APIResource{"v1": {aFoo}}},
				{bGroup, map[string][]metav1.APIResource{"v1": {bBar}}},
			}, nil,
		},
		{"groups failed, but has fallback with a only",
			&fakeFailingDiscovery{
				[]metav1.APIGroup{aGroup}, fmt.Errorf("error fetching groups"),
				map[string]*metav1.APIResourceList{"a/v1": &aResources, "b/v1": &bResources}, nil,
			},
			[]*APIGroupResources{
				{aGroup, map[string][]metav1.APIResource{"v1": {aFoo}}},
			}, nil,
		},
		{"groups failed, but has no fallback",
			&fakeFailingDiscovery{
				nil, fmt.Errorf("error fetching groups"),
				map[string]*metav1.APIResourceList{"a/v1": &aResources, "b/v1": &bResources}, nil,
			},
			nil, fmt.Errorf("error fetching groups"),
		},
		{"a failed, but has fallback",
			&fakeFailingDiscovery{
				[]metav1.APIGroup{aGroup, bGroup}, nil,
				map[string]*metav1.APIResourceList{"a/v1": &aResources, "b/v1": &bResources}, map[string]error{"a/v1": fmt.Errorf("a failed")},
			},
			[]*APIGroupResources{
				{aGroup, map[string][]metav1.APIResource{"v1": {aFoo}}},
				{bGroup, map[string][]metav1.APIResource{"v1": {bBar}}},
			}, nil, // TODO: do we want this?
		},
		{"a failed, but has no fallback",
			&fakeFailingDiscovery{
				[]metav1.APIGroup{aGroup, bGroup}, nil,
				map[string]*metav1.APIResourceList{"b/v1": &bResources}, map[string]error{"a/v1": fmt.Errorf("a failed")},
			},
			[]*APIGroupResources{
				{aGroup, map[string][]metav1.APIResource{}},
				{bGroup, map[string][]metav1.APIResource{"v1": {bBar}}},
			}, nil, // TODO: do we want this?
		},
		{"a and b failed, but have fallbacks",
			&fakeFailingDiscovery{
				[]metav1.APIGroup{aGroup, bGroup}, nil,
				map[string]*metav1.APIResourceList{"a/v1": &aResources, "b/v1": &bResources}, // TODO: both fallbacks are ignored
				map[string]error{"a/v1": fmt.Errorf("a failed"), "b/v1": fmt.Errorf("b failed")},
			},
			[]*APIGroupResources{
				{aGroup, map[string][]metav1.APIResource{"v1": {aFoo}}},
				{bGroup, map[string][]metav1.APIResource{"v1": {bBar}}},
			}, nil, // TODO: do we want this?
		},
	} {
		t.Run(test.name, func(t *testing.T) {
			got, err := GetAPIGroupResources(test.discovery)
			if err == nil && test.expectedError != nil {
				t.Fatalf("expected error %q, but got none", test.expectedError)
			} else if err != nil && test.expectedError == nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !reflect.DeepEqual(test.expected, got) {
				t.Errorf("unexpected result:\nexpected = %s\ngot = %s", dump.Pretty(test.expected), dump.Pretty(got))
			}
		})
	}

}

var _ DiscoveryInterface = &fakeFailingDiscovery{}

type fakeFailingDiscovery struct {
	groups    []metav1.APIGroup
	groupsErr error

	resourcesForGroupVersion    map[string]*metav1.APIResourceList
	resourcesForGroupVersionErr map[string]error
}

func (*fakeFailingDiscovery) RESTClient() restclient.Interface {
	return nil
}

func (d *fakeFailingDiscovery) ServerGroups() (*metav1.APIGroupList, error) {
	if d.groups == nil && d.groupsErr != nil {
		return nil, d.groupsErr
	}
	return &metav1.APIGroupList{Groups: d.groups}, d.groupsErr
}

func (d *fakeFailingDiscovery) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return ServerGroupsAndResources(d)
}
func (d *fakeFailingDiscovery) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	if rs, found := d.resourcesForGroupVersion[groupVersion]; found {
		return rs, d.resourcesForGroupVersionErr[groupVersion]
	}
	return nil, fmt.Errorf("not found")
}

func (d *fakeFailingDiscovery) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return ServerPreferredResources(d)
}

func (d *fakeFailingDiscovery) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return ServerPreferredNamespacedResources(d)
}

func (*fakeFailingDiscovery) ServerVersion() (*version.Info, error) {
	return &version.Info{}, nil
}

func (*fakeFailingDiscovery) OpenAPISchema() (*openapi_v2.Document, error) {
	panic("implement me")
}

func (c *fakeFailingDiscovery) OpenAPIV3() openapi.Client {
	panic("implement me")
}

func (c *fakeFailingDiscovery) WithLegacy() DiscoveryInterface {
	panic("implement me")
}

type fakeCachedDiscoveryInterface struct {
	invalidateCalls int
	fresh           bool
	enabledGroupA   bool
}

var _ CachedDiscoveryInterface = &fakeCachedDiscoveryInterface{}

func (c *fakeCachedDiscoveryInterface) Fresh() bool {
	return c.fresh
}

func (c *fakeCachedDiscoveryInterface) Invalidate() {
	c.invalidateCalls = c.invalidateCalls + 1
	c.fresh = true
	c.enabledGroupA = true
}

func (c *fakeCachedDiscoveryInterface) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeCachedDiscoveryInterface) ServerGroups() (*metav1.APIGroupList, error) {
	if c.enabledGroupA {
		return &metav1.APIGroupList{
			Groups: []metav1.APIGroup{aGroup},
		}, nil
	}
	return &metav1.APIGroupList{}, nil
}

func (c *fakeCachedDiscoveryInterface) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	return ServerGroupsAndResources(c)
}

func (c *fakeCachedDiscoveryInterface) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	if c.enabledGroupA && groupVersion == "a/v1" {
		return &aResources, nil
	}

	return nil, errors.NewNotFound(schema.GroupResource{}, "")
}

func (c *fakeCachedDiscoveryInterface) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	if c.enabledGroupA {
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

func (c *fakeCachedDiscoveryInterface) OpenAPIV3() openapi.Client {
	panic("implement me")
}

func (c *fakeCachedDiscoveryInterface) WithLegacy() DiscoveryInterface {
	panic("implement me")
}

var (
	aGroup = metav1.APIGroup{
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
	}
	bGroup = metav1.APIGroup{
		Name: "b",
		Versions: []metav1.GroupVersionForDiscovery{
			{
				GroupVersion: "b/v1",
				Version:      "v1",
			},
		},
		PreferredVersion: metav1.GroupVersionForDiscovery{
			GroupVersion: "b/v1",
			Version:      "v1",
		},
	}
	aResources = metav1.APIResourceList{
		GroupVersion: "a/v1",
		APIResources: []metav1.APIResource{aFoo},
	}
	aFoo = metav1.APIResource{
		Name:       "foo",
		Kind:       "Foo",
		Namespaced: false,
	}
	bResources = metav1.APIResourceList{
		GroupVersion: "b/v1",
		APIResources: []metav1.APIResource{bBar},
	}
	bBar = metav1.APIResource{
		Name:       "bar",
		Kind:       "Bar",
		Namespaced: true,
	}
)

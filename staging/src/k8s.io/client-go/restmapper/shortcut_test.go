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
	"testing"

	openapi_v2 "github.com/googleapis/gnostic/openapiv2"

	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
)

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected schema.GroupVersionResource
		srvRes   []*metav1.APIResourceList
	}{
		{
			name:     "storageclasses-no-replacement",
			arg:      "storageclasses",
			expected: schema.GroupVersionResource{Resource: "storageclasses"},
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "hpa-priority",
			arg:      "hpa",
			expected: schema.GroupVersionResource{Resource: "superhorizontalpodautoscalers", Group: "autoscaling"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "autoscaling/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "superhorizontalpodautoscalers",
							ShortNames: []string{"hpa"},
						},
					},
				},
				{
					GroupVersion: "autoscaling/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "horizontalpodautoscalers",
							ShortNames: []string{"hpa"},
						},
					},
				},
			},
		},
		{
			name:     "resource-override",
			arg:      "dpl",
			expected: schema.GroupVersionResource{Resource: "deployments", Group: "foo"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "foo/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "deployments",
							ShortNames: []string{"dpl"},
						},
					},
				},
				{
					GroupVersion: "extension/v1beta1",
					APIResources: []metav1.APIResource{
						{
							Name:       "deployments",
							ShortNames: []string{"deploy"},
						},
					},
				},
			},
		},
		{
			name:     "resource-match-preferred",
			arg:      "pods",
			expected: schema.GroupVersionResource{Resource: "pods", Group: ""},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{{Name: "pods", SingularName: "pod"}},
				},
				{
					GroupVersion: "acme.com/v1",
					APIResources: []metav1.APIResource{{Name: "poddlers", ShortNames: []string{"pods", "pod"}}},
				},
			},
		},
		{
			name:     "resource-match-singular-preferred",
			arg:      "pod",
			expected: schema.GroupVersionResource{Resource: "pod", Group: ""},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{{Name: "pods", SingularName: "pod"}},
				},
				{
					GroupVersion: "acme.com/v1",
					APIResources: []metav1.APIResource{{Name: "poddlers", ShortNames: []string{"pods", "pod"}}},
				},
			},
		},
	}

	for _, test := range tests {
		ds := &fakeDiscoveryClient{}
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}
		mapper := NewShortcutExpander(&fakeRESTMapper{}, ds).(shortcutExpander)

		actual := mapper.expandResourceShortcut(schema.GroupVersionResource{Resource: test.arg})
		if actual != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, actual)
		}
	}
}

func TestKindFor(t *testing.T) {
	tests := []struct {
		in       schema.GroupVersionResource
		expected schema.GroupVersionResource
		srvRes   []*metav1.APIResourceList
	}{
		{
			in:       schema.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "sc"},
			expected: schema.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "storageclasses"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "storage.k8s.io/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "storageclasses",
							ShortNames: []string{"sc"},
						},
					},
				},
			},
		},
		{
			in:       schema.GroupVersionResource{Group: "", Version: "", Resource: "sc"},
			expected: schema.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "storageclasses"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "storage.k8s.io/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "storageclasses",
							ShortNames: []string{"sc"},
						},
					},
				},
			},
		},
	}

	for i, test := range tests {
		ds := &fakeDiscoveryClient{}
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}

		delegate := &fakeRESTMapper{}
		mapper := NewShortcutExpander(delegate, ds)

		mapper.KindFor(test.in)
		if delegate.kindForInput != test.expected {
			t.Errorf("%d: unexpected data returned %#v, expected %#v", i, delegate.kindForInput, test.expected)
		}
	}
}

type fakeRESTMapper struct {
	kindForInput schema.GroupVersionResource
}

func (f *fakeRESTMapper) KindFor(resource schema.GroupVersionResource) (schema.GroupVersionKind, error) {
	f.kindForInput = resource
	return schema.GroupVersionKind{}, nil
}

func (f *fakeRESTMapper) KindsFor(resource schema.GroupVersionResource) ([]schema.GroupVersionKind, error) {
	return nil, nil
}

func (f *fakeRESTMapper) ResourceFor(input schema.GroupVersionResource) (schema.GroupVersionResource, error) {
	return schema.GroupVersionResource{}, nil
}

func (f *fakeRESTMapper) ResourcesFor(input schema.GroupVersionResource) ([]schema.GroupVersionResource, error) {
	return nil, nil
}

func (f *fakeRESTMapper) RESTMapping(gk schema.GroupKind, versions ...string) (*meta.RESTMapping, error) {
	return nil, nil
}

func (f *fakeRESTMapper) RESTMappings(gk schema.GroupKind, versions ...string) ([]*meta.RESTMapping, error) {
	return nil, nil
}

func (f *fakeRESTMapper) ResourceSingularizer(resource string) (singular string, err error) {
	return "", nil
}

type fakeDiscoveryClient struct {
	serverResourcesHandler func() ([]*metav1.APIResourceList, error)
}

var _ discovery.DiscoveryInterface = &fakeDiscoveryClient{}

func (c *fakeDiscoveryClient) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeDiscoveryClient) ServerGroups() (*metav1.APIGroupList, error) {
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

func (c *fakeDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	if groupVersion == "a/v1" {
		return &metav1.APIResourceList{APIResources: []metav1.APIResource{{Name: "widgets", Kind: "Widget"}}}, nil
	}

	return nil, errors.NewNotFound(schema.GroupResource{}, "")
}

// Deprecated: use ServerGroupsAndResources instead.
func (c *fakeDiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	_, rs, err := c.ServerGroupsAndResources()
	return rs, err
}

func (c *fakeDiscoveryClient) ServerGroupsAndResources() ([]*metav1.APIGroup, []*metav1.APIResourceList, error) {
	sgs, err := c.ServerGroups()
	if err != nil {
		return nil, nil, err
	}
	resultGroups := []*metav1.APIGroup{}
	for i := range sgs.Groups {
		resultGroups = append(resultGroups, &sgs.Groups[i])
	}
	if c.serverResourcesHandler != nil {
		rs, err := c.serverResourcesHandler()
		return resultGroups, rs, err
	}
	return resultGroups, []*metav1.APIResourceList{}, nil
}

func (c *fakeDiscoveryClient) ServerPreferredResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (c *fakeDiscoveryClient) ServerPreferredNamespacedResources() ([]*metav1.APIResourceList, error) {
	return nil, nil
}

func (c *fakeDiscoveryClient) ServerVersion() (*version.Info, error) {
	return &version.Info{}, nil
}

func (c *fakeDiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

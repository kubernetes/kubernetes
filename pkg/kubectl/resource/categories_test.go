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

package resource

import (
	"reflect"
	"testing"

	swagger "github.com/emicklei/go-restful-swagger12"

	"github.com/googleapis/gnostic/OpenAPIv2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/version"
	"k8s.io/client-go/discovery"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
)

func TestCategoryExpansion(t *testing.T) {
	tests := []struct {
		name string
		arg  string

		expected   []schema.GroupResource
		expectedOk bool
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: nil,
		},
		{
			name: "all-replacement",
			arg:  "all",
			expected: []schema.GroupResource{
				{Resource: "pods"},
				{Resource: "replicationcontrollers"},
				{Resource: "services"},
				{Resource: "statefulsets", Group: "apps"},
				{Resource: "horizontalpodautoscalers", Group: "autoscaling"},
				{Resource: "jobs", Group: "batch"},
				{Resource: "cronjobs", Group: "batch"},
				{Resource: "daemonsets", Group: "extensions"},
				{Resource: "deployments", Group: "extensions"},
				{Resource: "replicasets", Group: "extensions"},
			},
			expectedOk: true,
		},
	}

	for _, test := range tests {
		actual, actualOk := LegacyCategoryExpander.Expand(test.arg)
		if e, a := test.expected, actual; !reflect.DeepEqual(e, a) {
			t.Errorf("%s:  expected %s, got %s", test.name, e, a)
		}
		if e, a := test.expectedOk, actualOk; e != a {
			t.Errorf("%s:  expected %v, got %v", test.name, e, a)
		}
	}
}

func TestDiscoveryCategoryExpander(t *testing.T) {
	tests := []struct {
		category       string
		serverResponse []*metav1.APIResourceList
		expected       []schema.GroupResource
	}{
		{
			category: "all",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
							Categories: []string{"all"},
						},
					},
				},
			},
			expected: []schema.GroupResource{
				{
					Group:    "batch",
					Resource: "jobs",
				},
			},
		},
		{
			category: "all",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
						},
					},
				},
			},
		},
		{
			category: "targaryens",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
							Categories: []string{"all"},
						},
					},
				},
			},
		},
	}

	dc := &fakeDiscoveryClient{}
	for _, test := range tests {
		dc.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.serverResponse, nil
		}
		expander, err := NewDiscoveryCategoryExpander(SimpleCategoryExpander{}, dc)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		expanded, _ := expander.Expand(test.category)
		if !reflect.DeepEqual(expanded, test.expected) {
			t.Errorf("expected %v, got %v", test.expected, expanded)
		}
	}

}

type fakeDiscoveryClient struct {
	serverResourcesHandler func() ([]*metav1.APIResourceList, error)
}

var _ discovery.DiscoveryInterface = &fakeDiscoveryClient{}

func (c *fakeDiscoveryClient) RESTClient() restclient.Interface {
	return &fake.RESTClient{}
}

func (c *fakeDiscoveryClient) ServerGroups() (*metav1.APIGroupList, error) {
	return nil, nil
}

func (c *fakeDiscoveryClient) ServerResourcesForGroupVersion(groupVersion string) (*metav1.APIResourceList, error) {
	return &metav1.APIResourceList{}, nil
}

func (c *fakeDiscoveryClient) ServerResources() ([]*metav1.APIResourceList, error) {
	return c.serverResourcesHandler()
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

func (c *fakeDiscoveryClient) SwaggerSchema(version schema.GroupVersion) (*swagger.ApiDeclaration, error) {
	return &swagger.ApiDeclaration{}, nil
}

func (c *fakeDiscoveryClient) OpenAPISchema() (*openapi_v2.Document, error) {
	return &openapi_v2.Document{}, nil
}

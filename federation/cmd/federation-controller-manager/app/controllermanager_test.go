package app

import (
	"testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestResourceEnabled(t *testing.T) {

	testCases := []struct {
		runtimeConfig		map[string]string
		serverResources 	map[string]*unversioned.APIResourceList
		resource		unversioned.GroupVersionResource
		expectedResult		bool
	}{
		// no override, API server has Ingress enabled
		{
			runtimeConfig: map[string]string{},
			serverResources: map[string]*unversioned.APIResourceList{
				"extensions/v1beta1": {
					GroupVersion: "extensions/v1beta1",
					APIResources: []unversioned.APIResource{
						{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
					},
				},
			},
			resource: extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses"),
			expectedResult: true,
		},
		// no override, API server has Ingress disabled
		{
			runtimeConfig: map[string]string{},
			serverResources: map[string]*unversioned.APIResourceList{},
			resource: extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses"),
			expectedResult: false,
		},
		// API server has Ingress enabled, override runtime config to disable Ingress
		{
			runtimeConfig: map[string]string{
				"extensions/v1beta1/ingresses": "false",
			},
			serverResources: map[string]*unversioned.APIResourceList{
				"extensions/v1beta1": {
					GroupVersion: "extensions/v1beta1",
					APIResources: []unversioned.APIResource{
						{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
					},
				},
			},
			resource: extensionsapiv1beta1.SchemeGroupVersion.WithResource("ingresses"),
			expectedResult: false,
		},
	}

	for _, test := range testCases {
		resourceConfig := defaultAPIResourceConfigSource()
		resourceConfig.MergeResourceConfigs(test.runtimeConfig)

		actualHasResource := resourceEnabled(resourceConfig, test.serverResources, test.resource)
		if actualHasResource != test.expectedResult {
			t.Errorf("%s: expected %v, got %v", test.resource.String(), test.expectedResult, actualHasResource)
		}
	}
}

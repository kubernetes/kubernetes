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

package app

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	ingresscontroller "k8s.io/kubernetes/federation/pkg/federation-controller/ingress"
	"k8s.io/kubernetes/pkg/util/config"
	"testing"
)

func TestControllerEnabled(t *testing.T) {

	testCases := []struct {
		controllersConfig config.ConfigurationMap
		serverResources   []*metav1.APIResourceList
		controller        string
		requiredResources []schema.GroupVersionResource
		defaultValue      bool
		expectedResult    bool
	}{
		// no override, API server has Ingress enabled
		{
			controllersConfig: config.ConfigurationMap{},
			serverResources: []*metav1.APIResourceList{
				{
					GroupVersion: "extensions/v1beta1",
					APIResources: []metav1.APIResource{
						{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
					},
				},
			},
			controller:        ingresscontroller.ControllerName,
			requiredResources: ingresscontroller.RequiredResources,
			defaultValue:      true,
			expectedResult:    true,
		},
		// no override, API server has Ingress disabled
		{
			controllersConfig: config.ConfigurationMap{},
			serverResources:   []*metav1.APIResourceList{},
			controller:        ingresscontroller.ControllerName,
			requiredResources: ingresscontroller.RequiredResources,
			defaultValue:      true,
			expectedResult:    false,
		},
		// API server has Ingress enabled, override config to disable Ingress controller
		{
			controllersConfig: config.ConfigurationMap{
				ingresscontroller.ControllerName: "false",
			},
			serverResources: []*metav1.APIResourceList{
				{
					GroupVersion: "extensions/v1beta1",
					APIResources: []metav1.APIResource{
						{Name: "ingresses", Namespaced: true, Kind: "Ingress"},
					},
				},
			},
			controller:        ingresscontroller.ControllerName,
			requiredResources: ingresscontroller.RequiredResources,
			defaultValue:      true,
			expectedResult:    false,
		},
	}

	for _, test := range testCases {
		actualEnabled := controllerEnabled(test.controllersConfig, test.serverResources, test.controller, test.requiredResources, test.defaultValue)
		if actualEnabled != test.expectedResult {
			t.Errorf("%s controller: expected %v, got %v", test.controller, test.expectedResult, actualEnabled)
		}
	}
}

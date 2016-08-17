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

package genericapiserver

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	extensionsapiv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
)

func TestParseRuntimeConfig(t *testing.T) {
	extensionsGroupVersion := extensionsapiv1beta1.SchemeGroupVersion
	apiv1GroupVersion := apiv1.SchemeGroupVersion
	testCases := []struct {
		runtimeConfig         map[string]string
		defaultResourceConfig func() *ResourceConfig
		expectedAPIConfig     func() *ResourceConfig
		err                   bool
	}{
		{
			// everything default value.
			runtimeConfig: map[string]string{},
			defaultResourceConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			expectedAPIConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			err: false,
		},
		{
			// no runtimeConfig override.
			runtimeConfig: map[string]string{},
			defaultResourceConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.DisableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.DisableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			err: false,
		},
		{
			// version enabled by runtimeConfig override.
			runtimeConfig: map[string]string{
				"extensions/v1beta1": "",
			},
			defaultResourceConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.DisableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.EnableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			err: false,
		},
		{
			// disable resource
			runtimeConfig: map[string]string{
				"api/v1/pods": "false",
			},
			defaultResourceConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.EnableVersions(apiv1GroupVersion)
				return config
			},
			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.EnableVersions(apiv1GroupVersion)
				config.DisableResources(apiv1GroupVersion.WithResource("pods"))
				return config
			},
			err: false,
		},
		{
			// Disable v1.
			runtimeConfig: map[string]string{
				"api/v1": "false",
			},
			defaultResourceConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.DisableVersions(apiv1GroupVersion)
				return config
			},
			err: false,
		},
		{
			// Enable deployments and disable jobs.
			runtimeConfig: map[string]string{
				"extensions/v1beta1/anything": "true",
				"extensions/v1beta1/jobs":     "false",
			},
			defaultResourceConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.EnableVersions(extensionsGroupVersion)
				return config
			},

			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.EnableVersions(extensionsGroupVersion)
				config.DisableResources(extensionsGroupVersion.WithResource("jobs"))
				config.EnableResources(extensionsGroupVersion.WithResource("anything"))
				return config
			},
			err: false,
		},
		{
			// invalid runtime config
			runtimeConfig: map[string]string{
				"invalidgroup/version": "false",
			},
			defaultResourceConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			expectedAPIConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			err: true,
		},
		{
			// cannot disable individual resource when version is not enabled.
			runtimeConfig: map[string]string{
				"api/v1/pods": "false",
			},
			defaultResourceConfig: func() *ResourceConfig {
				return NewResourceConfig()
			},
			expectedAPIConfig: func() *ResourceConfig {
				config := NewResourceConfig()
				config.DisableResources(unversioned.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"})
				return config
			},
			err: true,
		},
	}
	for _, test := range testCases {
		actualDisablers, err := mergeAPIResourceConfigs(test.defaultResourceConfig(), test.runtimeConfig)
		if err == nil && test.err {
			t.Fatalf("expected error for test: %v", test)
		} else if err != nil && !test.err {
			t.Fatalf("unexpected error: %s, for test: %v", err, test)
		}

		expectedConfig := test.expectedAPIConfig()
		if err == nil && !reflect.DeepEqual(actualDisablers, expectedConfig) {
			t.Fatalf("%v: unexpected apiResourceDisablers. Actual: %v\n expected: %v", test.runtimeConfig, actualDisablers, expectedConfig)
		}
	}

}

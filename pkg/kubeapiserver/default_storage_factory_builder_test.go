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

package kubeapiserver

import (
	"reflect"
	"testing"

	"k8s.io/api/core/v1"
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serverstore "k8s.io/apiserver/pkg/server/storage"
)

func TestMergeAPIResourceConfigs(t *testing.T) {
	registry := newFakeRegistry()
	extensionsGroupVersion := extensionsapiv1beta1.SchemeGroupVersion
	apiv1GroupVersion := v1.SchemeGroupVersion
	testCases := []struct {
		runtimeConfig         map[string]string
		defaultResourceConfig func() *serverstore.ResourceConfig
		expectedAPIConfig     func() *serverstore.ResourceConfig
		err                   bool
	}{
		{
			// everything default value.
			runtimeConfig: map[string]string{},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			err: false,
		},
		{
			// no runtimeConfig override.
			runtimeConfig: map[string]string{},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
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
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				return config
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				return config
			},
			err: false,
		},
		{
			// disable resource
			runtimeConfig: map[string]string{
				"/v1/pods": "false",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				return config
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableResources(apiv1GroupVersion.WithResource("pods"))
				return config
			},
			err: false,
		},
		{
			// Disable v1.
			runtimeConfig: map[string]string{
				"/v1": "false",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableVersions(apiv1GroupVersion)
				return config
			},
			err: false,
		},
		{
			// Enable deployments and disable daemonsets.
			runtimeConfig: map[string]string{
				"extensions/v1beta1/anything":   "true",
				"extensions/v1beta1/daemonsets": "false",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.EnableVersions(extensionsGroupVersion)
				return config
			},

			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.EnableVersions(extensionsGroupVersion)
				config.DisableResources(extensionsGroupVersion.WithResource("daemonsets"))
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
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			err: false,
		},
		{
			// cannot disable individual resource when version is not enabled.
			runtimeConfig: map[string]string{
				"/v1/pods": "false",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableVersions(apiv1GroupVersion)
				return config
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableResources(schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"})
				return config
			},
			err: true,
		},
		{
			// enable all
			runtimeConfig: map[string]string{
				"api/all": "true",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.EnableVersions(registry.RegisteredGroupVersions()...)
				return config
			},
			err: false,
		},
		{
			// disable all
			runtimeConfig: map[string]string{
				"api/all": "false",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableVersions(registry.RegisteredGroupVersions()...)
				return config
			},
			err: false,
		},
	}

	for index, test := range testCases {
		t.Log(registry.RegisteredGroupVersions())
		actualDisablers, err := MergeAPIResourceConfigs(test.defaultResourceConfig(), test.runtimeConfig, registry)
		if err == nil && test.err {
			t.Fatalf("expected error for test case: %v", index)
		} else if err != nil && !test.err {
			t.Fatalf("unexpected error: %s, for test: %v", err, test)
		}

		expectedConfig := test.expectedAPIConfig()
		if err == nil && !reflect.DeepEqual(actualDisablers, expectedConfig) {
			t.Fatalf("case %v: unexpected apiResourceDisablers. Actual: %v\n expected: %v", index, actualDisablers.GroupVersionResourceConfigs, expectedConfig.GroupVersionResourceConfigs)
		}
	}
}

func newFakeAPIResourceConfigSource() *serverstore.ResourceConfig {
	ret := serverstore.NewResourceConfig()
	// NOTE: GroupVersions listed here will be enabled by default. Don't put alpha versions in the list.
	ret.EnableVersions(
		v1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
	)

	return ret
}

func newFakeRegistry() *registered.APIRegistrationManager {
	registry := registered.NewOrDie("")

	registry.RegisterGroup(apimachinery.GroupMeta{
		GroupVersion:  v1.SchemeGroupVersion,
		GroupVersions: []schema.GroupVersion{v1.SchemeGroupVersion},
	})
	registry.RegisterGroup(apimachinery.GroupMeta{
		GroupVersion:  extensionsapiv1beta1.SchemeGroupVersion,
		GroupVersions: []schema.GroupVersion{extensionsapiv1beta1.SchemeGroupVersion},
	})
	registry.RegisterVersions([]schema.GroupVersion{v1.SchemeGroupVersion, extensionsapiv1beta1.SchemeGroupVersion})
	return registry
}

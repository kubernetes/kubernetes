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

package resourceconfig

import (
	"reflect"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	extensionsapiv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/apimachinery"
	"k8s.io/apimachinery/pkg/apimachinery/registered"
	"k8s.io/apimachinery/pkg/runtime/schema"
	serverstore "k8s.io/apiserver/pkg/server/storage"
)

func TestParseRuntimeConfig(t *testing.T) {
	registry := newFakeRegistry()
	apiv1GroupVersion := apiv1.SchemeGroupVersion
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
			// only enable v1
			runtimeConfig: map[string]string{
				"api/all": "false",
				"/v1":     "true",
			},
			defaultResourceConfig: func() *serverstore.ResourceConfig {
				return newFakeAPIResourceConfigSource()
			},
			expectedAPIConfig: func() *serverstore.ResourceConfig {
				config := newFakeAPIResourceConfigSource()
				config.DisableVersions(extensionsapiv1beta1.SchemeGroupVersion)
				return config
			},
			err: false,
		},
	}
	for index, test := range testCases {
		actualDisablers, err := MergeAPIResourceConfigs(test.defaultResourceConfig(), test.runtimeConfig, registry)
		if err == nil && test.err {
			t.Fatalf("expected error for test case: %v", index)
		} else if err != nil && !test.err {
			t.Fatalf("unexpected error: %s, for test: %v", err, test)
		}

		expectedConfig := test.expectedAPIConfig()
		if err == nil && !reflect.DeepEqual(actualDisablers, expectedConfig) {
			t.Fatalf("%v: unexpected apiResourceDisablers. Actual: %v\n expected: %v", test.runtimeConfig, actualDisablers, expectedConfig)
		}
	}
}

func newFakeAPIResourceConfigSource() *serverstore.ResourceConfig {
	ret := serverstore.NewResourceConfig()
	// NOTE: GroupVersions listed here will be enabled by default. Don't put alpha versions in the list.
	ret.EnableVersions(
		apiv1.SchemeGroupVersion,
		extensionsapiv1beta1.SchemeGroupVersion,
	)

	return ret
}

func newFakeRegistry() *registered.APIRegistrationManager {
	registry := registered.NewOrDie("")

	registry.RegisterGroup(apimachinery.GroupMeta{
		GroupVersion: apiv1.SchemeGroupVersion,
	})
	registry.RegisterGroup(apimachinery.GroupMeta{
		GroupVersion: extensionsapiv1beta1.SchemeGroupVersion,
	})
	registry.RegisterVersions([]schema.GroupVersion{apiv1.SchemeGroupVersion, extensionsapiv1beta1.SchemeGroupVersion})
	return registry
}

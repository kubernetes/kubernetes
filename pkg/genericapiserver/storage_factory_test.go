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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/genericapiserver/options"
	"k8s.io/kubernetes/pkg/storage/storagebackend"
)

func TestUpdateEtcdOverrides(t *testing.T) {
	testCases := []struct {
		resource unversioned.GroupResource
		servers  []string
	}{
		{
			resource: unversioned.GroupResource{Group: api.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000"},
		},
		{
			resource: unversioned.GroupResource{Group: api.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000", "http://127.0.0.1:20000"},
		},
		{
			resource: unversioned.GroupResource{Group: extensions.GroupName, Resource: "resource"},
			servers:  []string{"http://127.0.0.1:10000"},
		},
	}

	defaultEtcdLocation := []string{"http://127.0.0.1"}
	for i, test := range testCases {
		defaultConfig := storagebackend.Config{
			Prefix:     options.DefaultEtcdPathPrefix,
			ServerList: defaultEtcdLocation,
		}
		storageFactory := NewDefaultStorageFactory(defaultConfig, "", api.Codecs, NewDefaultResourceEncodingConfig(), NewResourceConfig())
		storageFactory.SetEtcdLocation(test.resource, test.servers)

		var err error
		config, err := storageFactory.NewConfig(test.resource)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.ServerList, test.servers) {
			t.Errorf("%d: expected %v, got %v", i, test.servers, config.ServerList)
			continue
		}

		config, err = storageFactory.NewConfig(unversioned.GroupResource{Group: api.GroupName, Resource: "unlikely"})
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(config.ServerList, defaultEtcdLocation) {
			t.Errorf("%d: expected %v, got %v", i, defaultEtcdLocation, config.ServerList)
			continue
		}

	}
}

func TestCohabitingSerializationVersion(t *testing.T) {
	defaultConfig := storagebackend.Config{
		Prefix:     options.DefaultEtcdPathPrefix,
		ServerList: []string{"http://127.0.0.1"},
	}

	testCases := []struct {
		name              string
		factory           func() *DefaultStorageFactory
		requestedResource unversioned.GroupResource

		expectedResource unversioned.GroupResource
	}{
		{
			name: "request first",
			factory: func() *DefaultStorageFactory {
				resourceConfig := NewResourceConfig()
				f := NewDefaultStorageFactory(defaultConfig, "", api.Codecs, NewDefaultResourceEncodingConfig(), resourceConfig)
				resourceConfig.EnableVersions(api.SchemeGroupVersion)
				f.AddCohabitatingResources(api.Resource("one"), api.Resource("two"))
				return f
			},
			requestedResource: api.Resource("one"),
			expectedResource:  api.Resource("one"),
		},
		{
			name: "request second",
			factory: func() *DefaultStorageFactory {
				resourceConfig := NewResourceConfig()
				f := NewDefaultStorageFactory(defaultConfig, "", api.Codecs, NewDefaultResourceEncodingConfig(), resourceConfig)
				resourceConfig.EnableVersions(api.SchemeGroupVersion)
				f.AddCohabitatingResources(api.Resource("one"), api.Resource("two"))
				return f
			},
			requestedResource: api.Resource("two"),
			expectedResource:  api.Resource("one"),
		},
		{
			name: "ignore second",
			factory: func() *DefaultStorageFactory {
				resourceConfig := NewResourceConfig()
				f := NewDefaultStorageFactory(defaultConfig, "", api.Codecs, NewDefaultResourceEncodingConfig(), resourceConfig)
				resourceConfig.EnableVersions(api.SchemeGroupVersion)
				f.AddCohabitatingResources(api.Resource("one"), api.Resource("two"), api.Resource("three"))
				f.IgnoreCohabitingStorageVersion(api.Resource("two"))
				return f
			},
			requestedResource: api.Resource("two"),
			expectedResource:  api.Resource("two"),
		},
		{
			name: "ignore second, request third",
			factory: func() *DefaultStorageFactory {
				resourceConfig := NewResourceConfig()
				f := NewDefaultStorageFactory(defaultConfig, "", api.Codecs, NewDefaultResourceEncodingConfig(), resourceConfig)
				resourceConfig.EnableVersions(api.SchemeGroupVersion)
				f.AddCohabitatingResources(api.Resource("one"), api.Resource("two"), api.Resource("three"))
				f.IgnoreCohabitingStorageVersion(api.Resource("two"))
				return f
			},
			requestedResource: api.Resource("three"),
			expectedResource:  api.Resource("one"),
		},
	}

	for _, tc := range testCases {
		factory := tc.factory()
		testEncodingConfig := &testDefaultResourceEncodingConfig{ResourceEncodingConfig: factory.ResourceEncodingConfig}
		factory.ResourceEncodingConfig = testEncodingConfig

		factory.New(tc.requestedResource)

		if e, a := tc.expectedResource, testEncodingConfig.requestedResource; e != a {
			t.Errorf("%s: expected %v, got %v", tc.name, e, a)
		}
	}
}

type testDefaultResourceEncodingConfig struct {
	ResourceEncodingConfig
	requestedResource unversioned.GroupResource
}

func (o *testDefaultResourceEncodingConfig) StorageEncodingFor(resource unversioned.GroupResource) (unversioned.GroupVersion, error) {
	o.requestedResource = resource
	return api.SchemeGroupVersion, nil
}

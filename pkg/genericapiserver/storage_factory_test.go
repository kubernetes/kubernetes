/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
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
		actualEtcdConfig := etcdstorage.EtcdConfig{}
		newEtcdFn := func(ns runtime.NegotiatedSerializer, storageVersion, memoryVersion unversioned.GroupVersion, etcdConfig etcdstorage.EtcdConfig) (etcdStorage storage.Interface, err error) {
			actualEtcdConfig = etcdConfig
			return nil, nil
		}

		defaultEtcdConfig := etcdstorage.EtcdConfig{
			Prefix:     DefaultEtcdPathPrefix,
			ServerList: defaultEtcdLocation,
		}
		storageFactory := NewDefaultStorageFactory(defaultEtcdConfig, api.Codecs, NewDefaultResourceEncodingConfig(), NewResourceConfig())
		storageFactory.newEtcdFn = newEtcdFn
		storageFactory.SetEtcdLocation(test.resource, test.servers)

		var err error
		_, err = storageFactory.New(test.resource)
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(actualEtcdConfig.ServerList, test.servers) {
			t.Errorf("%d: expected %v, got %v", i, test.servers, actualEtcdConfig.ServerList)
			continue
		}

		_, err = storageFactory.New(unversioned.GroupResource{Group: api.GroupName, Resource: "unlikely"})
		if err != nil {
			t.Errorf("%d: unexpected error %v", i, err)
			continue
		}
		if !reflect.DeepEqual(actualEtcdConfig.ServerList, defaultEtcdLocation) {
			t.Errorf("%d: expected %v, got %v", i, defaultEtcdLocation, actualEtcdConfig.ServerList)
			continue
		}

	}
}

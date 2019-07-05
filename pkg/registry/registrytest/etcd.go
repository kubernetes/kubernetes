/*
Copyright 2015 The Kubernetes Authors.

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

package registrytest

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/options"
	serverstorage "k8s.io/apiserver/pkg/server/storage"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/kubeapiserver"
)

func NewEtcdStorage(t *testing.T, group string) (*storagebackend.Config, *etcd3testing.EtcdTestServer) {
	server, config := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	config.Codec = testapi.Groups[group].StorageCodec()
	return config, server
}

func NewEtcdStorageForResource(t *testing.T, resource schema.GroupResource) (*storagebackend.Config, *etcd3testing.EtcdTestServer) {
	t.Helper()

	server, config := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)

	options := options.NewEtcdOptions(config)
	completedConfig, err := kubeapiserver.NewStorageFactoryConfig().Complete(options)
	if err != nil {
		t.Fatal(err)
	}
	completedConfig.ApiResourceConfig = serverstorage.NewResourceConfig()
	factory, err := completedConfig.New()
	if err != nil {
		t.Fatal(err)
	}
	resourceConfig, err := factory.NewConfig(resource)
	if err != nil {
		t.Fatal(err)
	}
	return resourceConfig, server
}

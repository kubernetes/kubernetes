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

package rest

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/server/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
)

func TestGetServersToValidate(t *testing.T) {
	servers := componentStatusStorage{fakeStorageFactory{}}.serversToValidate()

	if e, a := 3, len(servers); e != a {
		t.Errorf("expected %v, got %v", e, a)
	}

	for _, server := range []string{"scheduler", "controller-manager", "etcd-0"} {
		if _, ok := servers[server]; !ok {
			t.Errorf("server list missing: %s", server)
		}
	}
}

type fakeStorageFactory struct{}

func (f fakeStorageFactory) NewConfig(groupResource schema.GroupResource, example runtime.Object) (*storagebackend.ConfigForResource, error) {
	return nil, nil
}

func (f fakeStorageFactory) ResourcePrefix(groupResource schema.GroupResource) string {
	return ""
}

func (f fakeStorageFactory) Backends() []storage.Backend {
	return []storage.Backend{{Server: "etcd-0"}}
}

func (f fakeStorageFactory) Configs() []storagebackend.Config {
	return []storagebackend.Config{{Transport: storagebackend.TransportConfig{ServerList: []string{"etcd-0"}}}}
}

/*
Copyright 2014 Google Inc. All rights reserved.

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
package registry

import (
	"fmt"
	"testing"

	"github.com/coreos/go-etcd/etcd"
)

type EtcdResponseWithError struct {
	R *etcd.Response
	E error
}

type FakeEtcdClient struct {
	Data        map[string]EtcdResponseWithError
	deletedKeys []string
	Err         error
	t           *testing.T
	Ix          int
}

func MakeFakeEtcdClient(t *testing.T) *FakeEtcdClient {
	return &FakeEtcdClient{
		t:    t,
		Data: map[string]EtcdResponseWithError{},
	}
}

func (f *FakeEtcdClient) AddChild(key, data string, ttl uint64) (*etcd.Response, error) {
	f.Ix = f.Ix + 1
	return f.Set(fmt.Sprintf("%s/%d", key, f.Ix), data, ttl)
}

func (f *FakeEtcdClient) Get(key string, sort, recursive bool) (*etcd.Response, error) {
	result := f.Data[key]
	if result.R == nil {
		f.t.Errorf("Unexpected get for %s", key)
		return &etcd.Response{}, &etcd.EtcdError{ErrorCode: 100}
	}
	return result.R, result.E
}

func (f *FakeEtcdClient) Set(key, value string, ttl uint64) (*etcd.Response, error) {
	result := EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: value,
			},
		},
	}
	f.Data[key] = result
	return result.R, f.Err
}
func (f *FakeEtcdClient) Create(key, value string, ttl uint64) (*etcd.Response, error) {
	return f.Set(key, value, ttl)
}
func (f *FakeEtcdClient) Delete(key string, recursive bool) (*etcd.Response, error) {
	f.deletedKeys = append(f.deletedKeys, key)
	return &etcd.Response{}, f.Err
}

func (f *FakeEtcdClient) Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error) {
	return nil, fmt.Errorf("Unimplemented")
}

func MakeTestEtcdRegistry(client EtcdClient, machines []string) *EtcdRegistry {
	registry := MakeEtcdRegistry(client, machines)
	registry.manifestFactory = &BasicManifestFactory{
		serviceRegistry: &MockServiceRegistry{},
	}
	return registry
}

/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package election

import (
	"reflect"
	"testing"

	"github.com/coreos/go-etcd/etcd"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/watch"
)

func TestEtcdMasterOther(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	path := "foo"
	if _, err := server.Client.Set(path, "baz", 0); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	master := NewEtcdMasterElector(server.Client)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "baz" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
}

func TestEtcdMasterNoOther(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	path := "foo"
	master := NewEtcdMasterElector(server.Client)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "bar" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
}

// MockClient is wrapper aroung tools.EtcdClient.
type MockClient struct {
	client tools.EtcdClient
	t      *testing.T
	// afterGetFunc is called after each Get() call.
	afterGetFunc func()
	calls        []string
}

func (m *MockClient) GetCluster() []string {
	return m.client.GetCluster()
}

func (m *MockClient) Get(key string, sort, recursive bool) (*etcd.Response, error) {
	m.calls = append(m.calls, "get")
	defer m.afterGetFunc()
	response, err := m.client.Get(key, sort, recursive)
	return response, err
}

func (m *MockClient) Set(key, value string, ttl uint64) (*etcd.Response, error) {
	return m.client.Set(key, value, ttl)
}

func (m *MockClient) Create(key, value string, ttl uint64) (*etcd.Response, error) {
	m.calls = append(m.calls, "create")
	return m.client.Create(key, value, ttl)
}

func (m *MockClient) CompareAndSwap(key, value string, ttl uint64, prevValue string, prevIndex uint64) (*etcd.Response, error) {
	return m.client.CompareAndSwap(key, value, ttl, prevValue, prevIndex)
}

func (m *MockClient) Delete(key string, recursive bool) (*etcd.Response, error) {
	return m.client.Delete(key, recursive)
}

func (m *MockClient) Watch(prefix string, waitIndex uint64, recursive bool, receiver chan *etcd.Response, stop chan bool) (*etcd.Response, error) {
	return m.client.Watch(prefix, waitIndex, recursive, receiver, stop)
}

func TestEtcdMasterNoOtherThenConflict(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	// We set up the following scenario:
	// - after each Get() call, we write "baz" to a path
	// - this is simulating someone else writing a data
	// - the value written by someone else is the new value
	path := "foo"
	client := &MockClient{
		client: server.Client,
		t:      t,
		afterGetFunc: func() {
			if _, err := server.Client.Set(path, "baz", 0); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		},
		calls: make([]string, 0),
	}

	master := NewEtcdMasterElector(client)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "baz" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()

	expectedCalls := []string{"get", "create", "get"}
	if !reflect.DeepEqual(client.calls, expectedCalls) {
		t.Errorf("unexpected calls: %#v", client.calls)
	}
}

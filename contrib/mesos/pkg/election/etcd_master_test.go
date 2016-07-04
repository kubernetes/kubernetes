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

package election

import (
	"testing"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"

	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	"k8s.io/kubernetes/pkg/watch"
)

func TestEtcdMasterOther(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	path := "foo"
	keysAPI := etcd.NewKeysAPI(server.Client)
	if _, err := keysAPI.Set(context.TODO(), path, "baz", nil); err != nil {
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

func TestEtcdMasterNoOtherThenConflict(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	path := "foo"
	master := NewEtcdMasterElector(server.Client)
	leader := NewEtcdMasterElector(server.Client)

	w_ldr := leader.Elect(path, "baz")
	result := <-w_ldr.ResultChan()
	w := master.Elect(path, "bar")
	result = <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "baz" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
	w_ldr.Stop()
}

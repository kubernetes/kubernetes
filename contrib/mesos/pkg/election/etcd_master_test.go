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
	"testing"

	"github.com/coreos/go-etcd/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/watch"
)

func TestEtcdMasterOther(t *testing.T) {
	path := "foo"
	etcd := tools.NewFakeEtcdClient(t)
	etcd.Set(path, "baz", 0)
	master := NewEtcdMasterElector(etcd)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "baz" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
}

func TestEtcdMasterNoOther(t *testing.T) {
	path := "foo"
	e := tools.NewFakeEtcdClient(t)
	e.TestIndex = true
	e.Data["foo"] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: tools.EtcdErrorCodeNotFound,
		},
	}
	master := NewEtcdMasterElector(e)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "bar" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
}

func TestEtcdMasterNoOtherThenConflict(t *testing.T) {
	path := "foo"
	e := tools.NewFakeEtcdClient(t)
	e.TestIndex = true
	// Ok, so we set up a chain of responses from etcd:
	//   1) Nothing there
	//   2) conflict (someone else wrote)
	//   3) new value (the data they wrote)
	empty := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: tools.EtcdErrorCodeNotFound,
		},
	}
	empty.N = &tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: &etcd.EtcdError{
			ErrorCode: tools.EtcdErrorCodeNodeExist,
		},
	}
	empty.N.N = &tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: "baz",
			},
		},
	}
	e.Data["foo"] = empty
	master := NewEtcdMasterElector(e)
	w := master.Elect(path, "bar")
	result := <-w.ResultChan()
	if result.Type != watch.Modified || result.Object.(Master) != "bar" {
		t.Errorf("unexpected event: %#v", result)
	}
	w.Stop()
}

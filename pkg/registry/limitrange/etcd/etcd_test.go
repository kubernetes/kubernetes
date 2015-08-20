/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package etcd

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func newStorage(t *testing.T) (*REST, *tools.FakeEtcdClient) {
	etcdStorage, fakeClient := registrytest.NewEtcdStorage(t)
	return NewREST(etcdStorage), fakeClient
}

func TestLimitRangeCreate(t *testing.T) {
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
		Spec: api.LimitRangeSpec{
			Limits: []api.LimitRangeItem{
				{
					Type: api.LimitTypePod,
					Max: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("100"),
						api.ResourceMemory: resource.MustParse("10000"),
					},
					Min: api.ResourceList{
						api.ResourceCPU:    resource.MustParse("0"),
						api.ResourceMemory: resource.MustParse("100"),
					},
				},
			},
		},
	}

	nodeWithLimitRange := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), limitRange),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	ctx := api.NewDefaultContext()
	key := "foo"
	prefix := etcdtest.AddPrefix("limitranges")

	path, err := etcdgeneric.NamespaceKeyFunc(ctx, prefix, key)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toCreate runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: emptyNode,
			expect:   nodeWithLimitRange,
			toCreate: limitRange,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: nodeWithLimitRange,
			expect:   nodeWithLimitRange,
			toCreate: limitRange,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		storage, fakeClient := newStorage(t)
		fakeClient.Data[path] = item.existing
		_, err := storage.Create(ctx, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		received := fakeClient.Data[path]
		var limitRange api.LimitRange
		if err := testapi.Codec().DecodeInto([]byte(received.R.Node.Value), &limitRange); err != nil {
			t.Errorf("unexpected error: %v", err)
		}
		// Unset CreationTimestamp and UID which are set automatically by infrastructure.
		limitRange.ObjectMeta.CreationTimestamp = util.Time{}
		limitRange.ObjectMeta.UID = ""
		received.R.Node.Value = runtime.EncodeOrDie(testapi.Codec(), &limitRange)

		if e, a := item.expect, received; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

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

package limitrange

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	etcdstorage "github.com/GoogleCloudPlatform/kubernetes/pkg/storage/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestLimitRangeEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, generic.Registry) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	s := etcdstorage.NewEtcdStorage(f, testapi.Codec(), etcdtest.PathPrefix())
	return f, NewEtcdRegistry(s)
}

func TestLimitRangeCreate(t *testing.T) {
	limitRange := &api.LimitRange{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
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
		fakeClient, registry := NewTestLimitRangeEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.CreateWithName(ctx, key, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

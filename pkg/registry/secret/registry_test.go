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

package secret

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	etcdgeneric "github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic/etcd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestSecretEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, generic.Registry) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	h := tools.NewEtcdHelper(f, testapi.Codec(), etcdtest.PathPrefix())
	return f, NewEtcdRegistry(h)
}

func TestSecretCreate(t *testing.T) {
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      "abc",
			Namespace: "foo",
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
		},
	}

	nodeWithSecret := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), secret),
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
	path, err := etcdgeneric.NamespaceKeyFunc(ctx, "/secrets", key)
	path = etcdtest.AddPrefix(path)

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
			expect:   nodeWithSecret,
			toCreate: secret,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: nodeWithSecret,
			expect:   nodeWithSecret,
			toCreate: secret,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestSecretEtcdRegistry(t)
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

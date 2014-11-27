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

package event

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

var testTTL uint64 = 60

func NewTestEventEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, generic.Registry) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	h := tools.EtcdHelper{f, testapi.Codec(), tools.RuntimeVersionAdapter{testapi.MetadataAccessor()}}
	return f, NewEtcdRegistry(h, testTTL)
}

func TestEventCreate(t *testing.T) {
	eventA := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Reason:     "forTesting",
	}
	eventB := &api.Event{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Reason:     "forTesting",
	}

	nodeWithEventA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), eventA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
				TTL:           int64(testTTL),
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	path := "/registry/events/foo"
	key := "foo"

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toCreate runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: emptyNode,
			expect:   nodeWithEventA,
			toCreate: eventA,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: nodeWithEventA,
			expect:   nodeWithEventA,
			toCreate: eventB,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestEventEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.Create(api.NewContext(), key, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

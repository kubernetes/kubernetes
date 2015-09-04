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

package registrytest

import (
	"testing"

	"github.com/coreos/go-etcd/etcd"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/rest/resttest"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	etcdgeneric "k8s.io/kubernetes/pkg/registry/generic/etcd"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

func NewEtcdStorage(t *testing.T) (storage.Interface, *tools.FakeEtcdClient) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	etcdStorage := etcdstorage.NewEtcdStorage(fakeClient, testapi.Codec(), etcdtest.PathPrefix())
	return etcdStorage, fakeClient
}

type Tester struct {
	tester     *resttest.Tester
	fakeClient *tools.FakeEtcdClient
	storage    *etcdgeneric.Etcd
}
type UpdateFunc func(runtime.Object) runtime.Object

func New(t *testing.T, fakeClient *tools.FakeEtcdClient, storage *etcdgeneric.Etcd) *Tester {
	return &Tester{
		tester:     resttest.New(t, storage, fakeClient.SetError),
		fakeClient: fakeClient,
		storage:    storage,
	}
}

func (t *Tester) TestNamespace() string {
	return t.tester.TestNamespace()
}

func (t *Tester) ClusterScope() *Tester {
	t.tester = t.tester.ClusterScope()
	return t
}

func (t *Tester) AllowCreateOnUpdate() *Tester {
	t.tester = t.tester.AllowCreateOnUpdate()
	return t
}

func (t *Tester) GeneratesName() *Tester {
	t.tester = t.tester.GeneratesName()
	return t
}

func (t *Tester) ReturnDeletedObject() *Tester {
	t.tester = t.tester.ReturnDeletedObject()
	return t
}

func (t *Tester) TestCreate(valid runtime.Object, invalid ...runtime.Object) {
	t.tester.TestCreate(
		valid,
		t.setObject,
		t.getObject,
		invalid...,
	)
}

func (t *Tester) TestUpdate(valid runtime.Object, validUpdateFunc UpdateFunc, invalidUpdateFunc ...UpdateFunc) {
	var invalidFuncs []resttest.UpdateFunc
	for _, f := range invalidUpdateFunc {
		invalidFuncs = append(invalidFuncs, resttest.UpdateFunc(f))
	}
	t.tester.TestUpdate(
		valid,
		t.setObject,
		t.setResourceVersion,
		t.getObject,
		resttest.UpdateFunc(validUpdateFunc),
		invalidFuncs...,
	)
}

func (t *Tester) TestDelete(valid runtime.Object) {
	t.tester.TestDelete(
		valid,
		t.setObject,
		t.getObject,
		isNotFoundEtcdError,
	)
}

func (t *Tester) TestDeleteGraceful(valid runtime.Object, expectedGrace int64) {
	t.tester.TestDeleteGraceful(
		valid,
		t.setObject,
		t.getObject,
		expectedGrace,
	)
}

func (t *Tester) TestGet(valid runtime.Object) {
	t.tester.TestGet(valid)
}

func (t *Tester) TestList(valid runtime.Object) {
	t.tester.TestList(
		valid,
		t.setObjectsForList,
		t.setResourceVersion,
	)
}

func (t *Tester) TestWatch(valid runtime.Object, labelsPass, labelsFail []labels.Set, fieldsPass, fieldsFail []fields.Set) {
	t.tester.TestWatch(
		valid,
		t.fakeClient.WaitForWatchCompletion,
		t.injectWatchError,
		t.emitObject,
		labelsPass,
		labelsFail,
		fieldsPass,
		fieldsFail,
		[]string{etcdstorage.EtcdCreate, etcdstorage.EtcdSet, etcdstorage.EtcdCAS, etcdstorage.EtcdDelete},
	)
}

// =============================================================================
// Helper functions

func (t *Tester) getObject(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return nil, err
	}
	key, err := t.storage.KeyFunc(ctx, meta.Name)
	if err != nil {
		return nil, err
	}
	key = etcdtest.AddPrefix(key)
	resp, err := t.fakeClient.Get(key, false, false)
	if err != nil {
		return nil, err
	}
	result := t.storage.NewFunc()
	if err := testapi.Codec().DecodeInto([]byte(resp.Node.Value), result); err != nil {
		return nil, err
	}
	return result, nil
}

func (t *Tester) setObject(ctx api.Context, obj runtime.Object) error {
	meta, err := api.ObjectMetaFor(obj)
	if err != nil {
		return err
	}
	key, err := t.storage.KeyFunc(ctx, meta.Name)
	if err != nil {
		return err
	}
	key = etcdtest.AddPrefix(key)
	_, err = t.fakeClient.Set(key, runtime.EncodeOrDie(testapi.Codec(), obj), 0)
	return err
}

func (t *Tester) setObjectsForList(objects []runtime.Object) []runtime.Object {
	result := make([]runtime.Object, len(objects))
	key := etcdtest.AddPrefix(t.storage.KeyRootFunc(t.tester.TestContext()))

	if len(objects) > 0 {
		nodes := make([]*etcd.Node, len(objects))
		for i, obj := range objects {
			encoded := runtime.EncodeOrDie(testapi.Codec(), obj)
			decoded, _ := testapi.Codec().Decode([]byte(encoded))
			nodes[i] = &etcd.Node{Value: encoded}
			result[i] = decoded
		}
		t.fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{
				Node: &etcd.Node{
					Nodes: nodes,
				},
			},
			E: nil,
		}
	} else {
		t.fakeClient.Data[key] = tools.EtcdResponseWithError{
			R: &etcd.Response{},
			E: t.fakeClient.NewError(tools.EtcdErrorCodeNotFound),
		}
	}
	return result
}

func (t *Tester) setResourceVersion(resourceVersion uint64) {
	t.fakeClient.ChangeIndex = resourceVersion
}

func (t *Tester) injectWatchError(err error) {
	t.fakeClient.WatchInjectError <- err
}

func (t *Tester) emitObject(obj runtime.Object, action string) error {
	encoded, err := testapi.Codec().Encode(obj)
	if err != nil {
		return err
	}
	node := &etcd.Node{
		Value: string(encoded),
	}
	var prevNode *etcd.Node = nil
	if action == etcdstorage.EtcdDelete {
		prevNode = node
	}
	t.fakeClient.WatchResponse <- &etcd.Response{
		Action:   action,
		Node:     node,
		PrevNode: prevNode,
	}
	return nil
}

func isNotFoundEtcdError(err error) bool {
	etcdError, ok := err.(*etcd.EtcdError)
	if !ok {
		return false
	}
	return etcdError.ErrorCode == tools.EtcdErrorCodeNotFound
}

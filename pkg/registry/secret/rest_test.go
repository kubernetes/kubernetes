/*
Copyright 2015 Google Inc. All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/rest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

type testRegistry struct {
	*registrytest.GenericRegistry
}

func NewTestREST() (testRegistry, *REST) {
	reg := testRegistry{registrytest.NewGeneric(nil)}
	return reg, NewREST(reg)
}

func testSecret(name string) *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      name,
			Namespace: "default",
		},
		Data: map[string][]byte{
			"data-1": []byte("value-1"),
		},
		Type: api.SecretTypeOpaque,
	}
}

func TestRESTCreate(t *testing.T) {
	table := []struct {
		ctx    api.Context
		secret *api.Secret
		valid  bool
	}{
		{
			ctx:    api.NewDefaultContext(),
			secret: testSecret("foo"),
			valid:  true,
		}, {
			ctx:    api.NewContext(),
			secret: testSecret("bar"),
			valid:  false,
		}, {
			ctx:    api.WithNamespace(api.NewContext(), "nondefault"),
			secret: testSecret("bazzzz"),
			valid:  false,
		},
	}

	for _, item := range table {
		_, storage := NewTestREST()
		c, err := storage.Create(item.ctx, item.secret)
		if !item.valid {
			if err == nil {
				ctxNS := api.NamespaceValue(item.ctx)
				t.Errorf("%v: Unexpected non-error: (%v, %v)", item.secret.Name, ctxNS, item.secret.Namespace)
			}
			continue
		}
		if err != nil {
			t.Errorf("%v: Unexpected error: %v", item.secret.Name, err)
			continue
		}
		if !api.HasObjectMetaSystemFieldValues(&item.secret.ObjectMeta) {
			t.Errorf("storage did not populate object meta field values")
		}
		if e, a := item.secret, c; !reflect.DeepEqual(e, a) {
			t.Errorf("diff: %s", util.ObjectDiff(e, a))
		}
		// Ensure we implement the interface
		_ = rest.Watcher(storage)
	}
}

func TestRESTUpdate(t *testing.T) {
	ctx := api.NewDefaultContext()
	registry, rest := NewTestREST()
	registry.CreateWithName(ctx, "foo", testSecret("foo"))
	modifiedSecret := testSecret("foo")
	modifiedSecret.Data = map[string][]byte{
		"data-2": []byte("value-2"),
	}

	updatedObj, created, err := rest.Update(ctx, modifiedSecret)
	if err != nil {
		t.Fatalf("Expected no error: %v", err)
	}
	if updatedObj == nil {
		t.Errorf("Expected non-nil object")
	}
	if created {
		t.Errorf("expected not created")
	}
	updatedSecret := updatedObj.(*api.Secret)
	if updatedSecret.Name != "foo" {
		t.Errorf("Expected foo, but got %v", updatedSecret.Name)
	}
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	secretA := testSecret("foo")
	_, err := rest.Create(api.NewDefaultContext(), secretA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	c, err := rest.Delete(api.NewDefaultContext(), secretA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if stat := c.(*api.Status); stat.Status != api.StatusSuccess {
		t.Errorf("unexpected status: %v", stat)
	}
}

func TestRESTGet(t *testing.T) {
	_, rest := NewTestREST()
	secretA := testSecret("foo")
	_, err := rest.Create(api.NewDefaultContext(), secretA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	got, err := rest.Get(api.NewDefaultContext(), secretA.Name)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := secretA, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTgetAttrs(t *testing.T) {
	_, rest := NewTestREST()
	secretA := testSecret("foo")
	label, field, err := rest.getAttrs(secretA)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := label, (labels.Set{}); !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
	expect := fields.Set{
		"type": string(api.SecretTypeOpaque),
	}
	if e, a := expect, field; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTList(t *testing.T) {
	reg, rest := NewTestREST()

	var (
		secretA = testSecret("a")
		secretB = testSecret("b")
		secretC = testSecret("c")
	)

	reg.ObjectList = &api.SecretList{
		Items: []api.Secret{*secretA, *secretB, *secretC},
	}
	got, err := rest.List(api.NewContext(), labels.Everything(), fields.Everything())
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	expect := &api.SecretList{
		Items: []api.Secret{*secretA, *secretB, *secretC},
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

func TestRESTWatch(t *testing.T) {
	secretA := testSecret("a")
	reg, rest := NewTestREST()
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), fields.Everything(), "0")
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, secretA)
	}()
	got := <-wi.ResultChan()
	if e, a := secretA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

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

package resourcequota

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

func makeRegistry(resourceList runtime.Object) (*registrytest.GenericRegistry, *REST) {
	registry := registrytest.NewGeneric(resourceList)
	rest := NewREST(registry)
	return registry, rest
}

func TestGet(t *testing.T) {
	registry, rest := makeRegistry(&api.ResourceQuotaList{})
	registry.Object = &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
	}
	ctx := api.NewDefaultContext()
	obj, err := rest.Get(ctx, "foo")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if obj == nil {
		t.Errorf("unexpected nil object")
	}
	registry.Object = nil
	registry.Err = errors.NewNotFound("ResourceQuota", "bar")

	obj, err = rest.Get(ctx, "bar")
	if err == nil {
		t.Errorf("unexpected non-error")
	}
	if obj != nil {
		t.Errorf("unexpected object: %v", obj)
	}

}

func TestList(t *testing.T) {
	_, rest := makeRegistry(&api.ResourceQuotaList{
		Items: []api.ResourceQuota{
			{
				ObjectMeta: api.ObjectMeta{
					Name:      "foo",
					Namespace: api.NamespaceDefault,
				},
			},
		},
	})

	ctx := api.NewDefaultContext()
	obj, err := rest.List(ctx, labels.Set{}.AsSelector(), fields.Set{}.AsSelector())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if obj == nil {
		t.Errorf("unexpected nil object")
	}
	list, ok := obj.(*api.ResourceQuotaList)
	if !ok || len(list.Items) != 1 {
		t.Errorf("unexpected list object: %v", obj)
	}

	obj, err = rest.List(ctx, labels.Set{"foo": "bar"}.AsSelector(), fields.Set{}.AsSelector())
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if obj == nil {
		t.Errorf("unexpected nil object")
	}
	list, ok = obj.(*api.ResourceQuotaList)
	if !ok || len(list.Items) != 0 {
		t.Errorf("unexpected list object: %v", obj)
	}
}

func TestUpdate(t *testing.T) {
	registry, rest := makeRegistry(&api.ResourceQuotaList{})
	resourceStatus := api.ResourceQuotaStatus{
		Hard: api.ResourceList{
			api.ResourceCPU: *resource.NewQuantity(10.0, resource.BinarySI),
		},
	}
	registry.Object = &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"bar": "foo",
			},
		},
		Status: resourceStatus,
	}
	invalidUpdates := []struct {
		obj runtime.Object
		err error
	}{
		{&api.Pod{}, nil},
		{&api.ResourceQuota{ObjectMeta: api.ObjectMeta{Namespace: "$%#%"}}, nil},
		{&api.ResourceQuota{
			ObjectMeta: api.ObjectMeta{
				Namespace: api.NamespaceDefault,
			},
		}, fmt.Errorf("test error")},
	}
	for _, test := range invalidUpdates {
		registry.Err = test.err
		ctx := api.NewDefaultContext()
		_, _, err := rest.Update(ctx, test.obj)
		if err == nil {
			t.Errorf("unexpected non-error for: %v", test.obj)
		}
		registry.Err = nil
	}

	ctx := api.NewDefaultContext()
	update := &api.ResourceQuota{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
			Labels: map[string]string{
				"foo": "bar",
			},
		},
		Spec: api.ResourceQuotaSpec{
			Hard: api.ResourceList{
				api.ResourceCPU: *resource.NewQuantity(10.0, resource.BinarySI),
			},
		},
		Status: api.ResourceQuotaStatus{
			Hard: api.ResourceList{
				api.ResourceCPU: *resource.NewQuantity(20.0, resource.BinarySI),
			},
		},
	}
	obj, _, err := rest.Update(ctx, update)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(obj.(*api.ResourceQuota).Labels, update.Labels) {
		t.Errorf("unexpected update object, labels don't match: %v vs %v", obj.(*api.ResourceQuota).Labels, update.Labels)
	}
	if !reflect.DeepEqual(obj.(*api.ResourceQuota).Spec, update.Spec) {
		t.Errorf("unexpected update object, specs don't match: %v vs %v", obj.(*api.ResourceQuota).Spec, update.Spec)
	}
	if !reflect.DeepEqual(obj.(*api.ResourceQuota).Status, registry.Object.(*api.ResourceQuota).Status) {
		t.Errorf("unexpected update object, status wasn't preserved: %v vs %v", obj.(*api.ResourceQuota).Status, registry.Object.(*api.ResourceQuota).Status)
	}
}

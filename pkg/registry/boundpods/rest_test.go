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

package boundpods

import (
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/registrytest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/watch"
)

func NewTestREST() (*registrytest.BoundPodsRegistry, *REST) {
	reg := &registrytest.BoundPodsRegistry{}
	return reg, NewREST(reg)
}

func TestRESTDelete(t *testing.T) {
	_, rest := NewTestREST()
	_, err := rest.Delete(api.NewContext(), "foo")
	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestRESTGet(t *testing.T) {
	reg, rest := NewTestREST()
	reg.BoundPods = []api.BoundPods{
		{
			Host: "test",
			Items: []api.BoundPod{
				{
					ObjectMeta: api.ObjectMeta{
						Name: "foo",
					},
				},
			},
		},
	}
	pods, err := rest.Get(api.NewContext(), "test")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pods.(*api.BoundPods).Host != "test" {
		t.Errorf("unexpected bound pods: %#v", pods)
	}

	reg.Err = errors.NewNotFound("BoundPods", "other")
	pods, err = rest.Get(api.NewContext(), "other")
	if !errors.IsNotFound(err) {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestRESTUpdate(t *testing.T) {
	_, rest := NewTestREST()
	_, err := rest.Update(api.NewContext(), &api.BoundPods{})
	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestRESTList(t *testing.T) {
	_, rest := NewTestREST()
	_, err := rest.List(api.NewContext(), labels.Everything(), labels.Set{"status": "tested"}.AsSelector())
	if err == nil {
		t.Fatalf("unexpected non-error")
	}
}

func TestRESTWatch(t *testing.T) {
	pods := &api.BoundPods{
		Host: "test",
		Items: []api.BoundPod{
			{
				ObjectMeta: api.ObjectMeta{
					Name: "foo",
				},
			},
		},
	}
	reg, rest := NewTestREST()
	reg.Broadcaster = watch.NewBroadcaster(0)
	wi, err := rest.Watch(api.NewContext(), labels.Everything(), labels.Everything(), "0")
	if err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	go func() {
		reg.Broadcaster.Action(watch.Added, pods)
	}()
	got := <-wi.ResultChan()
	if e, a := pods, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("diff: %s", util.ObjectDiff(e, a))
	}
}

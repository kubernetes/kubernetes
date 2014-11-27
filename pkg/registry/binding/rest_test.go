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

package binding

import (
	"errors"
	"net/http"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
)

func TestNewREST(t *testing.T) {
	mockRegistry := MockRegistry{
		OnApplyBinding: func(b *api.Binding) error { return nil },
	}
	b := NewREST(mockRegistry)

	binding := &api.Binding{
		PodID: "foo",
		Host:  "bar",
	}
	body, err := latest.Codec.Encode(binding)
	if err != nil {
		t.Fatalf("Unexpected encode error %v", err)
	}
	obj := b.New()
	err = latest.Codec.DecodeInto(body, obj)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := binding, obj; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, but got %#v", e, a)
	}
}

func TestRESTUnsupported(t *testing.T) {
	var ctx api.Context
	mockRegistry := MockRegistry{
		OnApplyBinding: func(b *api.Binding) error { return nil },
	}
	b := NewREST(mockRegistry)
	if _, err := b.Delete(ctx, "binding id"); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := b.Update(ctx, &api.Binding{PodID: "foo", Host: "new machine"}); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := b.Get(ctx, "binding id"); err == nil {
		t.Errorf("unexpected non-error")
	}
	if _, err := b.List(ctx, labels.Set{"name": "foo"}.AsSelector(), labels.Everything()); err == nil {
		t.Errorf("unexpected non-error")
	}
	// Try sending wrong object just to get 100% coverage
	if _, err := b.Create(ctx, &api.Pod{}); err == nil {
		t.Errorf("unexpected non-error")
	}
}

func TestRESTPost(t *testing.T) {
	table := []struct {
		b   *api.Binding
		err error
	}{
		{b: &api.Binding{PodID: "foo", Host: "bar"}, err: errors.New("no host bar")},
		{b: &api.Binding{PodID: "baz", Host: "qux"}, err: nil},
		{b: &api.Binding{PodID: "dvorak", Host: "qwerty"}, err: nil},
	}

	for i, item := range table {
		mockRegistry := MockRegistry{
			OnApplyBinding: func(b *api.Binding) error {
				if !reflect.DeepEqual(item.b, b) {
					t.Errorf("%v: expected %#v, but got %#v", i, item, b)
				}
				return item.err
			},
		}
		ctx := api.NewContext()
		b := NewREST(mockRegistry)
		resultChan, err := b.Create(ctx, item.b)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		var expect *api.Status
		if item.err == nil {
			expect = &api.Status{Status: api.StatusSuccess}
		} else {
			expect = &api.Status{
				Status:  api.StatusFailure,
				Code:    http.StatusInternalServerError,
				Message: item.err.Error(),
			}
		}
		if e, a := expect, (<-resultChan).Object; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: expected %#v, got %#v", i, e, a)
		}
	}
}

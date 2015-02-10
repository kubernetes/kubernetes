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
		result, err := b.Create(ctx, item.b)
		if err != nil && item.err == nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		if err == nil && item.err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}
		var expect interface{}
		if item.err == nil {
			expect = &api.Status{Status: api.StatusSuccess, Code: http.StatusCreated}
		}
		if e, a := expect, result; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: expected %#v, got %#v", i, e, a)
		}
	}
}

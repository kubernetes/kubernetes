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
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func TestBindingStorage_Extract(t *testing.T) {
	b := &BindingStorage{}

	binding := &api.Binding{
		PodID: "foo",
		Host:  "bar",
	}
	body, err := api.Encode(binding)
	if err != nil {
		t.Fatalf("Unexpected encode error %v", err)
	}
	obj := b.New()
	err = api.DecodeInto(body, obj)
	if err != nil {
		t.Fatalf("Unexpected error %v", err)
	}
	if e, a := binding, obj; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, but got %#v", e, a)
	}
}

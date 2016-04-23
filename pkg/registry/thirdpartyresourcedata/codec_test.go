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

package thirdpartyresourcedata

import (
	"encoding/json"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/runtime"
)

type Foo struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata,omitempty" description:"standard object metadata"`

	SomeField  string `json:"someField"`
	OtherField int    `json:"otherField"`
}

type FooList struct {
	unversioned.TypeMeta `json:",inline"`
	unversioned.ListMeta `json:"metadata,omitempty" description:"standard list metadata; see http://releases.k8s.io/HEAD/docs/devel/api-conventions.md#metadata"`

	Items []Foo `json:"items"`
}

func TestCodec(t *testing.T) {
	tests := []struct {
		obj       *Foo
		expectErr bool
		name      string
	}{
		{
			obj:       &Foo{ObjectMeta: api.ObjectMeta{Name: "bar"}},
			expectErr: true,
			name:      "missing kind",
		},
		{
			obj:  &Foo{ObjectMeta: api.ObjectMeta{Name: "bar"}, TypeMeta: unversioned.TypeMeta{Kind: "Foo"}},
			name: "basic",
		},
		{
			obj:  &Foo{ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "baz"}, TypeMeta: unversioned.TypeMeta{Kind: "Foo"}},
			name: "resource version",
		},
		{
			obj: &Foo{
				ObjectMeta: api.ObjectMeta{
					Name:              "bar",
					CreationTimestamp: unversioned.Time{Time: time.Unix(100, 0)},
				},
				TypeMeta: unversioned.TypeMeta{Kind: "Foo"},
			},
			name: "creation time",
		},
		{
			obj: &Foo{
				ObjectMeta: api.ObjectMeta{
					Name:            "bar",
					ResourceVersion: "baz",
					Labels:          map[string]string{"foo": "bar", "baz": "blah"},
				},
				TypeMeta: unversioned.TypeMeta{Kind: "Foo"},
			},
			name: "labels",
		},
	}
	for _, test := range tests {
		d := &thirdPartyResourceDataDecoder{kind: "Foo", delegate: testapi.Extensions.Codec()}
		e := &thirdPartyResourceDataEncoder{kind: "Foo", delegate: testapi.Extensions.Codec()}
		data, err := json.Marshal(test.obj)
		if err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}
		obj, err := runtime.Decode(d, data)
		if err != nil && !test.expectErr {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}
		if test.expectErr {
			if err == nil {
				t.Errorf("[%s] unexpected non-error", test.name)
			}
			continue
		}
		rsrcObj, ok := obj.(*extensions.ThirdPartyResourceData)
		if !ok {
			t.Errorf("[%s] unexpected object: %v", test.name, obj)
			continue
		}
		if !reflect.DeepEqual(rsrcObj.ObjectMeta, test.obj.ObjectMeta) {
			t.Errorf("[%s]\nexpected\n%v\nsaw\n%v\n", test.name, rsrcObj.ObjectMeta, test.obj.ObjectMeta)
		}
		var output Foo
		if err := json.Unmarshal(rsrcObj.Data, &output); err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}
		if !reflect.DeepEqual(&output, test.obj) {
			t.Errorf("[%s]\nexpected\n%v\nsaw\n%v\n", test.name, test.obj, &output)
		}

		data, err = runtime.Encode(e, rsrcObj)
		if err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
		}

		var output2 Foo
		if err := json.Unmarshal(data, &output2); err != nil {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
			continue
		}
		if !reflect.DeepEqual(&output2, test.obj) {
			t.Errorf("[%s]\nexpected\n%v\nsaw\n%v\n", test.name, test.obj, &output2)
		}
	}
}

func TestCreater(t *testing.T) {
	creater := NewObjectCreator("creater group", "creater version", api.Scheme)
	tests := []struct {
		name        string
		kind        unversioned.GroupVersionKind
		expectedObj runtime.Object
		expectErr   bool
	}{
		{
			name:        "valid ThirdPartyResourceData creation",
			kind:        unversioned.GroupVersionKind{Group: "creater group", Version: "creater version", Kind: "ThirdPartyResourceData"},
			expectedObj: &extensions.ThirdPartyResourceData{},
			expectErr:   false,
		},
		{
			name:        "invalid ThirdPartyResourceData creation",
			kind:        unversioned.GroupVersionKind{Version: "invalid version", Kind: "ThirdPartyResourceData"},
			expectedObj: nil,
			expectErr:   true,
		},
		{
			name:        "valid ListOptions creation",
			kind:        unversioned.GroupVersionKind{Version: "v1", Kind: "ListOptions"},
			expectedObj: &v1.ListOptions{},
			expectErr:   false,
		},
	}
	for _, test := range tests {
		out, err := creater.New(test.kind)
		if err != nil && !test.expectErr {
			t.Errorf("[%s] unexpected error: %v", test.name, err)
		}
		if err == nil && test.expectErr {
			t.Errorf("[%s] unexpected non-error", test.name)
		}
		if !reflect.DeepEqual(test.expectedObj, out) {
			t.Errorf("[%s] unexpected error: expect: %v, got: %v", test.name, test.expectedObj, out)
		}

	}
}

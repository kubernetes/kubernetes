/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package api

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/runtime"
)

type FakeAPIObject struct{}

func (*FakeAPIObject) IsAnAPIObject() {}

type ExtensionAPIObject struct {
	unversioned.TypeMeta
	ObjectMeta
}

func (*ExtensionAPIObject) IsAnAPIObject() {}

func TestGetReference(t *testing.T) {
	table := map[string]struct {
		obj       runtime.Object
		ref       *ObjectReference
		fieldPath string
		shouldErr bool
	}{
		"pod": {
			obj: &Pod{
				ObjectMeta: ObjectMeta{
					Name:            "foo",
					UID:             "bar",
					ResourceVersion: "42",
					SelfLink:        "/api/version1/pods/foo",
				},
			},
			fieldPath: ".desiredState.containers[0]",
			ref: &ObjectReference{
				Kind:            "Pod",
				APIVersion:      "version1",
				Name:            "foo",
				UID:             "bar",
				ResourceVersion: "42",
				FieldPath:       ".desiredState.containers[0]",
			},
		},
		"serviceList": {
			obj: &ServiceList{
				ListMeta: unversioned.ListMeta{
					ResourceVersion: "42",
					SelfLink:        "/api/version2/services",
				},
			},
			ref: &ObjectReference{
				Kind:            "ServiceList",
				APIVersion:      "version2",
				ResourceVersion: "42",
			},
		},
		"extensionAPIObject": {
			obj: &ExtensionAPIObject{
				TypeMeta: unversioned.TypeMeta{
					Kind: "ExtensionAPIObject",
				},
				ObjectMeta: ObjectMeta{
					Name:            "foo",
					UID:             "bar",
					ResourceVersion: "42",
					SelfLink:        "/custom_prefix/version1/extensions/foo",
				},
			},
			ref: &ObjectReference{
				Kind:            "ExtensionAPIObject",
				APIVersion:      "version1",
				Name:            "foo",
				UID:             "bar",
				ResourceVersion: "42",
			},
		},
		"badSelfLink": {
			obj: &ServiceList{
				ListMeta: unversioned.ListMeta{
					ResourceVersion: "42",
					SelfLink:        "version2/services",
				},
			},
			shouldErr: true,
		},
		"error": {
			obj:       &FakeAPIObject{},
			ref:       nil,
			shouldErr: true,
		},
		"errorNil": {
			obj:       nil,
			ref:       nil,
			shouldErr: true,
		},
	}

	for name, item := range table {
		ref, err := GetPartialReference(item.obj, item.fieldPath)
		if e, a := item.shouldErr, (err != nil); e != a {
			t.Errorf("%v: expected %v, got %v, err %v", name, e, a, err)
			continue
		}
		if e, a := item.ref, ref; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: expected %#v, got %#v", name, e, a)
		}
	}
}

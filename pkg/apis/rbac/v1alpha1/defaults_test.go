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

package v1alpha1_test

import (
	"reflect"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	_ "k8s.io/kubernetes/pkg/api/install"
	"k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/rbac/install"
	"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/runtime"
)

func TestSetDefaultProtectedAttribute(t *testing.T) {
	original := &v1alpha1.ProtectedAttribute{
		ObjectMeta: v1.ObjectMeta{
			Namespace: "ns1",
		},
		RoleRef: v1.ObjectReference{
			Kind:      "Role",
			Namespace: "",
		},
	}

	obj, err := roundTrip(t, original)
	if err != nil {
		t.Fatalf("roundTrip: %s", err)
	}

	got, ok := obj.(*v1alpha1.ProtectedAttribute)
	if !ok {
		t.Fatalf("Wrong type after roundTrip: %T", obj)
	}
	if ns := got.RoleRef.Namespace; ns != "ns1" {
		t.Fatalf("Expected role namespace to be defaulted to object namespace, got %q", ns)
	}
}

func roundTrip(t *testing.T, obj runtime.Object) (runtime.Object, error) {
	encodedObj, err := runtime.Encode(api.Codecs.LegacyCodec(v1alpha1.SchemeGroupVersion), obj)
	if err != nil {
		return nil, err
	}
	decodedObj, err := runtime.Decode(api.Codecs.UniversalDecoder(), encodedObj)
	if err != nil {
		return nil, err
	}

	res := reflect.New(reflect.TypeOf(obj).Elem()).Interface().(runtime.Object)
	err = api.Scheme.Convert(decodedObj, res)
	if err != nil {
		return nil, err
	}
	return res, nil
}

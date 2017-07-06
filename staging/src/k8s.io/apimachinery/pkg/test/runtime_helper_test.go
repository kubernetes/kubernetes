/*
Copyright 2014 The Kubernetes Authors.

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

package test

import (
	"testing"

	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	"k8s.io/apimachinery/pkg/runtime"
)

func TestDecodeList(t *testing.T) {
	pl := List{
		Items: []runtime.Object{
			&testapigroup.Carp{ObjectMeta: metav1.ObjectMeta{Name: "1"}},
			&runtime.Unknown{
				TypeMeta:    runtime.TypeMeta{Kind: "Carp", APIVersion: "v1"},
				Raw:         []byte(`{"kind":"Carp","apiVersion":"` + "v1" + `","metadata":{"name":"test"}}`),
				ContentType: runtime.ContentTypeJSON,
			},
		},
	}

	_, codecs := TestScheme()
	Codec := apitesting.TestCodec(codecs, testapigroup.SchemeGroupVersion)

	if errs := runtime.DecodeList(pl.Items, Codec); len(errs) != 0 {
		t.Fatalf("unexpected error %v", errs)
	}
	if pod, ok := pl.Items[1].(*testapigroup.Carp); !ok || pod.Name != "test" {
		t.Errorf("object not converted: %#v", pl.Items[1])
	}
}

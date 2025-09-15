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

package testing

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/recognizer"
)

type A struct{}

func (A) GetObjectKind() schema.ObjectKind { return schema.EmptyObjectKind }
func (a A) DeepCopyObject() runtime.Object {
	return a
}

func TestRecognizer(t *testing.T) {
	s := runtime.NewScheme()
	s.AddKnownTypes(schema.GroupVersion{Version: "v1"}, &A{})
	d := recognizer.NewDecoder(
		json.NewSerializerWithOptions(json.DefaultMetaFactory, s, s, json.SerializerOptions{}),
		json.NewSerializerWithOptions(json.DefaultMetaFactory, s, s, json.SerializerOptions{Yaml: true}),
	)
	out, _, err := d.Decode([]byte(`
kind: A
apiVersion: v1
`), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%#v", out)

	out, _, err = d.Decode([]byte(`
{
  "kind":"A",
  "apiVersion":"v1"
}
`), nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("%#v", out)
}

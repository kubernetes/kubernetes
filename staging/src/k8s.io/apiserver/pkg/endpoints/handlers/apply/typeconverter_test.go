/*
Copyright 2018 The Kubernetes Authors.

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

package apply_test

import (
	"path/filepath"
	"reflect"
	"testing"

	"github.com/ghodss/yaml"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apiserver/pkg/endpoints/handlers/apply"
	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
)

var fakeSchema = prototesting.Fake{
	Path: filepath.Join(
		"..", "..", "..", "..", "..", "..", "..", "..",
		"api", "openapi-spec", "swagger.json"),
}

func TestTypeConverter(t *testing.T) {
	d, err := fakeSchema.OpenAPISchema()
	if err != nil {
		t.Fatalf("Failed to parse OpenAPI schema: %v", err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		t.Fatalf("Failed to build OpenAPI models: %v", err)
	}

	tc, err := apply.NewTypeConverter(m)
	if err != nil {
		t.Fatalf("Failed to build TypeConverter: %v", err)
	}

	y := `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
  replicas: 3
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.15.4
`

	obj := &unstructured.Unstructured{Object: map[string]interface{}{}}
	if err := yaml.Unmarshal([]byte(y), &obj.Object); err != nil {
		t.Fatalf("Failed to parse yaml object: %v", err)
	}
	typed, err := tc.ObjectToTyped(obj)
	if err != nil {
		t.Fatalf("Failed to convert object to typed: %v", err)
	}
	newObj, err := tc.TypedToObject(typed)
	if err != nil {
		t.Fatalf("Failed to convert typed to object: %v", err)
	}
	if !reflect.DeepEqual(obj, newObj) {
		t.Errorf(`Round-trip failed:
Original object:
%#v
Final object:
%#v`, obj, newObj)
	}
}

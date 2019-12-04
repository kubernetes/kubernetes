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

package internal_test

import (
	"fmt"
	"path/filepath"
	"reflect"
	"testing"

	"sigs.k8s.io/structured-merge-diff/typed"
	"sigs.k8s.io/yaml"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/fieldmanager/internal"
	corev1 "k8s.io/api/core/v1"
	appsv1 "k8s.io/api/apps/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"

	"k8s.io/kube-openapi/pkg/util/proto"
	prototesting "k8s.io/kube-openapi/pkg/util/proto/testing"
)

var fakeSchema = prototesting.Fake{
	Path: filepath.Join("testdata", "swagger.json"),
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

	tc, err := internal.NewTypeConverter(m, false)
	if err != nil {
		t.Fatalf("Failed to build TypeConverter: %v", err)
	}

	dtc := internal.DeducedTypeConverter{}

	testCases := []struct {
		name string
		gvk schema.GroupVersionKind
		yaml string
	}{
		{
			name: "apps/v1.Deployment",
			gvk: appsv1.SchemeGroupVersion.WithKind("Deployment"),
			yaml: `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
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
`,
		}, {
			name: "extensions/v1beta1.Deployment",
			gvk: extensionsv1beta1.SchemeGroupVersion.WithKind("Deployment"),
			yaml: `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
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
`,
		}, {
			name: "v1.Pod",
			gvk: corev1.SchemeGroupVersion.WithKind("Pod"),
			yaml: `
apiVersion: v1
kind: Pod
metadata:
  name: nginx-pod
  labels:
    app: nginx
spec:
  containers:
  - name: nginx
    image: nginx:1.15.4
`,
		},
	}

	for _, testCase := range testCases {
		t.Run(fmt.Sprintf("%v ObjectToTyped with TypeConverter", testCase.name), func(t *testing.T) {
			testObjectToTyped(t, tc, testCase.yaml)
		})
		t.Run(fmt.Sprintf("%v ObjectToTyped with DeducedTypeConverter", testCase.name), func(t *testing.T) {
			testObjectToTyped(t, dtc, testCase.yaml)
		})
		t.Run(fmt.Sprintf("%v ObjectToTypedReflect with DeducedTypeConverter", testCase.name), func(t *testing.T) {
			testObjectToTypedReflect(t, dtc, testCase.yaml, testCase.gvk)
		})
	}
}

func testObjectToTyped(t *testing.T, tc internal.TypeConverter, y string) {
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

func testObjectToTypedReflect(t *testing.T, tc internal.TypeConverter, y string, gvk schema.GroupVersionKind) {
	scheme := runtime.NewScheme()
	if err :=corev1.AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add to scheme: %v", err)
	}
	if err :=appsv1.AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add to scheme: %v", err)
	}
	if err :=extensionsv1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("Failed to add to scheme: %v", err)
	}

	obj, err := scheme.New(gvk)
	if err != nil {
		t.Fatalf("Failed to construct object: %v", err)
	}
	if err := yaml.Unmarshal([]byte(y), &obj); err != nil {
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

	// convert back to typed object before comparing
	newBytes, err := yaml.Marshal(newObj)
	if err != nil {
		t.Fatalf("Failed to marshal object: %v", err)
	}
	newTyped, err:= scheme.New(gvk)
	if err != nil {
		t.Fatalf("Failed to construct object: %v", err)
	}
	if err := yaml.Unmarshal([]byte(newBytes), &newTyped); err != nil {
		t.Fatalf("Failed to parse yaml object: %v", err)
	}

	if !reflect.DeepEqual(obj, newTyped) {
		t.Errorf(`Round-trip failed:
Original object:
%#v
Final object:
%#v`, obj, newTyped)
	}
}

var result typed.TypedValue

func BenchmarkObjectToTyped(b *testing.B) {
	y := `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  name: nginx-deployment
  labels:
    app: nginx
spec:
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
		b.Fatalf("Failed to parse yaml object: %v", err)
	}

	d, err := fakeSchema.OpenAPISchema()
	if err != nil {
		b.Fatalf("Failed to parse OpenAPI schema: %v", err)
	}
	m, err := proto.NewOpenAPIData(d)
	if err != nil {
		b.Fatalf("Failed to build OpenAPI models: %v", err)
	}

	tc, err := internal.NewTypeConverter(m, false)
	if err != nil {
		b.Fatalf("Failed to build TypeConverter: %v", err)
	}

	b.ResetTimer()
	b.ReportAllocs()

	var r *typed.TypedValue
	for i := 0; i < b.N; i++ {
		var err error
		r, err = tc.ObjectToTyped(obj)
		if err != nil {
			b.Fatalf("Failed to convert object to typed: %v", err)
		}
	}
	result = *r
}

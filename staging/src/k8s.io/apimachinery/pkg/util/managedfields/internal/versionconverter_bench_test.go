/*
Copyright The Kubernetes Authors.

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

package internal

import (
	"fmt"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/structured-merge-diff/v6/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v6/merge"
)

// benchDeployment is a structured (reflect-backed) object whose JSON shape is a
// subset of the apps Deployment schema. Built-in types reach the version
// converter as reflect-backed typed values, so the pre-fast-path Convert had to
// deep-copy the whole object via TypedToObject just to read its apiVersion;
// this type reproduces that cost.
type benchDeployment struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
}

func (d *benchDeployment) DeepCopyObject() runtime.Object {
	out := &benchDeployment{TypeMeta: d.TypeMeta}
	d.ObjectMeta.DeepCopyInto(&out.ObjectMeta)
	return out
}

// structuredDeployment returns a reflect-backed Deployment with n labels and n
// annotations, exercising the built-in-type code path.
func structuredDeployment(apiVersion string, n int) runtime.Object {
	labels, annotations := metaMaps(n)
	return &benchDeployment{
		TypeMeta: metav1.TypeMeta{APIVersion: apiVersion, Kind: "Deployment"},
		ObjectMeta: metav1.ObjectMeta{
			Name:        "nginx-deployment",
			Namespace:   "default",
			Labels:      labels,
			Annotations: annotations,
		},
	}
}

// unstructuredDeployment returns a value-backed Deployment, exercising the CRD
// code path (the typed value already holds a map, so TypedToObject is cheap).
func unstructuredDeployment(apiVersion string, n int) runtime.Object {
	labels, annotations := metaMaps(n)
	return &unstructured.Unstructured{Object: map[string]interface{}{
		"apiVersion": apiVersion,
		"kind":       "Deployment",
		"metadata": map[string]interface{}{
			"name":        "nginx-deployment",
			"namespace":   "default",
			"labels":      toInterfaceMap(labels),
			"annotations": toInterfaceMap(annotations),
		},
	}}
}

func metaMaps(n int) (labels, annotations map[string]string) {
	labels = make(map[string]string, n)
	annotations = make(map[string]string, n)
	for i := range n {
		labels[fmt.Sprintf("k8s.io/label-%d", i)] = fmt.Sprintf("value-%d", i)
		annotations[fmt.Sprintf("k8s.io/annotation-%d", i)] = fmt.Sprintf("value-%d", i)
	}
	return labels, annotations
}

func toInterfaceMap(m map[string]string) map[string]interface{} {
	out := make(map[string]interface{}, len(m))
	for k, v := range m {
		out[k] = v
	}
	return out
}

func benchVersionConverter() merge.Converter {
	oc := fakeObjectConvertorForTestSchema{
		gvkForVersion("v1beta1"): objForGroupVersion("apps/v1beta1"),
		gvkForVersion("v1"):      objForGroupVersion("apps/v1"),
	}
	return newVersionConverter(testTypeConverter, oc, schema.GroupVersion{Group: "apps", Version: runtime.APIVersionInternal})
}

var benchBackings = []struct {
	name  string
	build func(apiVersion string, n int) runtime.Object
}{
	{"structured", structuredDeployment},
	{"unstructured", unstructuredDeployment},
}

var benchConversions = []struct {
	name    string
	version fieldpath.APIVersion
}{
	{"same-version", "apps/v1beta1"},
	{"cross-version", "apps/v1"},
}

var benchSizes = []int{0, 10, 100, 1000}

// BenchmarkVersionConverter measures versionConverter.Convert across reflect-backed
// (built-in) and value-backed (CRD) inputs, for the same-version (fast path) and
// cross-version (fallback) cases, over a range of object sizes.
func BenchmarkVersionConverter(b *testing.B) {
	vc := benchVersionConverter()
	for _, backing := range benchBackings {
		for _, conv := range benchConversions {
			for _, n := range benchSizes {
				input, err := testTypeConverter.ObjectToTyped(backing.build("apps/v1beta1", n))
				if err != nil {
					b.Fatalf("ObjectToTyped(%s, fields=%d): %v", backing.name, n, err)
				}
				b.Run(fmt.Sprintf("%s/%s/fields=%d", backing.name, conv.name, n), func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						if _, err := vc.Convert(input, conv.version); err != nil {
							b.Fatalf("Convert: %v", err)
						}
					}
				})
			}
		}
	}
}

// TestBenchmarkFastPathEngages guards the benchmark: it confirms the same-version
// case takes the fast path (no TypedToObject / ConvertToVersion) for both
// reflect-backed and value-backed inputs, and that the cross-version case does
// not, so the before/after comparison measures the intended code paths.
func TestBenchmarkFastPathEngages(t *testing.T) {
	for _, backing := range benchBackings {
		t.Run(backing.name, func(t *testing.T) {
			input, err := testTypeConverter.ObjectToTyped(backing.build("apps/v1beta1", 10))
			if err != nil {
				t.Fatalf("ObjectToTyped: %v", err)
			}

			// Same version: fast path, no materialization or conversion.
			tc := &countingTypeConverter{TypeConverter: testTypeConverter}
			oc := &countingObjectConvertor{ObjectConvertor: benchObjectConvertor()}
			vc := newVersionConverter(tc, oc, schema.GroupVersion{Group: "apps", Version: runtime.APIVersionInternal})
			out, err := vc.Convert(input, fieldpath.APIVersion("apps/v1beta1"))
			if err != nil {
				t.Fatalf("Convert same-version: %v", err)
			}
			if out != input {
				t.Errorf("same-version Convert should return input unchanged")
			}
			if tc.typedToObjectCalls != 0 {
				t.Errorf("same-version Convert called TypedToObject %d times, want 0", tc.typedToObjectCalls)
			}
			if oc.convertToVersionCalls != 0 {
				t.Errorf("same-version Convert called ConvertToVersion %d times, want 0", oc.convertToVersionCalls)
			}

			// Cross version: fallback path materializes the object.
			tc = &countingTypeConverter{TypeConverter: testTypeConverter}
			oc = &countingObjectConvertor{ObjectConvertor: benchObjectConvertor()}
			vc = newVersionConverter(tc, oc, schema.GroupVersion{Group: "apps", Version: runtime.APIVersionInternal})
			if _, err := vc.Convert(input, fieldpath.APIVersion("apps/v1")); err != nil {
				t.Fatalf("Convert cross-version: %v", err)
			}
			if tc.typedToObjectCalls == 0 {
				t.Errorf("cross-version Convert should call TypedToObject")
			}
		})
	}
}

func benchObjectConvertor() fakeObjectConvertorForTestSchema {
	return fakeObjectConvertorForTestSchema{
		gvkForVersion("v1beta1"): objForGroupVersion("apps/v1beta1"),
		gvkForVersion("v1"):      objForGroupVersion("apps/v1"),
	}
}

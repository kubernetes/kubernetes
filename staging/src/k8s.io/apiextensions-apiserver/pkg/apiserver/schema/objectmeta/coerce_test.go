/*
Copyright 2019 The Kubernetes Authors.

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

package objectmeta

import (
	"math/rand"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/apitesting/fuzzer"
	"k8s.io/apimachinery/pkg/api/equality"
	metafuzzer "k8s.io/apimachinery/pkg/apis/meta/fuzzer"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
)

func TestRoundtripObjectMeta(t *testing.T) {
	scheme := runtime.NewScheme()
	codecs := serializer.NewCodecFactory(scheme)
	codec := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, json.SerializerOptions{})
	seed := rand.Int63()
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(seed), codecs)

	N := 1000
	for i := 0; i < N; i++ {
		u := &unstructured.Unstructured{Object: map[string]interface{}{}}
		original := &metav1.ObjectMeta{}
		fuzzer.Fill(original)
		if err := SetObjectMeta(u.Object, original); err != nil {
			t.Fatalf("unexpected error setting ObjectMeta: %v", err)
		}
		o, _, err := GetObjectMeta(u.Object, false)
		if err != nil {
			t.Fatalf("unexpected error getting the Objectmeta: %v", err)
		}

		if !equality.Semantic.DeepEqual(original, o) {
			t.Errorf("diff: %v\nCodec: %#v", cmp.Diff(original, o), codec)
		}
	}
}

// TestMalformedObjectMetaFields sets a number of different random values and types for all
// metadata fields. If json.Unmarshal accepts them, compare that getObjectMeta
// gives the same result. Otherwise, drop malformed fields.
func TestMalformedObjectMetaFields(t *testing.T) {
	fuzzer := fuzzer.FuzzerFor(metafuzzer.Funcs, rand.NewSource(rand.Int63()), serializer.NewCodecFactory(runtime.NewScheme()))
	spuriousValues := func() []interface{} {
		return []interface{}{
			// primitives
			nil,
			int64(1),
			float64(1.5),
			true,
			"a",
			// well-formed complex values
			[]interface{}{"a", "b"},
			map[string]interface{}{"a": "1", "b": "2"},
			[]interface{}{int64(1), int64(2)},
			[]interface{}{float64(1.5), float64(2.5)},
			// known things json decoding tolerates
			map[string]interface{}{"a": "1", "b": nil},
			// malformed things
			map[string]interface{}{"a": "1", "b": []interface{}{"nested"}},
			[]interface{}{"a", int64(1), float64(1.5), true, []interface{}{"nested"}},
		}
	}
	N := 100
	for i := 0; i < N; i++ {
		fuzzedObjectMeta := &metav1.ObjectMeta{}
		fuzzer.Fill(fuzzedObjectMeta)
		goodMetaMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(fuzzedObjectMeta.DeepCopy())
		if err != nil {
			t.Fatal(err)
		}
		for _, pth := range jsonPaths(nil, goodMetaMap) {
			for _, v := range spuriousValues() {
				// skip values of same type, because they can only cause decoding errors further insides
				orig, err := JSONPathValue(goodMetaMap, pth, 0)
				if err != nil {
					t.Fatalf("unexpected to not find something at %v: %v", pth, err)
				}
				if reflect.TypeOf(v) == reflect.TypeOf(orig) {
					continue
				}

				// make a spurious map
				spuriousMetaMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(fuzzedObjectMeta.DeepCopy())
				if err != nil {
					t.Fatal(err)
				}
				if err := SetJSONPath(spuriousMetaMap, pth, 0, v); err != nil {
					t.Fatal(err)
				}

				// See if it can unmarshal to object meta
				spuriousJSON, err := utiljson.Marshal(spuriousMetaMap)
				if err != nil {
					t.Fatalf("error on %v=%#v: %v", pth, v, err)
				}
				expectedObjectMeta := &metav1.ObjectMeta{}
				if err := utiljson.Unmarshal(spuriousJSON, expectedObjectMeta); err != nil {
					// if standard json unmarshal would fail decoding this field, drop the field entirely
					truncatedMetaMap, err := runtime.DefaultUnstructuredConverter.ToUnstructured(fuzzedObjectMeta.DeepCopy())
					if err != nil {
						t.Fatal(err)
					}

					// we expect this logic for the different fields:
					switch {
					default:
						// delete complete top-level field by default
						DeleteJSONPath(truncatedMetaMap, pth[:1], 0)
					}

					truncatedJSON, err := utiljson.Marshal(truncatedMetaMap)
					if err != nil {
						t.Fatalf("error on %v=%#v: %v", pth, v, err)
					}
					expectedObjectMeta = &metav1.ObjectMeta{}
					if err := utiljson.Unmarshal(truncatedJSON, expectedObjectMeta); err != nil {
						t.Fatalf("error on %v=%#v: %v", pth, v, err)
					}
				}

				// make sure dropInvalidTypedFields+getObjectMeta matches what we expect
				u := &unstructured.Unstructured{Object: map[string]interface{}{"metadata": spuriousMetaMap}}
				actualObjectMeta, _, err := GetObjectMeta(u.Object, true)
				if err != nil {
					t.Errorf("got unexpected error after dropping invalid typed fields on %v=%#v: %v", pth, v, err)
					continue
				}

				if !equality.Semantic.DeepEqual(expectedObjectMeta, actualObjectMeta) {
					t.Errorf("%v=%#v, diff: %v\n", pth, v, cmp.Diff(expectedObjectMeta, actualObjectMeta))
					t.Errorf("expectedObjectMeta %#v", expectedObjectMeta)
				}
			}
		}
	}
}

func TestGetObjectMetaWithOptions(t *testing.T) {
	unknownAndMalformed := map[string]interface{}{
		"kind":       "Pod",
		"apiVersion": "v1",
		"metadata": map[string]interface{}{
			"name":         "my-meta",
			"unknownField": "foo",
			"generateName": nil,
			"generation":   nil,
			"labels": map[string]string{
				"foo": "bar",
			},
			"annotations": 11,
		},
	}

	unknownOnly := map[string]interface{}{
		"kind":       "Pod",
		"apiVersion": "v1",
		"metadata": map[string]interface{}{
			"name":         "my-meta",
			"unknownField": "foo",
			"generateName": nil,
			"generation":   nil,
			"labels": map[string]string{
				"foo": "bar",
			},
		},
	}

	malformedOnly := map[string]interface{}{
		"kind":       "Pod",
		"apiVersion": "v1",
		"metadata": map[string]interface{}{
			"name":         "my-meta",
			"generateName": nil,
			"generation":   nil,
			"labels": map[string]string{
				"foo": "bar",
			},
			"annotations": 11,
		},
	}

	var testcases = []struct {
		obj                     map[string]interface{}
		dropMalformedFields     bool
		returnUnknownFieldPaths bool
		expectedObject          *metav1.ObjectMeta
		expectedUnknownPaths    []string
		expectedErr             string
	}{
		{
			obj:                     unknownAndMalformed,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: false,
			expectedErr:             "json: cannot unmarshal number into Go struct field ObjectMeta.annotations of type map[string]string",
		},
		{
			obj:                     unknownAndMalformed,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: false,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
		{
			obj:                     unknownAndMalformed,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: true,
			expectedErr:             "json: cannot unmarshal number into Go struct field ObjectMeta.annotations of type map[string]string",
		},
		{
			obj:                     unknownAndMalformed,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: true,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			expectedUnknownPaths: []string{"metadata.unknownField"},
		},

		{
			obj:                     unknownOnly,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: false,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
		{
			obj:                     unknownOnly,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: false,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
		{
			obj:                     unknownOnly,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: true,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			expectedUnknownPaths: []string{"metadata.unknownField"},
		},
		{
			obj:                     unknownOnly,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: true,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
			expectedUnknownPaths: []string{"metadata.unknownField"},
		},

		{
			obj:                     malformedOnly,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: false,
			expectedErr:             "json: cannot unmarshal number into Go struct field ObjectMeta.annotations of type map[string]string",
		},
		{
			obj:                     malformedOnly,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: false,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
		{
			obj:                     malformedOnly,
			dropMalformedFields:     false,
			returnUnknownFieldPaths: true,
			expectedErr:             "json: cannot unmarshal number into Go struct field ObjectMeta.annotations of type map[string]string",
		},
		{
			obj:                     malformedOnly,
			dropMalformedFields:     true,
			returnUnknownFieldPaths: true,
			expectedObject: &metav1.ObjectMeta{
				Name: "my-meta",
				Labels: map[string]string{
					"foo": "bar",
				},
			},
		},
	}
	for _, tc := range testcases {
		opts := ObjectMetaOptions{
			ReturnUnknownFieldPaths: tc.returnUnknownFieldPaths,
			DropMalformedFields:     tc.dropMalformedFields,
		}
		obj, _, unknownPaths, err := GetObjectMetaWithOptions(tc.obj, opts)
		if !reflect.DeepEqual(tc.expectedObject, obj) {
			t.Errorf("expected: %v, got: %v", tc.expectedObject, obj)
		}
		if (err == nil && tc.expectedErr != "") || err != nil && (err.Error() != tc.expectedErr) {
			t.Errorf("expected: %v, got: %v", tc.expectedErr, err)
		}
		if !reflect.DeepEqual(tc.expectedUnknownPaths, unknownPaths) {
			t.Errorf("expected: %v, got: %v", tc.expectedUnknownPaths, unknownPaths)
		}
	}
}

func TestGetObjectMetaNils(t *testing.T) {
	u := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "Pod",
			"apiVersion": "v1",
			"metadata": map[string]interface{}{
				"generateName": nil,
				"generation":   nil,
				"labels": map[string]interface{}{
					"foo": nil,
				},
			},
		},
	}

	o, _, err := GetObjectMeta(u.Object, true)
	if err != nil {
		t.Fatal(err)
	}
	if o.GenerateName != "" {
		t.Errorf("expected null json generateName value to be read as \"\" string, but got: %q", o.GenerateName)
	}
	if o.Generation != 0 {
		t.Errorf("expected null json generation value to be read as zero, but got: %q", o.Generation)
	}
	if got, expected := o.Labels, map[string]string{"foo": ""}; !reflect.DeepEqual(got, expected) {
		t.Errorf("unexpected labels, expected=%#v, got=%#v", expected, got)
	}

	// double check this what the kube JSON decode is doing
	bs, _ := utiljson.Marshal(u.UnstructuredContent())
	kubeObj, _, err := clientgoscheme.Codecs.UniversalDecoder(corev1.SchemeGroupVersion).Decode(bs, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	pod, ok := kubeObj.(*corev1.Pod)
	if !ok {
		t.Fatalf("expected v1 Pod, got: %T", pod)
	}
	if got, expected := o.GenerateName, pod.ObjectMeta.GenerateName; got != expected {
		t.Errorf("expected generatedName to be %q, got %q", expected, got)
	}
	if got, expected := o.Generation, pod.ObjectMeta.Generation; got != expected {
		t.Errorf("expected generation to be %q, got %q", expected, got)
	}
	if got, expected := o.Labels, pod.ObjectMeta.Labels; !reflect.DeepEqual(got, expected) {
		t.Errorf("expected labels to be %v, got %v", expected, got)
	}
}

func TestGetObjectMeta(t *testing.T) {
	for i := 0; i < 100; i++ {
		u := &unstructured.Unstructured{Object: map[string]interface{}{
			"metadata": map[string]interface{}{
				"name": "good",
				"Name": "bad1",
				"nAme": "bad2",
				"naMe": "bad3",
				"namE": "bad4",

				"namespace": "good",
				"Namespace": "bad1",
				"nAmespace": "bad2",
				"naMespace": "bad3",
				"namEspace": "bad4",

				"creationTimestamp": "a",
			},
		}}

		meta, _, err := GetObjectMeta(u.Object, true)
		if err != nil {
			t.Fatal(err)
		}
		if meta.Name != "good" || meta.Namespace != "good" {
			t.Fatalf("got %#v", meta)
		}
	}
}

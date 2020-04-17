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

package unstructured_test

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/apis/testapigroup"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/test"
)

func TestObjectToUnstructuredConversion(t *testing.T) {
	scheme, _ := test.TestScheme()
	testCases := []struct {
		name                          string
		objectToConvert               runtime.Object
		expectedErr                   error
		expectedConvertedUnstructured *unstructured.Unstructured
	}{
		{
			name:                          "convert nil object to unstructured should fail",
			objectToConvert:               nil,
			expectedErr:                   fmt.Errorf("unable to convert object type <nil> to Unstructured, must be a runtime.Object"),
			expectedConvertedUnstructured: &unstructured.Unstructured{},
		},
		{
			name:            "convert versioned empty object to unstructured should work",
			objectToConvert: &testapigroupv1.Carp{},
			expectedConvertedUnstructured: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
					},
					"spec":   map[string]interface{}{},
					"status": map[string]interface{}{},
				},
			},
		},
		{
			name: "convert valid versioned object to unstructured should work",
			objectToConvert: &testapigroupv1.Carp{
				ObjectMeta: metav1.ObjectMeta{
					Name: "noxu",
				},
				Spec: testapigroupv1.CarpSpec{
					Hostname: "example.com",
				},
			},
			expectedConvertedUnstructured: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
		},
		{
			name:            "convert hub-versioned object to unstructured should fail",
			objectToConvert: &testapigroup.Carp{},
			expectedErr:     fmt.Errorf("unable to convert the internal object type *testapigroup.Carp to Unstructured without providing a preferred version to convert to"),
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			outUnstructured := &unstructured.Unstructured{}
			err := scheme.Convert(testCase.objectToConvert, outUnstructured, nil)
			if err != nil {
				assert.Equal(t, testCase.expectedErr, err)
				return
			}
			assert.Equal(t, testCase.expectedConvertedUnstructured, outUnstructured)
		})
	}
}

func TestUnstructuredToObjectConversion(t *testing.T) {
	scheme, _ := test.TestScheme()
	testCases := []struct {
		name                    string
		unstructuredToConvert   *unstructured.Unstructured
		convertingObject        runtime.Object
		expectPanic             bool
		expectedErrFunc         func(err error) bool
		expectedConvertedObject runtime.Object
	}{
		{
			name: "convert empty unstructured w/o gvk to versioned object should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{},
			},
			convertingObject: &testapigroupv1.Carp{},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewMissingKindErr("unstructured object has no kind"))
			},
		},
		{
			name: "convert empty versioned unstructured to versioned object should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
				},
			},
			convertingObject:        &testapigroupv1.Carp{},
			expectedConvertedObject: &testapigroupv1.Carp{},
		},
		{
			name: "convert empty unstructured w/o gvk to versioned object should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{},
			},
			convertingObject: &testapigroupv1.Carp{},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewMissingKindErr("unstructured object has no kind"))
			},
		},
		{
			name: "convert valid versioned unstructured to versioned object should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			convertingObject: &testapigroupv1.Carp{},
			expectedConvertedObject: &testapigroupv1.Carp{
				ObjectMeta: metav1.ObjectMeta{
					Name: "noxu",
				},
				Spec: testapigroupv1.CarpSpec{
					Hostname: "example.com",
				},
			},
		},
		{
			name: "convert valid versioned unstructured to hub-versioned object should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			convertingObject: &testapigroup.Carp{},
			expectedConvertedObject: &testapigroup.Carp{
				ObjectMeta: metav1.ObjectMeta{
					Name: "noxu",
				},
				Spec: testapigroup.CarpSpec{
					Hostname: "example.com",
				},
			},
		},
		{
			name: "convert unexisting-versioned unstructured to hub-versioned object should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v9",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			convertingObject: &testapigroup.Carp{},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewNotRegisteredGVKErrForTarget(
					scheme.Name(),
					schema.GroupVersionKind{Group: "", Version: "v9", Kind: "Carp"},
					nil,
				))
			},
		},
		{
			name: "convert valid versioned unstructured to object w/ a mismatching kind should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			convertingObject: &metav1.CreateOptions{},
			expectedErrFunc: func(err error) bool {
				return strings.HasPrefix(err.Error(), "converting (v1.Carp) to (v1.CreateOptions):")
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				v := recover()
				assert.Equal(t, testCase.expectPanic, v != nil, "unexpected panic")
			}()
			outObject := testCase.convertingObject.DeepCopyObject()
			// Convert by specifying destination object
			err := scheme.Convert(testCase.unstructuredToConvert, outObject, nil)
			if err != nil {
				if testCase.expectedErrFunc != nil {
					if !testCase.expectedErrFunc(err) {
						t.Errorf("error mismatched: %v", err)
					}
				}
				return
			}
			assert.Equal(t, testCase.expectedConvertedObject, outObject)
		})
	}
}

func TestUnstructuredToGVConversion(t *testing.T) {
	scheme, _ := test.TestScheme()
	// HACK: registering fake internal/v1beta1 api
	scheme.AddKnownTypes(schema.GroupVersion{Group: "foo", Version: "v1beta1"}, &testapigroup.Carp{})
	scheme.AddKnownTypes(schema.GroupVersion{Group: "foo", Version: "__internal"}, &testapigroup.Carp{})

	testCases := []struct {
		name                    string
		unstructuredToConvert   *unstructured.Unstructured
		targetGV                schema.GroupVersion
		expectPanic             bool
		expectedErrFunc         func(err error) bool
		expectedConvertedObject runtime.Object
	}{
		{
			name: "convert versioned unstructured to valid external version should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
				},
			},
			targetGV: schema.GroupVersion{Group: "", Version: "v1"},
			expectedConvertedObject: &testapigroupv1.Carp{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Carp",
				},
			},
		},
		{
			name: "convert hub-versioned unstructured to hub version should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "__internal",
					"kind":       "Carp",
				},
			},
			targetGV:                schema.GroupVersion{Group: "", Version: "__internal"},
			expectedConvertedObject: &testapigroup.Carp{},
		},
		{
			name: "convert empty unstructured w/o gvk to versioned should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{},
			},
			targetGV: schema.GroupVersion{Group: "", Version: "v1"},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewMissingKindErr("unstructured object has no kind"))
			},
			expectedConvertedObject: nil,
		},
		{
			name: "convert versioned unstructured to mismatching external version should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
				},
			},
			targetGV: schema.GroupVersion{Group: "foo", Version: "v1beta1"},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewNotRegisteredErrForTarget(
					scheme.Name(), reflect.TypeOf(testapigroupv1.Carp{}), schema.GroupVersion{Group: "foo", Version: "v1beta1"}))
			},
			expectedConvertedObject: nil,
		},
		{
			name: "convert versioned unstructured to mismatching internal version should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
				},
			},
			targetGV: schema.GroupVersion{Group: "foo", Version: "__internal"},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewNotRegisteredErrForTarget(
					scheme.Name(), reflect.TypeOf(testapigroupv1.Carp{}), schema.GroupVersion{Group: "foo", Version: "__internal"}))
			},
			expectedConvertedObject: nil,
		},
		{
			name: "convert valid versioned unstructured to its own version should work",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			targetGV: schema.GroupVersion{Group: "", Version: "v1"},
			expectedConvertedObject: &testapigroupv1.Carp{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
					Kind:       "Carp",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "noxu",
				},
				Spec: testapigroupv1.CarpSpec{
					Hostname: "example.com",
				},
			},
		},
		{
			name: "convert valid versioned unstructured to hub-version should work ignoring type meta",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			targetGV: schema.GroupVersion{Group: "", Version: "__internal"},
			expectedConvertedObject: &testapigroup.Carp{
				ObjectMeta: metav1.ObjectMeta{
					Name: "noxu",
				},
				Spec: testapigroup.CarpSpec{
					Hostname: "example.com",
				},
			},
		},
		{
			name: "convert valid versioned unstructured to unexisting version should fail",
			unstructuredToConvert: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Carp",
					"metadata": map[string]interface{}{
						"creationTimestamp": nil,
						"name":              "noxu",
					},
					"spec": map[string]interface{}{
						"hostname": "example.com",
					},
					"status": map[string]interface{}{},
				},
			},
			targetGV: schema.GroupVersion{Group: "", Version: "v9"},
			expectedErrFunc: func(err error) bool {
				return reflect.DeepEqual(err, runtime.NewNotRegisteredGVKErrForTarget(
					scheme.Name(),
					schema.GroupVersionKind{Group: "", Version: "v9", Kind: "Carp"},
					nil,
				))
			},
			expectedConvertedObject: nil,
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			defer func() {
				v := recover()
				assert.Equal(t, testCase.expectPanic, v != nil, "unexpected panic")
			}()
			// Convert by specifying destination object
			outObject, err := scheme.ConvertToVersion(testCase.unstructuredToConvert, testCase.targetGV)
			if testCase.expectedErrFunc != nil {
				if !testCase.expectedErrFunc(err) {
					t.Errorf("error mismatched: %v", err)
				}
			}
			assert.Equal(t, testCase.expectedConvertedObject, outObject)
		})
	}
}

func TestUnstructuredToUnstructuredConversion(t *testing.T) {
	// eventually, we don't want any inter-unstructured conversion happen, but for now, the conversion
	// just copy/pastes
	scheme, _ := test.TestScheme()
	inUnstructured := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"apiVersion": "v1",
			"kind":       "Carp",
		},
	}
	outUnstructured := &unstructured.Unstructured{}
	err := scheme.Convert(inUnstructured, outUnstructured, nil)
	assert.NoError(t, err)
	assert.Equal(t, inUnstructured, outUnstructured)
}

func benchmarkCarp() *testapigroupv1.Carp {
	t := metav1.Date(2015, 1, 1, 12, 0, 0, 0, time.UTC)
	return &testapigroupv1.Carp{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "name",
			Namespace: "namespace",
		},
		Spec: testapigroupv1.CarpSpec{
			RestartPolicy: "restart",
			NodeSelector: map[string]string{
				"label1": "value1",
				"label2": "value2",
			},
			ServiceAccountName: "service-account",
			HostNetwork:        false,
			HostPID:            true,
			Subdomain:          "hostname.subdomain.namespace.svc.domain",
		},
		Status: testapigroupv1.CarpStatus{
			Phase: "phase",
			Conditions: []testapigroupv1.CarpCondition{
				{
					Type:               "condition1",
					Status:             "true",
					LastProbeTime:      t,
					LastTransitionTime: t,
					Reason:             "reason",
					Message:            "message",
				},
			},
			Message: "message",
			Reason:  "reason",
			HostIP:  "1.2.3.4",
		},
	}
}

func BenchmarkToUnstructured(b *testing.B) {
	carp := benchmarkCarp()
	converter := runtime.DefaultUnstructuredConverter
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result, err := converter.ToUnstructured(carp)
		if err != nil {
			b.Fatalf("Unexpected conversion error: %v", err)
		}
		if len(result) != 3 {
			b.Errorf("Unexpected conversion result: %#v", result)
		}
	}
}

func BenchmarkFromUnstructured(b *testing.B) {
	carp := benchmarkCarp()
	converter := runtime.DefaultUnstructuredConverter
	unstr, err := converter.ToUnstructured(carp)
	if err != nil {
		b.Fatalf("Unexpected conversion error: %v", err)
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		result := testapigroupv1.Carp{}
		if err := converter.FromUnstructured(unstr, &result); err != nil {
			b.Fatalf("Unexpected conversion error: %v", err)
		}
		if result.Status.Phase != "phase" {
			b.Errorf("Unexpected conversion result: %#v", result)
		}
	}
}

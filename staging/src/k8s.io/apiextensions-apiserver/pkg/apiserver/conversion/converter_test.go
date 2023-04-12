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

package conversion

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestConversion(t *testing.T) {
	tests := []struct {
		Name            string
		ValidVersions   []string
		ClusterScoped   bool
		ToVersion       string
		SourceObject    runtime.Object
		ExpectedObject  runtime.Object
		ExpectedFailure string
	}{
		{
			Name:          "simple_conversion",
			ValidVersions: []string{"example.com/v1", "example.com/v2"},
			ClusterScoped: false,
			ToVersion:     "example.com/v2",
			SourceObject: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"metadata":   map[string]interface{}{},
					"other":      "data",
					"kind":       "foo",
				},
			},
			ExpectedObject: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"metadata":   map[string]interface{}{},
					"other":      "data",
					"kind":       "foo",
				},
			},
			ExpectedFailure: "",
		},
		{
			Name:          "failed_conversion_invalid_gv",
			ValidVersions: []string{"example.com/v1", "example.com/v2"},
			ClusterScoped: false,
			ToVersion:     "example.com/v3",
			SourceObject: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"other":      "data",
				},
			},
			ExpectedFailure: "invalid group/version: example.com/v3",
		},
		{
			Name:          "simple_list_conversion",
			ValidVersions: []string{"example.com/v1", "example.com/v2"},
			ClusterScoped: false,
			ToVersion:     "example.com/v2",
			SourceObject: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "fooList",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v1",
							"metadata":   map[string]interface{}{},
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v1",
							"metadata":   map[string]interface{}{},
							"kind":       "foo",
							"other":      "data2",
						},
					},
				},
			},
			ExpectedObject: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "fooList",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v2",
							"metadata":   map[string]interface{}{},
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v2",
							"metadata":   map[string]interface{}{},
							"kind":       "foo",
							"other":      "data2",
						},
					},
				},
			},
			ExpectedFailure: "",
		},
		{
			Name:          "list_with_invalid_gv",
			ValidVersions: []string{"example.com/v1", "example.com/v2"},
			ClusterScoped: false,
			ToVersion:     "example.com/v2",
			SourceObject: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "fooList",
				},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v1",
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v3",
							"kind":       "foo",
							"other":      "data2",
						},
					},
				},
			},
			ExpectedFailure: "invalid group/version: example.com/v3",
		},
	}

	for _, test := range tests {
		testCRD := apiextensionsv1.CustomResourceDefinition{
			Spec: apiextensionsv1.CustomResourceDefinitionSpec{
				Conversion: &apiextensionsv1.CustomResourceConversion{
					Strategy: apiextensionsv1.NoneConverter,
				},
			},
		}
		for _, v := range test.ValidVersions {
			gv, _ := schema.ParseGroupVersion(v)
			testCRD.Spec.Versions = append(testCRD.Spec.Versions, apiextensionsv1.CustomResourceDefinitionVersion{Name: gv.Version, Served: true})
			testCRD.Spec.Group = gv.Group
		}
		safeConverter, _, err := NewDelegatingConverter(&testCRD, NewNOPConverter())
		if err != nil {
			t.Fatalf("Cannot create converter: %v", err)
		}
		o := test.SourceObject.DeepCopyObject()
		toVersion, _ := schema.ParseGroupVersion(test.ToVersion)
		toVersions := schema.GroupVersions{toVersion}
		actual, err := safeConverter.ConvertToVersion(o, toVersions)
		if test.ExpectedFailure != "" {
			if err == nil || !strings.Contains(err.Error(), test.ExpectedFailure) {
				t.Fatalf("%s: Expected the call to fail with error message `%s` but err=%v", test.Name, test.ExpectedFailure, err)
			}
		} else {
			if err != nil {
				t.Fatalf("%s: conversion failed with : %v", test.Name, err)
			}
			if !reflect.DeepEqual(test.ExpectedObject, actual) {
				t.Fatalf("%s: Expected = %v, Actual = %v", test.Name, test.ExpectedObject, actual)
			}
		}
	}
}

func TestGetObjectsToConvert(t *testing.T) {
	v1Object := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "foo/v1", "kind": "Widget", "metadata": map[string]interface{}{"name": "myv1"}}}
	v2Object := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "foo/v2", "kind": "Widget", "metadata": map[string]interface{}{"name": "myv2"}}}
	v3Object := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "foo/v3", "kind": "Widget", "metadata": map[string]interface{}{"name": "myv3"}}}

	testcases := []struct {
		Name          string
		List          *unstructured.UnstructuredList
		APIVersion    string
		ValidVersions map[schema.GroupVersion]bool

		ExpectObjects []*unstructured.Unstructured
		ExpectError   bool
	}{
		{
			Name:       "empty list",
			List:       &unstructured.UnstructuredList{},
			APIVersion: "foo/v1",
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v1"}: true,
			},
			ExpectObjects: nil,
		},
		{
			Name: "one-item list, in desired version",
			List: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{*v1Object},
			},
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v1"}: true,
			},
			APIVersion:    "foo/v1",
			ExpectObjects: nil,
		},
		{
			Name: "one-item list, not in desired version",
			List: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{*v2Object},
			},
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v1"}: true,
				{Group: "foo", Version: "v2"}: true,
			},
			APIVersion:    "foo/v1",
			ExpectObjects: []*unstructured.Unstructured{v2Object},
		},
		{
			Name: "multi-item list, in desired version",
			List: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{*v1Object, *v1Object, *v1Object},
			},
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v1"}: true,
				{Group: "foo", Version: "v2"}: true,
			},
			APIVersion:    "foo/v1",
			ExpectObjects: nil,
		},
		{
			Name: "multi-item list, mixed versions",
			List: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{*v1Object, *v2Object, *v3Object},
			},
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v1"}: true,
				{Group: "foo", Version: "v2"}: true,
				{Group: "foo", Version: "v3"}: true,
			},
			APIVersion:    "foo/v1",
			ExpectObjects: []*unstructured.Unstructured{v2Object, v3Object},
		},
		{
			Name: "multi-item list, invalid versions",
			List: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{*v1Object, *v2Object, *v3Object},
			},
			ValidVersions: map[schema.GroupVersion]bool{
				{Group: "foo", Version: "v2"}: true,
				{Group: "foo", Version: "v3"}: true,
			},
			APIVersion:    "foo/v1",
			ExpectObjects: nil,
			ExpectError:   true,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			objects, err := getObjectsToConvert(tc.List, tc.APIVersion, tc.ValidVersions)
			gotError := err != nil
			if e, a := tc.ExpectError, gotError; e != a {
				t.Fatalf("error: expected %t, got %t", e, a)
			}
			if !reflect.DeepEqual(objects, tc.ExpectObjects) {
				t.Errorf("unexpected diff: %s", cmp.Diff(tc.ExpectObjects, objects))
			}
		})
	}
}

func TestConverterMutatesInput(t *testing.T) {
	testCRD := apiextensionsv1.CustomResourceDefinition{
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Conversion: &apiextensionsv1.CustomResourceConversion{
				Strategy: apiextensionsv1.NoneConverter,
			},
			Group: "test.k8s.io",
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{
					Name:   "v1alpha1",
					Served: true,
				},
				{
					Name:   "v1alpha2",
					Served: true,
				},
			},
		},
	}

	safeConverter, _, err := NewDelegatingConverter(&testCRD, &inputMutatingConverter{})
	if err != nil {
		t.Fatalf("Cannot create converter: %v", err)
	}

	input := &unstructured.UnstructuredList{
		Object: map[string]interface{}{
			"apiVersion": "test.k8s.io/v1alpha1",
		},
		Items: []unstructured.Unstructured{
			{
				Object: map[string]interface{}{
					"apiVersion": "test.k8s.io/v1alpha1",
					"metadata": map[string]interface{}{
						"name": "item1",
					},
				},
			},
			{
				Object: map[string]interface{}{
					"apiVersion": "test.k8s.io/v1alpha1",
					"metadata": map[string]interface{}{
						"name": "item2",
					},
				},
			},
		},
	}

	toVersion, _ := schema.ParseGroupVersion("test.k8s.io/v1alpha2")
	toVersions := schema.GroupVersions{toVersion}
	converted, err := safeConverter.ConvertToVersion(input, toVersions)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	convertedList := converted.(*unstructured.UnstructuredList)
	if e, a := 2, len(convertedList.Items); e != a {
		t.Fatalf("length: expected %d, got %d", e, a)
	}
}

type inputMutatingConverter struct{}

func (i *inputMutatingConverter) Convert(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
	out := &unstructured.UnstructuredList{}
	for _, obj := range in.Items {
		u := obj.DeepCopy()
		u.SetAPIVersion(targetGVK.String())
		out.Items = append(out.Items, *u)
	}

	in.Items = nil

	return out, nil
}

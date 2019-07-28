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

	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/util/webhook"
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
					"other":      "data",
					"kind":       "foo",
				},
			},
			ExpectedObject: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
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
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v1",
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
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v2",
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

	CRConverterFactory, err := NewCRConverterFactory(nil, func(resolver webhook.AuthenticationInfoResolver) webhook.AuthenticationInfoResolver { return nil })
	if err != nil {
		t.Fatalf("Cannot create conversion factory: %v", err)
	}
	for _, test := range tests {
		testCRD := apiextensions.CustomResourceDefinition{
			Spec: apiextensions.CustomResourceDefinitionSpec{
				Conversion: &apiextensions.CustomResourceConversion{
					Strategy: apiextensions.NoneConverter,
				},
			},
		}
		for _, v := range test.ValidVersions {
			gv, _ := schema.ParseGroupVersion(v)
			testCRD.Spec.Versions = append(testCRD.Spec.Versions, apiextensions.CustomResourceDefinitionVersion{Name: gv.Version, Served: true})
			testCRD.Spec.Group = gv.Group
		}
		safeConverter, _, err := CRConverterFactory.NewConverter(&testCRD)
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

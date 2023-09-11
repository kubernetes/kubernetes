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
	"fmt"
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
					"metadata":   map[string]interface{}{"name": "foo1"},
					"other":      "data",
					"kind":       "foo",
				},
			},
			ExpectedObject: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"metadata":   map[string]interface{}{"name": "foo1"},
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
							"metadata":   map[string]interface{}{"name": "foo1"},
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v1",
							"metadata":   map[string]interface{}{"name": "foo2"},
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
							"metadata":   map[string]interface{}{"name": "foo1"},
							"kind":       "foo",
							"other":      "data",
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "example.com/v2",
							"metadata":   map[string]interface{}{"name": "foo2"},
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
		t.Run(test.Name, func(t *testing.T) {
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
		})
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

		ExpectObjects []unstructured.Unstructured
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
			ExpectObjects: []unstructured.Unstructured{*v2Object},
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
			ExpectObjects: []unstructured.Unstructured{*v2Object, *v3Object},
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

func TestDelegatingCRConverterConvertToVersion(t *testing.T) {
	type args struct {
		in     runtime.Object
		target runtime.GroupVersioner
	}
	tests := []struct {
		name      string
		converter CRConverter
		args      args
		want      runtime.Object
		wantErr   bool
	}{
		{
			name:      "empty",
			converter: NewNOPConverter(),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.UnstructuredList{Object: map[string]interface{}{
				"apiVersion": "example.com/v2",
				"kind":       "FooList",
			}, Items: []unstructured.Unstructured{}},
		},
		{
			name:      "happy path in-place",
			converter: NewNOPConverter(),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.UnstructuredList{Object: map[string]interface{}{
				"apiVersion": "example.com/v2",
				"kind":       "FooList",
			}, Items: []unstructured.Unstructured{
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo1"},
					"spec":       map[string]interface{}{},
				}},
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo2"},
					"spec":       map[string]interface{}{},
				}},
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo3"},
					"spec":       map[string]interface{}{},
				}},
			}},
		},
		{
			name: "happy path copying",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				return NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.UnstructuredList{Object: map[string]interface{}{
				"apiVersion": "example.com/v2",
				"kind":       "FooList",
			}, Items: []unstructured.Unstructured{
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo1"},
					"spec":       map[string]interface{}{},
				}},
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo2"},
					"spec":       map[string]interface{}{},
				}},
				{Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo3"},
					"spec":       map[string]interface{}{},
				}},
			}},
		},
		{
			name: "mutating name",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
				ret.Items[0].SetName("mutated")
				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "mutating uid",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
				ret.Items[0].SetUID("mutated")
				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "mutating namespace",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
				ret.Items[0].SetNamespace("mutated")
				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "mutating kind",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
				ret.Items[0].SetKind("Moo")
				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "mutating labels and annotations",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)

				labels := ret.Items[0].GetLabels()
				labels["foo"] = "bar"
				ret.Items[0].SetLabels(labels)

				annotations := ret.Items[0].GetAnnotations()
				annotations["foo"] = "bar"
				ret.Items[0].SetAnnotations(annotations)

				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata": map[string]interface{}{
							"name":        "foo1",
							"labels":      map[string]interface{}{"a": "b"},
							"annotations": map[string]interface{}{"c": "d"},
						},
						"spec": map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata": map[string]interface{}{
						"name":        "foo1",
						"labels":      map[string]interface{}{"a": "b", "foo": "bar"},
						"annotations": map[string]interface{}{"c": "d", "foo": "bar"},
					},
					"spec": map[string]interface{}{},
				},
			},
		},
		{
			name: "mutating any other metadata",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGVK schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				ret, _ := NewNOPConverter().Convert(in.DeepCopy(), targetGVK)
				ret.Items[0].SetFinalizers([]string{"foo"})
				return ret, nil
			}),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{"name": "foo1"},
					"spec":       map[string]interface{}{},
				},
			},
		},
		{
			name:      "empty metadata",
			converter: NewNOPConverter(),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{},
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"metadata":   map[string]interface{}{},
					"spec":       map[string]interface{}{},
				},
			},
		},
		{
			name:      "missing metadata",
			converter: NewNOPConverter(),
			args: args{
				in: &unstructured.Unstructured{
					Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"spec":       map[string]interface{}{},
					},
				},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			want: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "example.com/v2",
					"kind":       "Foo",
					"spec":       map[string]interface{}{},
				},
			},
		},
		{
			name: "convertor error",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				return nil, fmt.Errorf("boom")
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "invalid number returned",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				in.Items[0].SetGroupVersionKind(targetGV.WithKind(in.Items[0].GroupVersionKind().Kind))
				in.Items = in.Items[:1]
				return in, nil
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "partial conversion",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				in.Items[0].SetGroupVersionKind(targetGV.WithKind(in.Items[0].GroupVersionKind().Kind))
				return in, nil
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "invalid single version",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				in.Items[0].SetGroupVersionKind(targetGV.WithKind(in.Items[0].GroupVersionKind().Kind))
				return in, nil
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v1",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v3",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
		{
			name: "invalid list version",
			converter: CRConverterFunc(func(in *unstructured.UnstructuredList, targetGV schema.GroupVersion) (*unstructured.UnstructuredList, error) {
				in.Items[0].SetGroupVersionKind(targetGV.WithKind(in.Items[0].GroupVersionKind().Kind))
				return in, nil
			}),
			args: args{
				in: &unstructured.UnstructuredList{Object: map[string]interface{}{
					"apiVersion": "example.com/v3",
					"kind":       "FooList",
				}, Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo1"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v2",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo2"},
						"spec":       map[string]interface{}{},
					}},
					{Object: map[string]interface{}{
						"apiVersion": "example.com/v1",
						"kind":       "Foo",
						"metadata":   map[string]interface{}{"name": "foo3"},
						"spec":       map[string]interface{}{},
					}},
				}},
				target: schema.GroupVersion{Group: "example.com", Version: "v2"},
			},
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &delegatingCRConverter{
				converter: tt.converter,
				validVersions: map[schema.GroupVersion]bool{
					{Group: "example.com", Version: "v1"}: true,
					{Group: "example.com", Version: "v2"}: true,
				},
			}
			got, err := c.ConvertToVersion(tt.args.in, tt.args.target)
			if (err != nil) != tt.wantErr {
				t.Errorf("ConvertToVersion() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ConvertToVersion() got = %v, want %v", got, tt.want)
			}
		})
	}
}

/*
Copyright 2023 The Kubernetes Authors.

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

package meta

import (
	"reflect"
	"strconv"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

const (
	fakeObjectItemsNum = 1000
	exemptObjectIndex  = fakeObjectItemsNum / 4
)

type SampleSpec struct {
	Flied int
}

type FooSpec struct {
	Flied int
}

type FooList struct {
	metav1.TypeMeta
	metav1.ListMeta
	Items []Foo
}

func (s *FooList) DeepCopyObject() runtime.Object { panic("unimplemented") }

type SampleList struct {
	metav1.TypeMeta
	metav1.ListMeta
	Items []Sample
}

func (s *SampleList) DeepCopyObject() runtime.Object { panic("unimplemented") }

type RawExtensionList struct {
	metav1.TypeMeta
	metav1.ListMeta

	Items []runtime.RawExtension
}

func (l RawExtensionList) DeepCopyObject() runtime.Object { panic("unimplemented") }

// NOTE: Foo struct itself is the implementer of runtime.Object.
type Foo struct {
	metav1.TypeMeta
	metav1.ObjectMeta
	Spec FooSpec
}

func (f Foo) GetObjectKind() schema.ObjectKind {
	tm := f.TypeMeta
	return &tm
}

func (f Foo) DeepCopyObject() runtime.Object { panic("unimplemented") }

// NOTE: the pointer of Sample that is the implementer of runtime.Object.
// the behavior is similar to our corev1.Pod. corev1.Node
type Sample struct {
	metav1.TypeMeta
	metav1.ObjectMeta
	Spec SampleSpec
}

func (s *Sample) GetObjectKind() schema.ObjectKind {
	tm := s.TypeMeta
	return &tm
}

func (s *Sample) DeepCopyObject() runtime.Object { panic("unimplemented") }

func fakeSampleList(numItems int) *SampleList {
	out := &SampleList{
		Items: make([]Sample, numItems),
	}

	for i := range out.Items {
		out.Items[i] = Sample{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "sample.org/v1",
				Kind:       "Sample",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      strconv.Itoa(i),
				Namespace: "default",
				Labels: map[string]string{
					"label-key-1": "label-value-1",
				},
				Annotations: map[string]string{
					"annotations-key-1": "annotations-value-1",
				},
			},
			Spec: SampleSpec{
				Flied: i,
			},
		}
	}
	return out
}

func fakeExtensionList(numItems int) *RawExtensionList {
	out := &RawExtensionList{
		Items: make([]runtime.RawExtension, numItems),
	}

	for i := range out.Items {
		out.Items[i] = runtime.RawExtension{
			Object: &Foo{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "sample.org/v2",
					Kind:       "Sample",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name:      strconv.Itoa(i),
					Namespace: "default",
					Labels: map[string]string{
						"label-key-1": "label-value-1",
					},
					Annotations: map[string]string{
						"annotations-key-1": "annotations-value-1",
					},
				},
				Spec: FooSpec{
					Flied: i,
				},
			},
		}
	}
	return out
}

func fakeUnstructuredList(numItems int) runtime.Unstructured {
	out := &unstructured.UnstructuredList{
		Items: make([]unstructured.Unstructured, numItems),
	}

	for i := range out.Items {
		out.Items[i] = unstructured.Unstructured{
			Object: map[string]interface{}{
				"apiVersion": "v1",
				"kind":       "Pod",
				"metadata": map[string]interface{}{
					"creationTimestamp": nil,
					"name":              strconv.Itoa(i),
				},
				"spec": map[string]interface{}{
					"hostname": "example.com",
				},
				"status": map[string]interface{}{},
			},
		}
	}
	return out
}

func fakeFooList(numItems int) *FooList {
	out := &FooList{
		Items: make([]Foo, numItems),
	}

	for i := range out.Items {
		out.Items[i] = Foo{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "foo.org/v1",
				Kind:       "Foo",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      strconv.Itoa(i),
				Namespace: "default",
				Labels: map[string]string{
					"label-key-1": "label-value-1",
				},
				Annotations: map[string]string{
					"annotations-key-1": "annotations-value-1",
				},
			},
			Spec: FooSpec{
				Flied: i,
			},
		}
	}
	return out
}

func TestEachList(t *testing.T) {
	tests := []struct {
		name            string
		generateFunc    func(num int) (list runtime.Object)
		expectObjectNum int
	}{
		{
			name: "StructReceiverList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeFooList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "PointerReceiverList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeSampleList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "RawExtensionList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeExtensionList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "UnstructuredList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeUnstructuredList(fakeObjectItemsNum)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("EachListItem", func(t *testing.T) {
				expectObjectNames := map[string]struct{}{}
				for i := 0; i < tc.expectObjectNum; i++ {
					expectObjectNames[strconv.Itoa(i)] = struct{}{}
				}
				list := tc.generateFunc(tc.expectObjectNum)
				err := EachListItem(list, func(object runtime.Object) error {
					o, err := Accessor(object)
					if err != nil {
						return err
					}
					delete(expectObjectNames, o.GetName())
					return nil
				})
				if err != nil {
					t.Errorf("each list item %#v: %v", list, err)
				}
				if len(expectObjectNames) != 0 {
					t.Fatal("expectObjectNames should be empty")
				}
			})
			t.Run("EachListItemWithAlloc", func(t *testing.T) {
				expectObjectNames := map[string]struct{}{}
				for i := 0; i < tc.expectObjectNum; i++ {
					expectObjectNames[strconv.Itoa(i)] = struct{}{}
				}
				list := tc.generateFunc(tc.expectObjectNum)
				err := EachListItemWithAlloc(list, func(object runtime.Object) error {
					o, err := Accessor(object)
					if err != nil {
						return err
					}
					delete(expectObjectNames, o.GetName())
					return nil
				})
				if err != nil {
					t.Errorf("each list %#v with alloc: %v", list, err)
				}
				if len(expectObjectNames) != 0 {
					t.Fatal("expectObjectNames should be empty")
				}
			})
		})
	}
}

func TestExtractList(t *testing.T) {
	tests := []struct {
		name            string
		generateFunc    func(num int) (list runtime.Object)
		expectObjectNum int
	}{
		{
			name: "StructReceiverList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeFooList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "PointerReceiverList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeSampleList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "RawExtensionList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeExtensionList(num)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
		{
			name: "UnstructuredList",
			generateFunc: func(num int) (list runtime.Object) {
				return fakeUnstructuredList(fakeObjectItemsNum)
			},
			expectObjectNum: fakeObjectItemsNum,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Run("ExtractList", func(t *testing.T) {
				expectObjectNames := map[string]struct{}{}
				for i := 0; i < tc.expectObjectNum; i++ {
					expectObjectNames[strconv.Itoa(i)] = struct{}{}
				}
				list := tc.generateFunc(tc.expectObjectNum)
				objs, err := ExtractList(list)
				if err != nil {
					t.Fatalf("extract list %#v: %v", list, err)
				}
				for i := range objs {
					var (
						o   metav1.Object
						err error
						obj = objs[i]
					)

					if reflect.TypeOf(obj).Kind() == reflect.Struct {
						copy := reflect.New(reflect.TypeOf(obj))
						copy.Elem().Set(reflect.ValueOf(obj))
						o, err = Accessor(copy.Interface())
					} else {
						o, err = Accessor(obj)
					}
					if err != nil {
						t.Fatalf("Accessor object %#v: %v", obj, err)
					}
					delete(expectObjectNames, o.GetName())
				}
				if len(expectObjectNames) != 0 {
					t.Fatal("expectObjectNames should be empty")
				}
			})
			t.Run("ExtractListWithAlloc", func(t *testing.T) {
				expectObjectNames := map[string]struct{}{}
				for i := 0; i < tc.expectObjectNum; i++ {
					expectObjectNames[strconv.Itoa(i)] = struct{}{}
				}
				list := tc.generateFunc(tc.expectObjectNum)
				objs, err := ExtractListWithAlloc(list)
				if err != nil {
					t.Fatalf("extract list with alloc: %v", err)
				}
				for i := range objs {
					var (
						o   metav1.Object
						err error
						obj = objs[i]
					)
					if reflect.TypeOf(obj).Kind() == reflect.Struct {
						copy := reflect.New(reflect.TypeOf(obj))
						copy.Elem().Set(reflect.ValueOf(obj))
						o, err = Accessor(copy.Interface())
					} else {
						o, err = Accessor(obj)
					}
					if err != nil {
						t.Fatalf("Accessor object %#v: %v", obj, err)
					}
					delete(expectObjectNames, o.GetName())
				}
				if len(expectObjectNames) != 0 {
					t.Fatal("expectObjectNames should be empty")
				}
			})
		})
	}
}

func BenchmarkExtractListItem(b *testing.B) {
	tests := []struct {
		name string
		list runtime.Object
	}{
		{
			name: "StructReceiverList",
			list: fakeFooList(fakeObjectItemsNum),
		},
		{
			name: "PointerReceiverList",
			list: fakeSampleList(fakeObjectItemsNum),
		},
		{
			name: "RawExtensionList",
			list: fakeExtensionList(fakeObjectItemsNum),
		},
	}
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ExtractList(tc.list)
				if err != nil {
					b.Fatalf("ExtractList: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

func BenchmarkEachListItem(b *testing.B) {
	tests := []struct {
		name string
		list runtime.Object
	}{
		{
			name: "StructReceiverList",
			list: fakeFooList(fakeObjectItemsNum),
		},
		{
			name: "PointerReceiverList",
			list: fakeSampleList(fakeObjectItemsNum),
		},
		{
			name: "RawExtensionList",
			list: fakeExtensionList(fakeObjectItemsNum),
		},
		{
			name: "UnstructuredList",
			list: fakeUnstructuredList(fakeObjectItemsNum),
		},
	}
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				err := EachListItem(tc.list, func(object runtime.Object) error {
					return nil
				})
				if err != nil {
					b.Fatalf("EachListItem: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

func BenchmarkExtractListItemWithAlloc(b *testing.B) {
	tests := []struct {
		name string
		list runtime.Object
	}{
		{
			name: "StructReceiverList",
			list: fakeFooList(fakeObjectItemsNum),
		},
		{
			name: "PointerReceiverList",
			list: fakeSampleList(fakeObjectItemsNum),
		},
		{
			name: "RawExtensionList",
			list: fakeExtensionList(fakeObjectItemsNum),
		},
	}
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_, err := ExtractListWithAlloc(tc.list)
				if err != nil {
					b.Fatalf("ExtractListWithAlloc: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

func BenchmarkEachListItemWithAlloc(b *testing.B) {
	tests := []struct {
		name string
		list runtime.Object
	}{
		{
			name: "StructReceiverList",
			list: fakeFooList(fakeObjectItemsNum),
		},
		{
			name: "PointerReceiverList",
			list: fakeSampleList(fakeObjectItemsNum),
		},
		{
			name: "RawExtensionList",
			list: fakeExtensionList(fakeObjectItemsNum),
		},
		{
			name: "UnstructuredList",
			list: fakeUnstructuredList(fakeObjectItemsNum),
		},
	}
	for _, tc := range tests {
		b.Run(tc.name, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				err := EachListItemWithAlloc(tc.list, func(object runtime.Object) error {
					return nil
				})
				if err != nil {
					b.Fatalf("EachListItemWithAlloc: %v", err)
				}
			}
			b.StopTimer()
		})
	}
}

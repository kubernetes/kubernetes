package v1

import (
	reflect "reflect"
	"sync"
	"testing"
)

func TestUnsafeConversionWithRealObjects(t *testing.T) {
	m := &PartialObjectMetadata{
		ObjectMeta: ObjectMeta{
			Name: "object",
			Annotations: map[string]string{
				"a": "1",
			},
		},
	}

	obj, err := scheme.UnsafeConvertToVersion(m, SchemeGroupVersion)
	if err != nil {
		t.Fatal(err)
	}
	converted, ok := obj.(*PartialObjectMetadata)
	if !ok {
		t.Fatal(err)
	}
	if expect, actual := reflect.ValueOf(m).Pointer(), reflect.ValueOf(converted).Pointer(); expect == actual {
		t.Errorf("UnsafeConvertToVersion should not return the same object because we mutate TypeMeta, %d vs %d", expect, actual)
	}

	if expect, actual := reflect.ValueOf(m.Annotations).Pointer(), reflect.ValueOf(converted.Annotations).Pointer(); expect != actual {
		t.Errorf("UnsafeConvertToVersion should reference the same nested fields, %d vs %d", expect, actual)
	}
}

func BenchmarkUnsafeConversionWithRealObjects(b *testing.B) {
	m := &PartialObjectMetadata{
		ObjectMeta: ObjectMeta{
			Name: "object",
			Annotations: map[string]string{
				"a": "1",
			},
		},
	}

	for i := 0; i < b.N; i++ {
		obj, err := scheme.UnsafeConvertToVersion(m, SchemeGroupVersion)
		if err != nil {
			b.Fatal(err)
		}
		_, ok := obj.(*PartialObjectMetadata)
		if !ok {
			b.Fatal(err)
		}
	}
}

func TestUnsafeConversionConcurrently(t *testing.T) {
	m := &PartialObjectMetadata{
		ObjectMeta: ObjectMeta{
			Name: "object",
			Annotations: map[string]string{
				"a": "1",
			},
		},
	}

	iterations := 10
	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			obj, err := scheme.UnsafeConvertToVersion(m, SchemeGroupVersion)
			if err != nil {
				t.Error(err)
				return
			}
			_, ok := obj.(*PartialObjectMetadata)
			if !ok {
				t.Errorf("unknown type %T", obj)
			}
		}
	}()
	go func() {
		defer wg.Done()
		for i := 0; i < iterations; i++ {
			obj, err := scheme.UnsafeConvertToVersion(m, SchemeGroupVersion)
			if err != nil {
				t.Error(err)
				return
			}
			_, ok := obj.(*PartialObjectMetadata)
			if !ok {
				t.Errorf("unknown type %T", obj)
			}
		}
	}()
	wg.Wait()
}

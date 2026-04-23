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

package cacher

import (
	"bytes"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

type mockEncoder struct {
	identifier     runtime.Identifier
	expectedResult string
	expectedError  error

	callsNumber int32
}

func newMockEncoder(id, result string, err error) *mockEncoder {
	return &mockEncoder{
		identifier:     runtime.Identifier(id),
		expectedResult: result,
		expectedError:  err,
	}
}

func (e *mockEncoder) encode(_ runtime.Object, w io.Writer) error {
	atomic.AddInt32(&e.callsNumber, 1)
	if e.expectedError != nil {
		return e.expectedError
	}
	_, err := w.Write([]byte(e.expectedResult))
	return err
}

func TestCachingObject(t *testing.T) {
	object, err := newCachingObject(&v1.Pod{})
	if err != nil {
		t.Fatalf("couldn't create cachingObject: %v", err)
	}

	encoders := []*mockEncoder{
		newMockEncoder("1", "result1", nil),
		newMockEncoder("2", "", fmt.Errorf("mock error")),
		newMockEncoder("3", "result3", nil),
	}

	for _, encoder := range encoders {
		buffer := bytes.NewBuffer(nil)
		err := object.CacheEncode(encoder.identifier, encoder.encode, buffer)
		if a, e := err, encoder.expectedError; e != a {
			t.Errorf("%s: unexpected error: %v, expected: %v", encoder.identifier, a, e)
		}
		if a, e := buffer.String(), encoder.expectedResult; e != a {
			t.Errorf("%s: unexpected result: %s, expected: %s", encoder.identifier, a, e)
		}
	}
	for _, encoder := range encoders {
		if encoder.callsNumber != 1 {
			t.Errorf("%s: unexpected number of encode() calls: %d", encoder.identifier, encoder.callsNumber)
		}
	}
}

func TestCachingObjectFieldAccessors(t *testing.T) {
	object, err := newCachingObject(&v1.Pod{})
	if err != nil {
		t.Fatalf("couldn't create cachingObject: %v", err)
	}

	// Given accessors for all fields implement the same logic,
	// we are choosing an arbitrary one to test.
	namespace := "namespace"
	object.SetNamespace(namespace)

	encodeNamespace := func(obj runtime.Object, w io.Writer) error {
		accessor, err := meta.Accessor(obj)
		if err != nil {
			t.Fatalf("failed to get accessor for %#v: %v", obj, err)
		}
		_, err = w.Write([]byte(accessor.GetNamespace()))
		return err
	}
	buffer := bytes.NewBuffer(nil)
	if err := object.CacheEncode("", encodeNamespace, buffer); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if a, e := buffer.String(), namespace; a != e {
		t.Errorf("unexpected serialization: %s, expected: %s", a, e)
	}

	// GetObject should also set namespace.
	buffer.Reset()
	if err := encodeNamespace(object.GetObject(), buffer); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if a, e := buffer.String(), namespace; a != e {
		t.Errorf("unexpected serialization: %s, expected: %s", a, e)
	}
}

func TestCachingObjectRaces(t *testing.T) {
	object, err := newCachingObject(&v1.Pod{})
	if err != nil {
		t.Fatalf("couldn't create cachingObject: %v", err)
	}

	encoders := []*mockEncoder{}
	for i := 0; i < 10; i++ {
		encoder := newMockEncoder(fmt.Sprintf("%d", i), "result", nil)
		encoders = append(encoders, encoder)
	}

	numWorkers := 1000
	wg := &sync.WaitGroup{}
	wg.Add(numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func() {
			defer wg.Done()
			object.SetNamespace("namespace")
			buffer := bytes.NewBuffer(nil)
			for _, encoder := range encoders {
				buffer.Reset()
				if err := object.CacheEncode(encoder.identifier, encoder.encode, buffer); err != nil {
					t.Errorf("unexpected error: %v", err)
				}
				if callsNumber := atomic.LoadInt32(&encoder.callsNumber); callsNumber != 1 {
					t.Errorf("unexpected number of serializations: %d", callsNumber)
				}
			}
			accessor, err := meta.Accessor(object.GetObject())
			if err != nil {
				t.Errorf("failed to get accessor: %v", err)
				return
			}
			if namespace := accessor.GetNamespace(); namespace != "namespace" {
				t.Errorf("unexpected namespace: %s", namespace)
			}
		}()
	}
	wg.Wait()
}

// BenchmarkCachingObject_DualIdentifier characterizes the server-side cost of
// KEP-5958: when some watchers opt out of managedFields, each event must be
// serialized under two Identifiers instead of one. Two modes are reported:
//
//   - single:  status quo (all watchers receive the full object).
//   - dual:    transition steady state (mixed clients). Marek's
//     "serialize twice each event" concern lives here.
//
// The accompanying b.Logf prints the per-object byte size of full vs. stripped
// so the bytes-on-wire savings are visible alongside the cache-fill CPU cost.
func BenchmarkCachingObject_DualIdentifier(b *testing.B) {
	pod := loadExemplarPod(b)
	full := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{})
	stripped := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{ExcludeManagedFields: true})
	if full.Identifier() == stripped.Identifier() {
		b.Fatalf("expected distinct identifiers, both were %q", full.Identifier())
	}

	var fullBuf, strippedBuf bytes.Buffer
	if err := full.Encode(pod, &fullBuf); err != nil {
		b.Fatal(err)
	}
	if err := stripped.Encode(pod, &strippedBuf); err != nil {
		b.Fatal(err)
	}
	b.Logf("per-object payload: full=%d B, stripped=%d B, savings=%d B (%.0f%%) per opt-out watcher",
		fullBuf.Len(), strippedBuf.Len(),
		fullBuf.Len()-strippedBuf.Len(),
		float64(fullBuf.Len()-strippedBuf.Len())/float64(fullBuf.Len())*100)

	run := func(b *testing.B, encoders ...runtime.Encoder) {
		b.ReportAllocs()
		var buf bytes.Buffer
		for i := 0; i < b.N; i++ {
			obj, err := newCachingObject(pod)
			if err != nil {
				b.Fatal(err)
			}
			for _, enc := range encoders {
				buf.Reset()
				if err := enc.Encode(obj, &buf); err != nil {
					b.Fatal(err)
				}
			}
		}
	}

	b.Run("variant=single", func(b *testing.B) { run(b, full) })
	b.Run("variant=dual", func(b *testing.B) { run(b, full, stripped) })
}

func TestCachingObjectLazyDeepCopy(t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "name",
			ResourceVersion: "123",
		},
	}
	object, err := newCachingObject(pod)
	if err != nil {
		t.Fatalf("couldn't create cachingObject: %v", err)
	}

	if object.deepCopied != false {
		t.Errorf("object deep-copied without the need")
	}

	object.SetResourceVersion("123")
	if object.deepCopied != false {
		t.Errorf("object deep-copied on no-op change")
	}
	object.SetResourceVersion("234")
	if object.deepCopied != true {
		t.Errorf("object not deep-copied on change")
	}
}

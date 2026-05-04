/*
Copyright 2026 The Kubernetes Authors.

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

package cbor

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/fxamacker/cbor/v2"
	"github.com/google/go-cmp/cmp"

	"sigs.k8s.io/randfill"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"
)

func TestCollectionsEncoding(t *testing.T) {
	t.Run("Normal", func(t *testing.T) {
		testCollectionsEncoding(t, false)
	})
	t.Run("Streaming", func(t *testing.T) {
		testCollectionsEncoding(t, true)
	})
}

func testCollectionsEncoding(t *testing.T, streamingEnabled bool) {
	var buf writeCountingBuffer
	var remainingItems int64 = 1
	for _, tc := range []struct {
		name         string
		in           runtime.Object
		cannotStream bool
	}{
		// Preserving the distinction between integers and floating-point numbers
		{
			name: "Struct with floats",
			in: &StructWithFloatsList{
				Items: []StructWithFloats{
					{
						Int:     1,
						Float32: float32(1),
						Float64: 1.1,
					},
				},
			},
		},
		{
			name: "Unstructured object float",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"int":     1,
					"float32": float32(1),
					"float64": 1.1,
				},
			},
		},
		{
			name: "Unstructured items float",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"int":     1,
							"float32": float32(1),
							"float64": 1.1,
						},
					},
				},
			},
		},
		// Handling structs with duplicate field names (JSON tag names) without producing duplicate keys in the encoded output
		{
			name: "StructWithDuplicatedTags",
			in: &StructWithDuplicatedTagsList{
				Items: []StructWithDuplicatedTags{
					{
						Key1: "key1",
						Key2: "key2",
					},
				},
			},
		},
		// Encoding Go strings containing invalid UTF-8 sequences without error
		{
			name: "UnstructuredList object invalid UTF-8",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"key": "\x80", // first byte is a continuation byte
				},
			},
		},
		{
			name: "UnstructuredList items invalid UTF-8",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"key": "\x80",
						},
					},
				},
			},
		},
		// Preserving the distinction between absent, present-but-null, and present-and-empty states for slices and maps
		{
			name: "CarpList items nil",
			in: &testapigroupv1.CarpList{
				Items: nil,
			},
		},
		{
			name: "CarpList slice nil",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Status: testapigroupv1.CarpStatus{
							Conditions: nil,
						},
					},
				},
			},
		},
		{
			name: "CarpList map nil",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Spec: testapigroupv1.CarpSpec{
							NodeSelector: nil,
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList items nil",
			in: &unstructured.UnstructuredList{
				Items: nil,
			},
		},
		{
			name: "UnstructuredList items slice nil",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"slice": ([]string)(nil),
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList items map nil",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"map": (map[string]string)(nil),
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList object nil",
			in: &unstructured.UnstructuredList{
				Object: nil,
			},
		},
		{
			name: "UnstructuredList object slice nil",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": ([]string)(nil),
				},
			},
		},
		{
			name: "UnstructuredList object map nil",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"map": (map[string]string)(nil),
				},
			},
		},
		{
			name: "CarpList items empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{},
			},
		},
		{
			name: "CarpList slice empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Status: testapigroupv1.CarpStatus{
							Conditions: []testapigroupv1.CarpCondition{},
						},
					},
				},
			},
		},
		{
			name: "CarpList map empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{
					{
						Spec: testapigroupv1.CarpSpec{
							NodeSelector: map[string]string{},
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList items empty",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{},
			},
		},
		{
			name: "UnstructuredList items slice empty",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"slice": []string{},
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList items map empty",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"map": map[string]string{},
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList object empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{},
			},
		},
		{
			name: "UnstructuredList object slice empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": []string{},
				},
			},
		},
		{
			name: "UnstructuredList object map empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"map": map[string]string{},
				},
			},
		},
		// Handling structs implementing json.Marshaler method
		{
			name:         "List with json.Marshaler cannot be streamed",
			in:           &ListWithMarshalJSONList{},
			cannotStream: true,
		},
		{
			name: "Struct with json.Marshaler",
			in: &StructWithMarshalJSONList{
				Items: []StructWithMarshalJSON{
					{},
				},
			},
		},
		// Handling structs implementing json.Marshaler but NOT cbor.Marshaler
		{
			name:         "List with json.Marshaler but no cbor.Marshaler cannot be streamed",
			in:           &ListWithMarshalJSONNoCBORList{},
			cannotStream: true,
		},
		{
			name: "Struct with json.Marshaler but no cbor.Marshaler",
			in: &StructWithMarshalJSONNoCBORList{
				Items: []StructWithMarshalJSONNoCBOR{
					{},
				},
			},
		},
		// Handling raw bytes.
		{
			name: "Struct with raw bytes",
			in: &StructWithRawBytesList{
				Items: []StructWithRawBytes{
					{
						Slice: []byte{0x01, 0x02, 0x03},
						Array: [3]byte{0x01, 0x02, 0x03},
					},
				},
			},
		},
		{
			name: "UnstructuredList object raw bytes",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": []byte{0x01, 0x02, 0x03},
					"array": [3]byte{0x01, 0x02, 0x03},
				},
			},
		},
		{
			name: "UnstructuredList items raw bytes",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"slice": []byte{0x01, 0x02, 0x03},
							"array": [3]byte{0x01, 0x02, 0x03},
						},
					},
				},
			},
		},
		// Other scenarios:
		{
			name: "List just kind",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind: "List",
				},
			},
		},
		{
			name: "List just apiVersion",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
				},
			},
		},
		{
			name: "List no elements",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "2345",
				},
				Items: []testapigroupv1.Carp{},
			},
		},
		{
			name: "List one element with continue",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion:    "2345",
					Continue:           "abc",
					RemainingItemCount: &remainingItems,
				},
				Items: []testapigroupv1.Carp{
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod",
						Namespace: "default",
					}},
				},
			},
		},
		{
			name: "List two elements",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "2345",
				},
				Items: []testapigroupv1.Carp{
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod",
						Namespace: "default",
					}},
					{TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
						Name:      "pod2",
						Namespace: "default2",
					}},
				},
			},
		},
		{
			name: "List with extra field cannot be streamed",
			in: &ListWithAdditionalFields{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "2345",
				},
				Items: []testapigroupv1.Carp{},
			},
			cannotStream: true,
		},
		{
			name: "Not a collection cannot be streamed",
			in: &testapigroupv1.Carp{
				TypeMeta: metav1.TypeMeta{
					Kind:       "List",
					APIVersion: "v1",
				},
			},
			cannotStream: true,
		},
		{
			name: "UnstructuredList empty",
			in:   &unstructured.UnstructuredList{},
		},
		{
			name: "UnstructuredList just kind",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List"},
			},
		},
		{
			name: "UnstructuredList just apiVersion",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "v1"},
			},
		},
		{
			name: "UnstructuredList no elements",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{"resourceVersion": "2345"}},
				Items:  []unstructured.Unstructured{},
			},
		},
		{
			name: "UnstructuredList one element with continue",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{
					"resourceVersion":    "2345",
					"continue":           "abc",
					"remainingItemCount": "1",
				}},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Carp",
							"metadata": map[string]interface{}{
								"name":      "pod",
								"namespace": "default",
							},
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList two elements",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{
					"resourceVersion": "2345",
				}},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Carp",
							"metadata": map[string]interface{}{
								"name":      "pod",
								"namespace": "default",
							},
						},
					},
					{
						Object: map[string]interface{}{
							"apiVersion": "v1",
							"kind":       "Carp",
							"metadata": map[string]interface{}{
								"name":      "pod2",
								"namespace": "default",
							},
						},
					},
				},
			},
		},
		{
			name: "UnstructuredList conflict on items",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"items": []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"name": "pod",
						},
					},
				}},
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"name": "pod2",
						},
					},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			buf.Reset()
			s := NewSerializer(nil, nil, StreamingCollectionsEncoding(streamingEnabled))
			if err := s.Encode(tc.in, &buf); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}

			var normalBuf bytes.Buffer
			normalS := NewSerializer(nil, nil, StreamingCollectionsEncoding(false))
			if err := normalS.Encode(tc.in, &normalBuf); err != nil {
				t.Fatalf("normal encode error: %v", err)
			}

			if diff := cmp.Diff(buf.Bytes(), normalBuf.Bytes()); diff != "" {
				t.Errorf("streaming and normal encoding differ:\n%s", diff)
			}

			expectStreaming := !tc.cannotStream && streamingEnabled
			if expectStreaming && buf.writeCount <= 2 {
				t.Errorf("expected streaming but Write was called only: %d", buf.writeCount)
			}
			if !expectStreaming && buf.writeCount > 2 {
				t.Errorf("expected non-streaming but Write was called more than once: %d", buf.writeCount)
			}
		})
	}
}

type StructWithFloatsList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithFloats `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *StructWithFloatsList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithFloats struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Int     int
	Float32 float32
	Float64 float64
}

func (s *StructWithFloats) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithDuplicatedTagsList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithDuplicatedTags `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *StructWithDuplicatedTagsList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithDuplicatedTags struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	Key1 string `json:"key"`
	Key2 string `json:"key"` //nolint:govet
}

func (s *StructWithDuplicatedTags) DeepCopyObject() runtime.Object {
	return nil
}

type ListWithMarshalJSONList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []string `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *ListWithMarshalJSONList) DeepCopyObject() runtime.Object {
	return nil
}

func (l *ListWithMarshalJSONList) MarshalJSON() ([]byte, error) {
	return []byte(`"marshallJSON"`), nil
}

type StructWithMarshalJSONList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithMarshalJSON `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (s *StructWithMarshalJSONList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithMarshalJSON struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

func (l *StructWithMarshalJSON) DeepCopyObject() runtime.Object {
	return nil
}

func (l *StructWithMarshalJSON) MarshalJSON() ([]byte, error) {
	return []byte(`"marshallJSON"`), nil
}

type ListWithMarshalJSONNoCBORList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithMarshalJSONNoCBOR `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *ListWithMarshalJSONNoCBORList) DeepCopyObject() runtime.Object {
	return nil
}

func (l *ListWithMarshalJSONNoCBORList) MarshalJSON() ([]byte, error) {
	return []byte(`"marshalJSONNoCBOR"`), nil
}

type StructWithMarshalJSONNoCBORList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithMarshalJSONNoCBOR `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (s *StructWithMarshalJSONNoCBORList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithMarshalJSONNoCBOR struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

func (l *StructWithMarshalJSONNoCBOR) DeepCopyObject() runtime.Object {
	return nil
}

func (l *StructWithMarshalJSONNoCBOR) MarshalJSON() ([]byte, error) {
	return []byte(`"marshalJSONNoCBOR"`), nil
}

type StructWithRawBytesList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithRawBytes `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (s *StructWithRawBytesList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithRawBytes struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Slice             []byte
	Array             [3]byte
}

func (s *StructWithRawBytes) DeepCopyObject() runtime.Object {
	return nil
}

type ListWithAdditionalFields struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []testapigroupv1.Carp `json:"items" protobuf:"bytes,2,rep,name=items"`
	AdditionalField int
}

func (s *ListWithAdditionalFields) DeepCopyObject() runtime.Object {
	return nil
}

type writeCountingBuffer struct {
	writeCount int
	bytes.Buffer
}

func (b *writeCountingBuffer) Write(data []byte) (int, error) {
	b.writeCount++
	return b.Buffer.Write(data)
}

func (b *writeCountingBuffer) Reset() {
	b.writeCount = 0
	b.Buffer.Reset()
}

func TestFuzzCollectionsEncoding(t *testing.T) {
	disableFuzzFieldsV1 := func(field *metav1.FieldsV1, c randfill.Continue) {}
	fuzzUnstructuredList := func(list *unstructured.UnstructuredList, c randfill.Continue) {
		list.Object = map[string]interface{}{
			"kind":       "List",
			"apiVersion": "v1",
			c.String(0):  c.String(0),
			c.String(0):  c.Uint64(),
			c.String(0):  c.Bool(),
			"metadata": map[string]interface{}{
				"resourceVersion":    fmt.Sprintf("%d", c.Uint64()),
				"continue":           c.String(0),
				"remainingItemCount": fmt.Sprintf("%d", c.Uint64()),
				c.String(0):          c.String(0),
			}}
		c.Fill(&list.Items)
	}
	fuzzMap := func(kvs map[string]interface{}, c randfill.Continue) {
		kvs[c.String(0)] = c.Bool()
		kvs[c.String(0)] = c.Uint64()
		kvs[c.String(0)] = c.String(0)
	}
	f := randfill.New().Funcs(disableFuzzFieldsV1, fuzzUnstructuredList, fuzzMap)
	streamingBuffer := &bytes.Buffer{}
	normalSerializer := NewSerializer(nil, nil)
	normalBuffer := &bytes.Buffer{}
	t.Run("CarpList", func(t *testing.T) {
		for i := 0; i < 1000; i++ {
			list := &testapigroupv1.CarpList{}
			f.Fill(list)
			streamingBuffer.Reset()
			normalBuffer.Reset()
			if _, err := streamingBuffer.Write(selfDescribedCBOR); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			ok, err := streamEncodeCollections(list, streamingBuffer, modes.Encode)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if err := normalSerializer.Encode(list, normalBuffer); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(normalBuffer.Bytes(), streamingBuffer.Bytes()); diff != "" {
				t.Logf("normal: %x", normalBuffer.Bytes())
				t.Logf("streaming: %x", streamingBuffer.Bytes())
				t.Errorf("not matching:\n%s", diff)
			}
		}
	})
	t.Run("UnstructuredList", func(t *testing.T) {
		for i := 0; i < 1000; i++ {
			list := &unstructured.UnstructuredList{}
			f.Fill(list)
			streamingBuffer.Reset()
			normalBuffer.Reset()
			if _, err := streamingBuffer.Write(selfDescribedCBOR); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			ok, err := streamEncodeCollections(list, streamingBuffer, modes.Encode)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if err := normalSerializer.Encode(list, normalBuffer); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(normalBuffer.Bytes(), streamingBuffer.Bytes()); diff != "" {
				t.Logf("normal: %x", normalBuffer.Bytes())
				t.Logf("streaming: %x", streamingBuffer.Bytes())
				t.Errorf("not matching:\n%s", diff)
			}
		}
	})
	// Test EncodeNondeterministic: key order may differ from Encode, so we only
	// verify the output is structurally valid CBOR (has selfDescribedCBOR prefix).
	t.Run("CarpList/Nondeterministic", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			list := &testapigroupv1.CarpList{}
			f.Fill(list)
			streamingBuffer.Reset()
			if _, err := streamingBuffer.Write(selfDescribedCBOR); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			ok, err := streamEncodeCollections(list, streamingBuffer, modes.EncodeNondeterministic)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if !bytes.HasPrefix(streamingBuffer.Bytes(), selfDescribedCBOR) {
				t.Errorf("streaming output missing selfDescribedCBOR prefix")
			}
		}
	})
	t.Run("UnstructuredList/Nondeterministic", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			list := &unstructured.UnstructuredList{}
			f.Fill(list)
			streamingBuffer.Reset()
			if _, err := streamingBuffer.Write(selfDescribedCBOR); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			ok, err := streamEncodeCollections(list, streamingBuffer, modes.EncodeNondeterministic)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if !bytes.HasPrefix(streamingBuffer.Bytes(), selfDescribedCBOR) {
				t.Errorf("streaming output missing selfDescribedCBOR prefix")
			}
		}
	})
}

// extractCBORMapKeys decodes the top-level CBOR map keys (as strings) from data in
// insertion order, skipping the selfDescribedCBOR tag prefix (0xd9d9f7) if present.
// Only the immediate keys of the outermost map are returned; values are skipped via
// cbor streaming decoder. Keys must be CBOR byte strings (major type 2) or text
// strings (major type 3) with a short length (≤23 bytes).
func extractCBORMapKeys(t *testing.T, data []byte) []string {
	t.Helper()
	// Skip selfDescribedCBOR tag prefix if present.
	if bytes.HasPrefix(data, selfDescribedCBOR) {
		data = data[len(selfDescribedCBOR):]
	}
	if len(data) == 0 {
		return nil
	}
	// Parse map header manually to get the number of entries and
	// advance pos past the header byte(s).
	pos := 0
	mapByte := data[pos]
	pos++
	if mapByte>>5 != 5 {
		t.Fatalf("extractCBORMapKeys: expected major type 5 (map), got byte 0x%02x", mapByte)
	}
	addInfo := mapByte & 0x1f
	var mapSize int
	switch {
	case addInfo <= 23:
		mapSize = int(addInfo)
	case addInfo == 24:
		mapSize = int(data[pos])
		pos++
	case addInfo == 25:
		mapSize = int(data[pos])<<8 | int(data[pos+1])
		pos += 2
	default:
		t.Fatalf("extractCBORMapKeys: unsupported map size additional info %d", addInfo)
	}
	// Use a streaming decoder to read key+value pairs one at a time.
	// This correctly handles each value's variable byte length.
	dec := cbor.NewDecoder(bytes.NewReader(data[pos:]))
	keys := make([]string, 0, mapSize)
	for i := 0; i < mapSize; i++ {
		// Decode the key as a RawMessage, then extract the string from the raw bytes.
		var rawKey cbor.RawMessage
		if err := dec.Decode(&rawKey); err != nil {
			t.Fatalf("extractCBORMapKeys: decode key %d: %v", i, err)
		}
		if len(rawKey) == 0 {
			t.Fatalf("extractCBORMapKeys: empty raw key at index %d", i)
		}
		keyMajor := rawKey[0] >> 5
		if keyMajor != 2 && keyMajor != 3 {
			t.Fatalf("extractCBORMapKeys: key %d has unexpected major type %d (byte 0x%02x)", i, keyMajor, rawKey[0])
		}
		// Extract the string content: skip the header byte(s).
		keyHdrLen := 1
		if rawKey[0]&0x1f == 24 {
			keyHdrLen = 2 // 1 type byte + 1 length byte
		}
		keys = append(keys, string(rawKey[keyHdrLen:]))
		// Skip the value.
		var rawVal cbor.RawMessage
		if err := dec.Decode(&rawVal); err != nil {
			t.Fatalf("extractCBORMapKeys: decode value for key %q: %v", keys[len(keys)-1], err)
		}
	}
	return keys
}

// TestStreamEncodeCollectionsDeterministic verifies that streamEncodeCollections
// with modes.Encode (SortBytewiseLexical) produces:
//  1. Idempotent output: the same input always encodes to identical bytes.
//  2. Correct key order: top-level map keys follow SortBytewiseLexical
//     (shorter length first, then lexicographic within same length).
func TestStreamEncodeCollectionsDeterministic(t *testing.T) {
	wantKeyOrder := []string{"kind", "items", "metadata", "apiVersion"}

	for _, tc := range []struct {
		name string
		in   runtime.Object
	}{
		{
			name: "CarpList with all top-level fields",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "CarpList",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "42",
				},
				Items: []testapigroupv1.Carp{
					{ObjectMeta: metav1.ObjectMeta{Name: "a"}},
				},
			},
		},
		{
			name: "UnstructuredList with all top-level fields",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{"resourceVersion": "42"},
				},
				Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{"name": "a"}},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// 1. Idempotence: encode twice, bytes must match.
			var buf1, buf2 bytes.Buffer
			for _, buf := range []*bytes.Buffer{&buf1, &buf2} {
				if _, err := buf.Write(selfDescribedCBOR); err != nil {
					t.Fatal(err)
				}
				ok, err := streamEncodeCollections(tc.in, buf, modes.Encode)
				if err != nil {
					t.Fatalf("streamEncodeCollections error: %v", err)
				}
				if !ok {
					t.Fatalf("expected streaming encoder to handle %T", tc.in)
				}
			}
			if diff := cmp.Diff(buf1.Bytes(), buf2.Bytes()); diff != "" {
				t.Errorf("deterministic encoding is not idempotent:\n%s", diff)
			}

			// 2. Key order follows SortBytewiseLexical.
			// Expected order for {kind(4), items(5), metadata(8), apiVersion(10)}:
			// shorter length first → kind < items < metadata < apiVersion
			gotKeys := extractCBORMapKeys(t, buf1.Bytes())
			if diff := cmp.Diff(wantKeyOrder, gotKeys); diff != "" {
				t.Errorf("top-level key order does not follow SortBytewiseLexical:\n%s", diff)
			}
		})
	}
}

// TestStreamEncodeCollectionsNondeterministic verifies that streamEncodeCollections
// with modes.EncodeNondeterministic (SortFastShuffle):
//  1. Semantic correctness: the output decodes to the same object as deterministic encoding.
//  2. Non-idempotence: across multiple trials the key order is observed to vary
//     (probabilistic; uses a multi-key object to make the probability of flake negligible).
func TestStreamEncodeCollectionsNondeterministic(t *testing.T) {
	// A list with kind+apiVersion+metadata+items = 4 keys.
	// With SortFastShuffle the number of possible orderings is 4! = 24.
	// Over 200 trials the probability of seeing only 1 unique ordering is (1/24)^199 ≈ 0.
	const nTrials = 200

	for _, tc := range []struct {
		name string
		in   runtime.Object
	}{
		{
			name: "CarpList",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind:       "CarpList",
					APIVersion: "v1",
				},
				ListMeta: metav1.ListMeta{
					ResourceVersion: "42",
				},
				Items: []testapigroupv1.Carp{
					{ObjectMeta: metav1.ObjectMeta{Name: "a"}},
				},
			},
		},
		{
			name: "UnstructuredList",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind":       "List",
					"apiVersion": "v1",
					"metadata":   map[string]interface{}{"resourceVersion": "42"},
				},
				Items: []unstructured.Unstructured{
					{Object: map[string]interface{}{"name": "a"}},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Encode once with deterministic mode as reference for semantic equality.
			var detBuf bytes.Buffer
			if _, err := detBuf.Write(selfDescribedCBOR); err != nil {
				t.Fatal(err)
			}
			ok, err := streamEncodeCollections(tc.in, &detBuf, modes.Encode)
			if err != nil {
				t.Fatalf("deterministic encode error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to handle %T", tc.in)
			}
			var detObj map[string]interface{}
			if err := modes.Decode.Unmarshal(detBuf.Bytes()[len(selfDescribedCBOR):], &detObj); err != nil {
				t.Fatalf("decode deterministic output: %v", err)
			}

			// Run nTrials of nondeterministic encoding.
			uniqueOutputs := make(map[string]struct{})
			for i := 0; i < nTrials; i++ {
				var buf bytes.Buffer
				if _, err := buf.Write(selfDescribedCBOR); err != nil {
					t.Fatal(err)
				}
				ok, err := streamEncodeCollections(tc.in, &buf, modes.EncodeNondeterministic)
				if err != nil {
					t.Fatalf("trial %d: nondeterministic encode error: %v", i, err)
				}
				if !ok {
					t.Fatalf("trial %d: expected streaming encoder to handle %T", i, tc.in)
				}
				payload := buf.Bytes()[len(selfDescribedCBOR):]

				// Semantic correctness: decoded value must equal the deterministic reference.
				var ndetObj map[string]interface{}
				if err := modes.Decode.Unmarshal(payload, &ndetObj); err != nil {
					t.Fatalf("trial %d: decode nondeterministic output: %v", i, err)
				}
				if diff := cmp.Diff(detObj, ndetObj); diff != "" {
					t.Errorf("trial %d: semantic mismatch between deterministic and nondeterministic:\n%s", i, diff)
				}

				uniqueOutputs[string(payload)] = struct{}{}
			}

			// Non-idempotence: must have observed at least 2 distinct byte sequences.
			if len(uniqueOutputs) < 2 {
				t.Errorf("nondeterministic encoding produced only %d unique byte sequence(s) over %d trials; expected varied output", len(uniqueOutputs), nTrials)
			}
			t.Logf("%s: observed %d unique encodings over %d trials", tc.name, len(uniqueOutputs), nTrials)
		})
	}
}

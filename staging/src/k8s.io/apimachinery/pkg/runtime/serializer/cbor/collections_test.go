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

package cbor

import (
	"bytes"
	"fmt"
	"reflect"
	"testing"

	"github.com/google/go-cmp/cmp"

	"sigs.k8s.io/randfill"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/cbor/internal/modes"
)

// TestStreamingCollectionsEncoding verifies that streaming encoding produces
// output identical to normal non-streaming encoding, and that the streaming
// encoder actually uses multiple Write calls (not just buffering everything).
func TestStreamingCollectionsEncoding(t *testing.T) {
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
		// Handling structs implementing cbor.Marshaler but NOT json.Marshaler
		{
			name:         "List with cbor.Marshaler cannot be streamed",
			in:           &ListWithMarshalCBORList{},
			cannotStream: true,
		},
		{
			name: "Struct with cbor.Marshaler",
			in: &StructWithMarshalCBORList{
				Items: []StructWithMarshalCBOR{
					{},
				},
			},
		},
		// Handling structs implementing both json.Marshaler and cbor.Marshaler
		{
			name:         "List with json.Marshaler and cbor.Marshaler cannot be streamed",
			in:           &ListWithBothMarshalersList{},
			cannotStream: true,
		},
		{
			name: "Struct with json.Marshaler and cbor.Marshaler",
			in: &StructWithBothMarshalersList{
				Items: []StructWithBothMarshalers{
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
					{
						TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
							Name:      "pod",
							Namespace: "default",
						},
					},
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
					{
						TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
							Name:      "pod",
							Namespace: "default",
						},
					},
					{
						TypeMeta: metav1.TypeMeta{APIVersion: "v1", Kind: "Carp"}, ObjectMeta: metav1.ObjectMeta{
							Name:      "pod2",
							Namespace: "default2",
						},
					},
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
				Object: map[string]interface{}{
					"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{"resourceVersion": "2345"},
				},
				Items: []unstructured.Unstructured{},
			},
		},
		{
			name: "UnstructuredList one element with continue",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{
						"resourceVersion":    "2345",
						"continue":           "abc",
						"remainingItemCount": "1",
					},
				},
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
				Object: map[string]interface{}{
					"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{
						"resourceVersion": "2345",
					},
				},
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
				Object: map[string]interface{}{
					"items": []unstructured.Unstructured{
						{
							Object: map[string]interface{}{
								"name": "pod",
							},
						},
					},
				},
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
			s := NewSerializer(nil, nil, StreamingCollectionsEncoding(true))
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

			expectStreaming := !tc.cannotStream
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

type ListWithMarshalCBORList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []string `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *ListWithMarshalCBORList) DeepCopyObject() runtime.Object {
	return nil
}

func (l *ListWithMarshalCBORList) MarshalCBOR() ([]byte, error) {
	return []byte("\x6bmarshalCBOR"), nil
}

type StructWithMarshalCBORList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithMarshalCBOR `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (s *StructWithMarshalCBORList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithMarshalCBOR struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

func (l *StructWithMarshalCBOR) DeepCopyObject() runtime.Object {
	return nil
}

func (l *StructWithMarshalCBOR) MarshalCBOR() ([]byte, error) {
	return []byte("\x6bmarshalCBOR"), nil
}

type ListWithBothMarshalersList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []string `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (l *ListWithBothMarshalersList) DeepCopyObject() runtime.Object {
	return nil
}

func (l *ListWithBothMarshalersList) MarshalJSON() ([]byte, error) {
	return []byte(`"marshalJSON"`), nil
}

func (l *ListWithBothMarshalersList) MarshalCBOR() ([]byte, error) {
	return []byte("\x6bmarshalCBOR"), nil
}

type StructWithBothMarshalersList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	Items           []StructWithBothMarshalers `json:"items" protobuf:"bytes,2,rep,name=items"`
}

func (s *StructWithBothMarshalersList) DeepCopyObject() runtime.Object {
	return nil
}

type StructWithBothMarshalers struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
}

func (l *StructWithBothMarshalers) DeepCopyObject() runtime.Object {
	return nil
}

func (l *StructWithBothMarshalers) MarshalJSON() ([]byte, error) {
	return []byte(`"marshalJSON"`), nil
}

func (l *StructWithBothMarshalers) MarshalCBOR() ([]byte, error) {
	return []byte("\x6bmarshalCBOR"), nil
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
			c.String(0):  int64(c.Intn(1000000)), // Limit to int64 range
			c.String(0):  c.Bool(),
			"metadata": map[string]interface{}{
				"resourceVersion":    fmt.Sprintf("%d", c.Intn(1000000)), // String format
				"continue":           c.String(0),
				"remainingItemCount": fmt.Sprintf("%d", c.Intn(1000000)), // String format
				c.String(0):          c.String(0),
			},
		}
		c.Fill(&list.Items)
	}
	fuzzMap := func(kvs map[string]interface{}, c randfill.Continue) {
		kvs[c.String(0)] = c.Bool()
		kvs[c.String(0)] = int64(c.Intn(1000000)) // Limit to int64 range
		kvs[c.String(0)] = c.String(0)
	}
	f := randfill.New().Funcs(disableFuzzFieldsV1, fuzzUnstructuredList, fuzzMap)
	streamingSerializer := NewSerializer(nil, nil)
	normalSerializer := NewSerializer(nil, nil, StreamingCollectionsEncoding(false))

	for _, tc := range []struct {
		name   string
		newObj func() runtime.Object
	}{
		{name: "CarpList", newObj: func() runtime.Object { return &testapigroupv1.CarpList{} }},
		{name: "UnstructuredList", newObj: func() runtime.Object { return &unstructured.UnstructuredList{} }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var streamingBuf writeCountingBuffer
			var normalBuf, ndetBuf bytes.Buffer
			for i := range 1000 {
				obj := tc.newObj()
				f.Fill(obj)

				// Non-streaming, deterministic encode as the reference for all comparisons.
				normalBuf.Reset()
				if err := normalSerializer.Encode(obj, &normalBuf); err != nil {
					t.Fatalf("trial %d: normal encode error: %v", i, err)
				}

				// Streaming deterministic encode must match the reference byte-for-byte.
				streamingBuf.Reset()
				if err := streamingSerializer.Encode(obj, &streamingBuf); err != nil {
					t.Fatalf("trial %d: streaming encode error: %v", i, err)
				}
				if diff := cmp.Diff(normalBuf.Bytes(), streamingBuf.Bytes()); diff != "" {
					t.Logf("normal:    %x", normalBuf.Bytes())
					t.Logf("streaming: %x", streamingBuf.Bytes())
					t.Fatalf("trial %d: streaming and non-streaming differ:\n%s", i, diff)
				}
				if streamingBuf.writeCount <= 2 {
					t.Errorf("trial %d: expected streaming encoding to use more than 2 writes, got %d", i, streamingBuf.writeCount)
				}

				// Streaming nondeterministic encode must decode to the same value.
				ndetBuf.Reset()
				if err := streamingSerializer.EncodeNondeterministic(obj, &ndetBuf); err != nil {
					t.Fatalf("trial %d: nondeterministic encode error: %v", i, err)
				}

				detPayload := normalBuf.Bytes()[len(selfDescribedCBOR):]
				ndetPayload := ndetBuf.Bytes()[len(selfDescribedCBOR):]

				var detObj, ndetObj interface{}
				if err := modes.Decode.Unmarshal(detPayload, &detObj); err != nil {
					t.Fatalf("trial %d: decode deterministic: %v", i, err)
				}
				if err := modes.Decode.Unmarshal(ndetPayload, &ndetObj); err != nil {
					t.Fatalf("trial %d: decode nondeterministic: %v", i, err)
				}
				if diff := cmp.Diff(detObj, ndetObj); diff != "" {
					t.Errorf("trial %d: semantic mismatch between deterministic and nondeterministic:\n%s", i, diff)
				}

				detTyped := reflect.New(reflect.TypeOf(obj).Elem()).Interface()
				ndetTyped := reflect.New(reflect.TypeOf(obj).Elem()).Interface()
				if err := modes.Decode.Unmarshal(detPayload, detTyped); err != nil {
					t.Fatalf("trial %d: decode deterministic into %T: %v", i, obj, err)
				}
				if err := modes.Decode.Unmarshal(ndetPayload, ndetTyped); err != nil {
					t.Fatalf("trial %d: decode nondeterministic into %T: %v", i, obj, err)
				}
				if !reflect.DeepEqual(detTyped, ndetTyped) {
					t.Errorf("trial %d: typed %T mismatch between deterministic and nondeterministic encodings", i, obj)
				}
			}
		})
	}
}

// TestStreamEncodeCollectionsDeterministic verifies that the streaming serializer
// produces output identical to the normal non-streaming serializer.
func TestStreamEncodeCollectionsDeterministic(t *testing.T) {
	streamingSerializer := NewSerializer(nil, nil)
	normalSerializer := NewSerializer(nil, nil, StreamingCollectionsEncoding(false))

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
					{ObjectMeta: metav1.ObjectMeta{Name: "b"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "c"}},
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
					{Object: map[string]interface{}{"name": "b"}},
					{Object: map[string]interface{}{"name": "b"}},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Encode with streaming enabled.
			var streamingBuf writeCountingBuffer
			if err := streamingSerializer.Encode(tc.in, &streamingBuf); err != nil {
				t.Fatalf("streaming encode error: %v", err)
			}

			// Encode with normal non-streaming encoder.
			var normalBuf bytes.Buffer
			if err := normalSerializer.Encode(tc.in, &normalBuf); err != nil {
				t.Fatalf("normal encode error: %v", err)
			}

			// Output must be identical.
			if diff := cmp.Diff(normalBuf.Bytes(), streamingBuf.Bytes()); diff != "" {
				t.Logf("normal: %x", normalBuf.Bytes())
				t.Logf("streaming: %x", streamingBuf.Bytes())
				t.Errorf("streaming output differs from normal encoding:\n%s", diff)
			}

			if streamingBuf.writeCount <= 2 {
				t.Errorf("expected streaming encoding to use more than 2 writes, got %d", streamingBuf.writeCount)
			}
		})
	}
}

// TestStreamEncodeCollectionsNondeterministic verifies that the streaming serializer's
// EncodeNondeterministic method:
//  1. Semantic correctness: the output decodes to the same object as deterministic encoding.
//  2. Non-idempotence: across multiple trials the key order is observed to vary
//     (probabilistic; uses a multi-key object to make the probability of flake negligible).
func TestStreamEncodeCollectionsNondeterministic(t *testing.T) {
	streamingSerializer := NewSerializer(nil, nil)
	normalSerializer := NewSerializer(nil, nil, StreamingCollectionsEncoding(false))

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
					{ObjectMeta: metav1.ObjectMeta{Name: "b"}},
					{ObjectMeta: metav1.ObjectMeta{Name: "c"}},
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
					{Object: map[string]interface{}{"name": "b"}},
					{Object: map[string]interface{}{"name": "c"}},
				},
			},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			// Encode once with non-streaming, deterministic mode as reference for semantic equality.
			var detBuf bytes.Buffer
			if err := normalSerializer.Encode(tc.in, &detBuf); err != nil {
				t.Fatalf("deterministic encode error: %v", err)
			}
			var detObj map[string]interface{}
			if err := modes.Decode.Unmarshal(detBuf.Bytes()[len(selfDescribedCBOR):], &detObj); err != nil {
				t.Fatalf("decode deterministic output: %v", err)
			}
			detTyped := reflect.New(reflect.TypeOf(tc.in).Elem()).Interface()
			if err := modes.Decode.Unmarshal(detBuf.Bytes()[len(selfDescribedCBOR):], detTyped); err != nil {
				t.Fatalf("decode deterministic output into %T: %v", tc.in, err)
			}

			// Run nTrials of nondeterministic encoding, stopping early once we
			// observe a second distinct byte sequence.
			var firstEncoding string
			varied := false
			for i := range nTrials {
				var buf writeCountingBuffer
				if err := streamingSerializer.EncodeNondeterministic(tc.in, &buf); err != nil {
					t.Fatalf("trial %d: nondeterministic encode error: %v", i, err)
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

				ndetTyped := reflect.New(reflect.TypeOf(tc.in).Elem()).Interface()
				if err := modes.Decode.Unmarshal(payload, ndetTyped); err != nil {
					t.Fatalf("trial %d: decode nondeterministic output into %T: %v", i, tc.in, err)
				}
				if !reflect.DeepEqual(detTyped, ndetTyped) {
					t.Errorf("trial %d: typed %T mismatch between deterministic and nondeterministic encodings", i, tc.in)
				}

				if buf.writeCount <= 2 {
					t.Errorf("trial %d: expected streaming encoding to use more than 2 writes, got %d", i, buf.writeCount)
				}

				enc := string(payload)
				if i == 0 {
					firstEncoding = enc
				} else if enc != firstEncoding {
					varied = true
					break
				}
			}

			if !varied {
				t.Errorf("nondeterministic encoding produced only 1 unique byte sequence over %d trials; expected varied output", nTrials)
			}
		})
	}
}

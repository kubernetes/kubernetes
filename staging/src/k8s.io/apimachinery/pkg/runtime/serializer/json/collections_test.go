/*
Copyright 2025 The Kubernetes Authors.

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

package json

import (
	"bytes"
	"fmt"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	testapigroupv1 "k8s.io/apimachinery/pkg/apis/testapigroup/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/randfill"
)

func TestCollectionsEncoding(t *testing.T) {
	t.Run("Normal", func(t *testing.T) {
		testCollectionsEncoding(t, NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{}), false)
	})
	t.Run("Streaming", func(t *testing.T) {
		testCollectionsEncoding(t, NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{StreamingCollectionsEncoding: true}), true)
	})
}

// testCollectionsEncoding should provide comprehensive tests to validate streaming implementation of encoder.
func testCollectionsEncoding(t *testing.T, s *Serializer, streamingEnabled bool) {
	var buf writeCountingBuffer
	var remainingItems int64 = 1
	// As defined in KEP-5116 we it should include the following scenarios:
	// Context: https://github.com/kubernetes/enhancements/tree/master/keps/sig-api-machinery/5116-streaming-response-encoding#unit-tests
	for _, tc := range []struct {
		name         string
		in           runtime.Object
		cannotStream bool
		expect       string
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"Int\":1,\"Float32\":1,\"Float64\":1.1}]}\n",
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
			expect: "{\"float32\":1,\"float64\":1.1,\"int\":1,\"items\":[]}\n",
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
			expect: "{\"items\":[{\"float32\":1,\"float64\":1.1,\"int\":1}]}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null}}]}\n",
		},
		// Encoding Go strings containing invalid UTF-8 sequences without error
		{
			name: "UnstructuredList object invalid UTF-8 ",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"key": "\x80", // first byte is a continuation byte
				},
			},
			expect: "{\"items\":[],\"key\":\"\\ufffd\"}\n",
		},
		{
			name: "UnstructuredList items invalid UTF-8 ",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{
					{
						Object: map[string]interface{}{
							"key": "\x80", // first byte is a continuation byte
						},
					},
				},
			},
			expect: "{\"items\":[{\"key\":\"\\ufffd\"}]}\n",
		},
		// Preserving the distinction between absent, present-but-null, and present-and-empty states for slices and maps
		{
			name: "CarpList items nil",
			in: &testapigroupv1.CarpList{
				Items: nil,
			},
			expect: "{\"metadata\":{},\"items\":null}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"spec\":{},\"status\":{}}]}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"spec\":{},\"status\":{}}]}\n",
		},
		{
			name: "UnstructuredList items nil",
			in: &unstructured.UnstructuredList{
				Items: nil,
			},
			expect: "{\"items\":[]}\n",
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
			expect: "{\"items\":[{\"slice\":null}]}\n",
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
			expect: "{\"items\":[{\"map\":null}]}\n",
		},
		{
			name: "UnstructuredList object nil",
			in: &unstructured.UnstructuredList{
				Object: nil,
			},
			expect: "{\"items\":[]}\n",
		},
		{
			name: "UnstructuredList object slice nil",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": ([]string)(nil),
				},
			},
			expect: "{\"items\":[],\"slice\":null}\n",
		},
		{
			name: "UnstructuredList object map nil",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"map": (map[string]string)(nil),
				},
			},
			expect: "{\"items\":[],\"map\":null}\n",
		},
		{
			name: "CarpList items empty",
			in: &testapigroupv1.CarpList{
				Items: []testapigroupv1.Carp{},
			},
			expect: "{\"metadata\":{},\"items\":[]}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"spec\":{},\"status\":{}}]}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"spec\":{},\"status\":{}}]}\n",
		},
		{
			name: "UnstructuredList items empty",
			in: &unstructured.UnstructuredList{
				Items: []unstructured.Unstructured{},
			},
			expect: "{\"items\":[]}\n",
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
			expect: "{\"items\":[{\"slice\":[]}]}\n",
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
			expect: "{\"items\":[{\"map\":{}}]}\n",
		},
		{
			name: "UnstructuredList object empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{},
			},
			expect: "{\"items\":[]}\n",
		},
		{
			name: "UnstructuredList object slice empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": []string{},
				},
			},
			expect: "{\"items\":[],\"slice\":[]}\n",
		},
		{
			name: "UnstructuredList object map empty",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"map": map[string]string{},
				},
			},
			expect: "{\"items\":[],\"map\":{}}\n",
		},
		// Handling structs implementing MarshallJSON method, especially built-in collection types.
		{
			name:         "List with MarshallJSON cannot be streamed",
			in:           &ListWithMarshalJSONList{},
			expect:       "\"marshallJSON\"\n",
			cannotStream: true,
		},
		{
			name: "Struct with MarshallJSON",
			in: &StructWithMarshalJSONList{
				Items: []StructWithMarshalJSON{
					{},
				},
			},
			expect: "{\"metadata\":{},\"items\":[\"marshallJSON\"]}\n",
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
			expect: "{\"metadata\":{},\"items\":[{\"metadata\":{\"creationTimestamp\":null},\"Slice\":\"AQID\",\"Array\":[1,2,3]}]}\n",
		},
		{
			name: "UnstructuredList object raw bytes",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{
					"slice": []byte{0x01, 0x02, 0x03},
					"array": [3]byte{0x01, 0x02, 0x03},
				},
			},
			expect: "{\"array\":[1,2,3],\"items\":[],\"slice\":\"AQID\"}\n",
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
			expect: "{\"items\":[{\"array\":[1,2,3],\"slice\":\"AQID\"}]}\n",
		},
		// Other scenarios:
		{
			name: "List just kind",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					Kind: "List",
				},
			},
			expect: "{\"kind\":\"List\",\"metadata\":{},\"items\":null}\n",
		},
		{
			name: "List just apiVersion",
			in: &testapigroupv1.CarpList{
				TypeMeta: metav1.TypeMeta{
					APIVersion: "v1",
				},
			},
			expect: "{\"apiVersion\":\"v1\",\"metadata\":{},\"items\":null}\n",
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
			expect: "{\"kind\":\"List\",\"apiVersion\":\"v1\",\"metadata\":{\"resourceVersion\":\"2345\"},\"items\":[]}\n",
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
			expect: "{\"kind\":\"List\",\"apiVersion\":\"v1\",\"metadata\":{\"resourceVersion\":\"2345\",\"continue\":\"abc\",\"remainingItemCount\":1},\"items\":[{\"kind\":\"Carp\",\"apiVersion\":\"v1\",\"metadata\":{\"name\":\"pod\",\"namespace\":\"default\",\"creationTimestamp\":null},\"spec\":{},\"status\":{}}]}\n",
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
			expect: `{"kind":"List","apiVersion":"v1","metadata":{"resourceVersion":"2345"},"items":[{"kind":"Carp","apiVersion":"v1","metadata":{"name":"pod","namespace":"default","creationTimestamp":null},"spec":{},"status":{}},{"kind":"Carp","apiVersion":"v1","metadata":{"name":"pod2","namespace":"default2","creationTimestamp":null},"spec":{},"status":{}}]}
`,
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
			expect:       "{\"kind\":\"List\",\"apiVersion\":\"v1\",\"metadata\":{\"resourceVersion\":\"2345\"},\"items\":[],\"AdditionalField\":0}\n",
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
			expect:       "{\"kind\":\"List\",\"apiVersion\":\"v1\",\"metadata\":{\"creationTimestamp\":null},\"spec\":{},\"status\":{}}\n",
		},
		{
			name:   "UnstructuredList empty",
			in:     &unstructured.UnstructuredList{},
			expect: "{\"items\":[]}\n",
		},
		{
			name: "UnstructuredList just kind",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List"},
			},
			expect: "{\"items\":[],\"kind\":\"List\"}\n",
		},
		{
			name: "UnstructuredList just apiVersion",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"apiVersion": "v1"},
			},
			expect: "{\"apiVersion\":\"v1\",\"items\":[]}\n",
		},
		{
			name: "UnstructuredList no elements",
			in: &unstructured.UnstructuredList{
				Object: map[string]interface{}{"kind": "List", "apiVersion": "v1", "metadata": map[string]interface{}{"resourceVersion": "2345"}},
				Items:  []unstructured.Unstructured{},
			},
			expect: "{\"apiVersion\":\"v1\",\"items\":[],\"kind\":\"List\",\"metadata\":{\"resourceVersion\":\"2345\"}}\n",
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
			expect: "{\"apiVersion\":\"v1\",\"items\":[{\"apiVersion\":\"v1\",\"kind\":\"Carp\",\"metadata\":{\"name\":\"pod\",\"namespace\":\"default\"}}],\"kind\":\"List\",\"metadata\":{\"continue\":\"abc\",\"remainingItemCount\":\"1\",\"resourceVersion\":\"2345\"}}\n",
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
			expect: "{\"apiVersion\":\"v1\",\"items\":[{\"apiVersion\":\"v1\",\"kind\":\"Carp\",\"metadata\":{\"name\":\"pod\",\"namespace\":\"default\"}},{\"apiVersion\":\"v1\",\"kind\":\"Carp\",\"metadata\":{\"name\":\"pod2\",\"namespace\":\"default\"}}],\"kind\":\"List\",\"metadata\":{\"resourceVersion\":\"2345\"}}\n",
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
			expect: "{\"items\":[{\"name\":\"pod2\"}]}\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			buf.Reset()
			if err := s.Encode(tc.in, &buf); err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			t.Logf("encoded: %s", buf.String())
			if diff := cmp.Diff(buf.String(), tc.expect); diff != "" {
				t.Errorf("not matching:\n%s", diff)
			}
			expectStreaming := !tc.cannotStream && streamingEnabled
			if expectStreaming && buf.writeCount <= 1 {
				t.Errorf("expected streaming but Write was called only: %d", buf.writeCount)
			}
			if !expectStreaming && buf.writeCount > 1 {
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
	normalSerializer := NewSerializerWithOptions(DefaultMetaFactory, nil, nil, SerializerOptions{StreamingCollectionsEncoding: false})
	normalBuffer := &bytes.Buffer{}
	t.Run("CarpList", func(t *testing.T) {
		for i := 0; i < 1000; i++ {
			list := &testapigroupv1.CarpList{}
			f.Fill(list)
			streamingBuffer.Reset()
			normalBuffer.Reset()
			ok, err := streamEncodeCollections(list, streamingBuffer)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if err := normalSerializer.Encode(list, normalBuffer); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(normalBuffer.String(), streamingBuffer.String()); diff != "" {
				t.Logf("normal: %s", normalBuffer.String())
				t.Logf("streaming: %s", streamingBuffer.String())
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
			ok, err := streamEncodeCollections(list, streamingBuffer)
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if !ok {
				t.Fatalf("expected streaming encoder to encode %T", list)
			}
			if err := normalSerializer.Encode(list, normalBuffer); err != nil {
				t.Fatal(err)
			}
			if diff := cmp.Diff(normalBuffer.String(), streamingBuffer.String()); diff != "" {
				t.Logf("normal: %s", normalBuffer.String())
				t.Logf("streaming: %s", streamingBuffer.String())
				t.Errorf("not matching:\n%s", diff)
			}
		}
	})
}

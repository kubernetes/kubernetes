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

package json_test

import (
	"bytes"
	gojson "encoding/json"
	"encoding/json/jsontext"
	"errors"
	"fmt"
	"math"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/versioning"
	"sigs.k8s.io/yaml"
)

// A v1 encoder would encode this as {}, so the marker proves that the v2
// engine was reached through the versioning wrapper.
type typeMetaEncodingProbe struct {
	err error
}

func (p typeMetaEncodingProbe) MarshalJSONTo(enc *jsontext.Encoder) error {
	if p.err != nil {
		return p.err
	}
	return enc.WriteToken(jsontext.String("json/v2"))
}

// TestTypeMetaInjectionUsesJSONV2 verifies that both versioning wrappers reach
// the v2 engine in compact, pretty, and YAML modes, restore existing TypeMeta,
// and return marshaling errors without writing partial output.
func TestTypeMetaInjectionUsesJSONV2(t *testing.T) {
	gvk := schema.GroupVersionKind{Group: "test.example", Version: "v1", Kind: "Test"}
	scheme := runtime.NewScheme()
	scheme.AddKnownTypeWithName(gvk, &testDecodable{})
	marshalErr := errors.New("encoding probe failed")
	for _, mode := range []struct {
		name    string
		options json.SerializerOptions
	}{
		{name: "compact"},
		{name: "pretty", options: json.SerializerOptions{Pretty: true}},
		{name: "yaml", options: json.SerializerOptions{Yaml: true}},
	} {
		for _, wrapper := range []string{"WithVersionEncoder", "versioning codec"} {
			for _, original := range []metav1.TypeMeta{{}, {APIVersion: "old.example/v2", Kind: "Old"}} {
				for _, fail := range []bool{false, true} {
					t.Run(fmt.Sprintf("%s/%s/original=%s/fail=%t", mode.name, wrapper, original.Kind, fail), func(t *testing.T) {
						probe := typeMetaEncodingProbe{}
						if fail {
							probe.err = marshalErr
						}
						obj := &testDecodable{TypeMeta: original, Interface: probe}
						s := json.NewSerializerWithOptions(json.DefaultMetaFactory, scheme, scheme, mode.options)
						var encoder runtime.Encoder = runtime.WithVersionEncoder{Version: gvk.GroupVersion(), Encoder: s, ObjectTyper: scheme}
						if wrapper == "versioning codec" {
							// A nil encode version exercises the codec's own TypeMeta injection
							// rather than relying on conversion to populate the outgoing GVK.
							encoder = versioning.NewDefaultingCodecForScheme(scheme, s, s, nil, nil)
						}
						var out bytes.Buffer
						err := encoder.Encode(obj, &out)
						if obj.TypeMeta != original {
							t.Errorf("input TypeMeta = %#v, want %#v", obj.TypeMeta, original)
						}
						if fail {
							if !errors.Is(err, marshalErr) {
								t.Fatalf("encoding error = %v, want %v", err, marshalErr)
							}
							if out.Len() != 0 {
								t.Errorf("failed encoding wrote %q", out.Bytes())
							}
							return
						}
						if err != nil {
							t.Fatal(err)
						}
						data := out.Bytes()
						if mode.options.Yaml {
							data, err = yaml.YAMLToJSON(data)
							if err != nil {
								t.Fatal(err)
							}
						}
						var got struct {
							metav1.TypeMeta
							Interface string
						}
						if err := gojson.Unmarshal(data, &got); err != nil {
							t.Fatal(err)
						}
						if got.GroupVersionKind() != gvk {
							t.Errorf("encoded GVK = %v, want %v", got.GroupVersionKind(), gvk)
						}
						if got.Interface != "json/v2" {
							t.Errorf("encoding probe = %q, want json/v2", got.Interface)
						}
					})
				}
			}
		}
	}
}

// TestEncodeLegacyCompatibility compares v2 output with encoding/json for
// typed, raw, unstructured, and list objects in each applicable output mode.
// The list cases also exercise streaming buffer reuse with large and small items.
func TestEncodeLegacyCompatibility(t *testing.T) {
	payload := struct {
		Bool  bool              `json:"bool,omitempty"`
		Int   int               `json:"int,omitempty"`
		Empty struct{}          `json:"empty,omitempty"`
		Slice []string          `json:"slice"`
		Map   map[string]string `json:"map"`
		Bytes [2]byte           `json:"bytes"`
		Text  string            `json:"text"`
	}{Text: "<>&\u2028\u2029"}
	for _, tc := range []struct {
		name       string
		obj        runtime.Object
		streamable bool
	}{
		{name: "typed", obj: &testDecodable{Interface: payload}},
		{name: "raw marshaler", obj: &testDecodable{Interface: gojson.RawMessage(` { "empty": { }, "array": [ ], "text": "<>&" } `)}},
		{name: "unstructured", obj: &unstructured.Unstructured{Object: map[string]any{
			"z": int64(math.MaxInt64), "a": []any{nil, "<>&"}, "m": map[string]any{},
		}}},
		{name: "list", streamable: true, obj: &unstructured.UnstructuredList{
			Object: map[string]any{
				"kind": "List", "apiVersion": "v1",
				"metadata": map[string]any{"continue": "<>&"},
			},
			Items: []unstructured.Unstructured{
				{Object: map[string]any{"z": int64(math.MaxInt64), "a": strings.Repeat("x", 10000)}},
				{Object: map[string]any{"nil": nil, "empty": []any{}, "text": "<>&"}},
			},
		}},
	} {
		for _, mode := range []struct {
			name    string
			options json.SerializerOptions
		}{
			{name: "compact"},
			{name: "pretty", options: json.SerializerOptions{Pretty: true}},
			{name: "yaml", options: json.SerializerOptions{Yaml: true}},
			{name: "streaming", options: json.SerializerOptions{StreamingCollectionsEncoding: true}},
		} {
			if mode.options.StreamingCollectionsEncoding && !tc.streamable {
				continue
			}
			t.Run(tc.name+"/"+mode.name, func(t *testing.T) {
				var want []byte
				var err error
				switch {
				case mode.options.Yaml:
					want, err = gojson.Marshal(tc.obj)
					if err == nil {
						want, err = yaml.JSONToYAML(want)
					}
				case mode.options.Pretty:
					want, err = gojson.MarshalIndent(tc.obj, "", "  ")
				default:
					var reference bytes.Buffer
					err = gojson.NewEncoder(&reference).Encode(tc.obj)
					want = reference.Bytes()
				}
				if err != nil {
					t.Fatal(err)
				}
				var got bytes.Buffer
				s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, mode.options)
				if err := s.Encode(tc.obj, &got); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Bytes(), want) {
					t.Errorf("encoding = %q, want %q", got.Bytes(), want)
				}
			})
		}
	}
}

type failingEncodingWriter struct {
	err error
}

func (w failingEncodingWriter) Write([]byte) (int, error) {
	return 0, w.err
}

// TestEncodeErrors verifies that marshaling failures do not write partial
// compact, pretty, YAML, or streaming-fallback output and that each mode
// propagates errors from the destination writer.
func TestEncodeErrors(t *testing.T) {
	for _, options := range []json.SerializerOptions{
		{},
		{Pretty: true},
		{Yaml: true},
		{StreamingCollectionsEncoding: true},
	} {
		t.Run(fmt.Sprintf("%+v", options), func(t *testing.T) {
			s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, options)
			// A large prefix detects encoders that flush before discovering an
			// invalid field, leaving a partial response on the wire.
			obj := &testDecodable{Other: strings.Repeat("x", 10000), Interface: math.Inf(1)}
			var got bytes.Buffer
			if err := s.Encode(obj, &got); err == nil {
				t.Fatal("expected an error encoding an infinite number")
			}
			if got.Len() != 0 {
				t.Fatalf("marshaling error wrote %d bytes", got.Len())
			}
			writeErr := errors.New("write failed")
			if err := s.Encode(&metav1.Status{}, failingEncodingWriter{err: writeErr}); !errors.Is(err, writeErr) {
				t.Fatalf("writer error = %v, want %v", err, writeErr)
			}
		})
	}
}

// TestEncodeConcurrent verifies that parallel calls sharing one serializer and
// its pooled buffers remain isolated and retain encoding/json-compatible output.
func TestEncodeConcurrent(t *testing.T) {
	s := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{})
	for i := range 16 {
		t.Run(fmt.Sprint(i), func(t *testing.T) {
			t.Parallel()
			for j := range 10 {
				var got, want bytes.Buffer
				obj := &testDecodable{Value: i, Interface: j}
				if err := s.Encode(obj, &got); err != nil {
					t.Fatal(err)
				}
				if err := gojson.NewEncoder(&want).Encode(obj); err != nil {
					t.Fatal(err)
				}
				if !bytes.Equal(got.Bytes(), want.Bytes()) {
					t.Fatalf("encoding = %q, want %q", got.Bytes(), want.Bytes())
				}
			}
		})
	}
}

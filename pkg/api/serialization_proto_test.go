/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package api_test

import (
	"bytes"
	"encoding/hex"
	"math/rand"
	"testing"

	"github.com/gogo/protobuf/proto"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/v1"
	_ "k8s.io/kubernetes/pkg/apis/extensions"
	_ "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer/protobuf"
	"k8s.io/kubernetes/pkg/util/diff"
)

func init() {
	codecsToTest = append(codecsToTest, func(version unversioned.GroupVersion, item runtime.Object) (runtime.Codec, error) {
		s := protobuf.NewSerializer(api.Scheme, runtime.ObjectTyperToTyper(api.Scheme), "application/arbitrary.content.type")
		return api.Codecs.CodecForVersions(s, testapi.ExternalGroupVersions(), nil), nil
	})
}

func TestUniversalDeserializer(t *testing.T) {
	expected := &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "test"}}
	d := api.Codecs.UniversalDeserializer()
	for _, mediaType := range []string{"application/json", "application/yaml", "application/vnd.kubernetes.protobuf"} {
		e, ok := api.Codecs.SerializerForMediaType(mediaType, nil)
		if !ok {
			t.Fatal(mediaType)
		}
		buf := &bytes.Buffer{}
		if err := e.EncodeToStream(expected, buf); err != nil {
			t.Fatalf("%s: %v", mediaType, err)
		}
		obj, _, err := d.Decode(buf.Bytes(), &unversioned.GroupVersionKind{Kind: "Pod", Version: "v1"}, nil)
		if err != nil {
			t.Fatalf("%s: %v", mediaType, err)
		}
		if !api.Semantic.DeepEqual(expected, obj) {
			t.Fatalf("%s: %#v", mediaType, obj)
		}
	}
}

func TestProtobufRoundTrip(t *testing.T) {
	obj := &v1.Pod{}
	apitesting.FuzzerFor(t, v1.SchemeGroupVersion, rand.NewSource(benchmarkSeed)).Fuzz(obj)
	data, err := obj.Marshal()
	if err != nil {
		t.Fatal(err)
	}
	out := &v1.Pod{}
	if err := out.Unmarshal(data); err != nil {
		t.Fatal(err)
	}
	if !api.Semantic.Equalities.DeepEqual(out, obj) {
		t.Logf("marshal\n%s", hex.Dump(data))
		t.Fatalf("Unmarshal is unequal\n%s", diff.ObjectGoPrintSideBySide(out, obj))
	}
}

// BenchmarkEncodeCodec measures the cost of performing a codec encode, which includes
// reflection (to clear APIVersion and Kind)
func BenchmarkEncodeCodecProtobuf(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	s := protobuf.NewSerializer(nil, nil, "application/arbitrary.content.type")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(s, &items[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkEncodeCodecFromInternalProtobuf measures the cost of performing a codec encode,
// including conversions and any type setting. This is a "full" encode.
func BenchmarkEncodeCodecFromInternalProtobuf(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	encodable := make([]api.Pod, width)
	for i := range items {
		if err := api.Scheme.Convert(&items[i], &encodable[i]); err != nil {
			b.Fatal(err)
		}
	}
	s := protobuf.NewSerializer(nil, nil, "application/arbitrary.content.type")
	codec := api.Codecs.EncoderForVersion(s, v1.SchemeGroupVersion)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := runtime.Encode(codec, &encodable[i%width]); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

func BenchmarkEncodeProtobufGeneratedMarshal(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := items[i%width].Marshal(); err != nil {
			b.Fatal(err)
		}
	}
	b.StopTimer()
}

// BenchmarkDecodeJSON provides a baseline for regular JSON decode performance
func BenchmarkDecodeIntoProtobuf(b *testing.B) {
	items := benchmarkItems()
	width := len(items)
	encoded := make([][]byte, width)
	for i := range items {
		data, err := (&items[i]).Marshal()
		if err != nil {
			b.Fatal(err)
		}
		encoded[i] = data
		validate := &v1.Pod{}
		if err := proto.Unmarshal(data, validate); err != nil {
			b.Fatalf("Failed to unmarshal %d: %v\n%#v", i, err, items[i])
		}
	}

	for i := 0; i < b.N; i++ {
		obj := v1.Pod{}
		if err := proto.Unmarshal(encoded[i%width], &obj); err != nil {
			b.Fatal(err)
		}
	}
}

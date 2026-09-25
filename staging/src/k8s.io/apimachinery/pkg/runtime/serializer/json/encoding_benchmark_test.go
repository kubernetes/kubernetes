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
	"io"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime/serializer/json"
)

// BenchmarkEncode measures compact serializer encoding across representative object sizes.
func BenchmarkEncode(b *testing.B) {
	serializer := json.NewSerializerWithOptions(json.DefaultMetaFactory, nil, nil, json.SerializerOptions{})
	for _, size := range []struct {
		name  string
		bytes int
	}{
		{name: "1KiB", bytes: 1 << 10},
		{name: "10KiB", bytes: 10 << 10},
		{name: "100KiB", bytes: 100 << 10},
		{name: "1MiB", bytes: 1 << 20},
	} {
		b.Run(size.name, func(b *testing.B) {
			obj := &testDecodable{Other: strings.Repeat("x", size.bytes)}
			b.ReportAllocs()
			b.SetBytes(int64(size.bytes))
			for b.Loop() {
				if err := serializer.Encode(obj, io.Discard); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

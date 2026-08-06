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

package filters

import (
	"net/http"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
)

func BenchmarkNegotiatesStreamingCollectionEncoding(b *testing.B) {
	ns := serializer.NewCodecFactory(runtime.NewScheme(),
		serializer.WithStreamingCollectionEncodingToJSON(),
		serializer.WithStreamingCollectionEncodingToProtobuf())

	cases := []struct {
		name   string
		accept string
	}{
		{"protobuf", "application/vnd.kubernetes.protobuf"},
		{"json", "application/json"},
		{"protobuf-fallback", "application/vnd.kubernetes.protobuf,*/*"},
		{"empty", ""},
	}

	for _, tc := range cases {
		b.Run(tc.name, func(b *testing.B) {
			req, _ := http.NewRequest(http.MethodGet, "/api/v1/pods", nil)
			if tc.accept != "" {
				req.Header.Set("Accept", tc.accept)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				_ = negotiatesStreamingCollectionEncoding(req, ns)
			}
		})
	}
}

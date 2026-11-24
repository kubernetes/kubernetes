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

package compression

import (
	"bytes"
	"context"
	"os"
	"path/filepath"
	"testing"
)

// fixtureDir is where k8s.io/api keeps its round-trip fixtures, relative to this package.
var fixtureDir = filepath.Join("..", "..", "..", "..", "..", "api", "testdata", "HEAD")

var fixtureNames = []string{
	"core.v1.Pod", "apps.v1.StatefulSet", "apps.v1.Deployment", "batch.v1.Job",
	"core.v1.Node", "core.v1.Endpoints", "events.k8s.io.v1.Event", "core.v1.Service",
	"rbac.authorization.k8s.io.v1.Role", "core.v1.Secret", "core.v1.ConfigMap", "coordination.k8s.io.v1.Lease",
}

// loadFixtures reads the k8s.io/api fixtures, which live in another module. A
// checkout of k8s.io/apiserver on its own doesn't have that module next to it,
// so the test skips there. It skips only when the whole module is missing; a
// missing or renamed fixture still fails.
func loadFixtures(tb testing.TB) map[string][]byte {
	tb.Helper()
	if _, err := os.Stat(filepath.Dir(filepath.Dir(fixtureDir))); os.IsNotExist(err) {
		tb.Skipf("k8s.io/api is not checked out next to this module")
	}
	out := map[string][]byte{}
	for _, name := range fixtureNames {
		data, err := os.ReadFile(filepath.Join(fixtureDir, name+".pb"))
		if err != nil {
			tb.Fatal(err)
		}
		out[name] = data
	}
	return out
}

// TestFixturesRoundTrip checks the benchmark inputs on every test run: each
// fixture must load and survive deflateFrame and unframe byte for byte.
func TestFixturesRoundTrip(t *testing.T) {
	ctx := context.Background()
	for name, data := range loadFixtures(t) {
		framed, err := deflateFrame(data)
		if err != nil {
			t.Fatal(err)
		}
		out, kind, err := unframe(ctx, framed)
		if err != nil || kind != framedDeflate || !bytes.Equal(out, data) {
			t.Fatalf("%s: kind %v, err %v", name, kind, err)
		}
	}
}

func BenchmarkCompressFixtures(b *testing.B) {
	for name, data := range loadFixtures(b) {
		b.Run(name, func(b *testing.B) {
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			var framed []byte
			for i := 0; i < b.N; i++ {
				var err error
				if framed, err = deflateFrame(data); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportMetric(float64(len(framed))/float64(len(data)), "stored/plaintext")
		})
	}
}

func BenchmarkDecompressFixtures(b *testing.B) {
	ctx := context.Background()
	for name, data := range loadFixtures(b) {
		b.Run(name, func(b *testing.B) {
			framed, err := deflateFrame(data)
			if err != nil {
				b.Fatal(err)
			}
			b.ReportAllocs()
			b.SetBytes(int64(len(data)))
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := unframe(ctx, framed); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

// BenchmarkUnframeLegacy measures what every cluster pays on every read, because
// the decorator is installed whether or not compression is on.
func BenchmarkUnframeLegacy(b *testing.B) {
	ctx := context.Background()
	// unframe reads only the first byte here, so any value that doesn't start with
	// 0x00 costs the same, and a literal keeps this running without the fixtures.
	data := []byte(`{"apiVersion":"v1","kind":"ConfigMap","metadata":{"name":"cm","namespace":"default"},"data":{"k":"v"}}`)
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if _, _, err := unframe(ctx, data); err != nil {
			b.Fatal(err)
		}
	}
}

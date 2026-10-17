//go:build linux && amd64 && !race

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

package bytecache

import (
	"fmt"
	"reflect"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func codecPod(i int) *corev1.Pod {
	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      fmt.Sprintf("pod-%d", i),
			Namespace: "default",
			Labels:    map[string]string{"app": fmt.Sprintf("pod-%d", i), "tier": "backend"},
		},
		Spec: corev1.PodSpec{
			NodeName: fmt.Sprintf("node-%d", i%100),
			Containers: []corev1.Container{{
				Name:  "c",
				Image: fmt.Sprintf("registry.example.com/app:v%d", i),
				Resources: corev1.ResourceRequirements{
					Requests: corev1.ResourceList{
						corev1.ResourceCPU:    resource.MustParse("100m"),
						corev1.ResourceMemory: resource.MustParse("128Mi"),
					},
				},
			}},
		},
		Status: corev1.PodStatus{Phase: corev1.PodRunning, PodIP: "10.0.0.1"},
	}
	for e := range 8 {
		pod.Spec.Containers[0].Env = append(pod.Spec.Containers[0].Env,
			corev1.EnvVar{Name: fmt.Sprintf("E%d", e), Value: fmt.Sprintf("v%d", e)})
	}
	return pod
}

func newCacheKind(t testing.TB, mode string) *Cache {
	t.Helper()
	t.Setenv("KUBE_BYTECACHE", mode)
	return New()
}

// The self-check must accept reloc and proto for pods, and must blacklist
// gob: resource.Quantity keeps its value in unexported fields, which gob
// silently drops.
func TestCodecSelfCheck(t *testing.T) {
	for _, tc := range []struct {
		mode        string
		wantEncoded bool
	}{
		{"1", true},
		{"proto", true},
		{"gob", false},
	} {
		t.Run(tc.mode, func(t *testing.T) {
			c := newCacheKind(t, tc.mode)
			c.Set("k", codecPod(1))
			encoded, passed, _, _ := c.Stats()
			if tc.wantEncoded && (encoded != 1 || passed != 0) {
				t.Fatalf("mode %s: encoded=%d passed=%d, want encoded", tc.mode, encoded, passed)
			}
			if !tc.wantEncoded && (encoded != 0 || passed != 1) {
				t.Fatalf("mode %s: encoded=%d passed=%d, want passthrough (self-check should reject)", tc.mode, encoded, passed)
			}
			// Whatever the storage, reads must be faithful.
			got, ok := c.Get("k")
			if !ok || !reflect.DeepEqual(codecPod(1), got) {
				t.Fatalf("mode %s: round-trip mismatch", tc.mode)
			}
		})
	}
}

// Direct demonstration of gob's lossiness on k8s types (why the self-check
// exists): without it, Quantities would silently decode as empty.
func TestGobIsLossyOnQuantities(t *testing.T) {
	c := newCacheKind(t, "gob")
	if !c.ensureArena() {
		t.Fatal("arena")
	}
	pod := codecPod(1)
	e, err := c.encodeGob(pod)
	if err != nil {
		t.Fatalf("gob encode unexpectedly failed: %v", err)
	}
	decoded, err := c.decodeGob(e)
	if err != nil {
		t.Fatalf("gob decode failed: %v", err)
	}
	got := decoded.(*corev1.Pod).Spec.Containers[0].Resources.Requests[corev1.ResourceCPU]
	want := pod.Spec.Containers[0].Resources.Requests[corev1.ResourceCPU]
	if got.String() == want.String() {
		t.Fatalf("expected gob to corrupt Quantity (got %s == want %s); if gob became faithful, update the docs", got.String(), want.String())
	}
	t.Logf("gob corrupted Quantity as expected: %q -> %q (size %d bytes vs proto/reloc below)", want.String(), got.String(), e.size)
}

func TestCodecSizes(t *testing.T) {
	for _, mode := range []string{"1", "proto", "gob"} {
		c := newCacheKind(t, mode)
		if !c.ensureArena() {
			t.Fatal("arena")
		}
		rv := reflect.ValueOf(codecPod(1))
		e, err := c.encodeKind(rv, rv.Interface())
		if err != nil {
			t.Fatalf("%s: %v", mode, err)
		}
		extra := 0
		if mode == "1" {
			extra = 4 * e.relocN
		}
		t.Logf("%-6s %6d bytes/pod at rest (+%d reloc list)", mode, e.size, extra)
	}
}

func benchCodec(b *testing.B, mode string, decode bool) {
	c := newCacheKind(b, mode)
	if !c.ensureArena() {
		b.Fatal("arena")
	}
	pod := codecPod(1)
	rv := reflect.ValueOf(pod)
	e, err := c.encodeKind(rv, pod)
	if err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		if decode {
			if _, err := c.decode(e); err != nil {
				b.Fatal(err)
			}
		} else {
			if _, err := c.encodeKind(rv, pod); err != nil {
				c.arena.Reset()
				i--
			}
		}
	}
}

func BenchmarkEncodeReloc(b *testing.B) { benchCodec(b, "1", false) }
func BenchmarkEncodeProto(b *testing.B) { benchCodec(b, "proto", false) }
func BenchmarkEncodeGob(b *testing.B)   { benchCodec(b, "gob", false) }
func BenchmarkDecodeReloc(b *testing.B) { benchCodec(b, "1", true) }
func BenchmarkDecodeProto(b *testing.B) { benchCodec(b, "proto", true) }
func BenchmarkDecodeGob(b *testing.B)   { benchCodec(b, "gob", true) }

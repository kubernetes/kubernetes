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

package watch

import (
	"bytes"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/protobuf"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/client-go/kubernetes/scheme"
)

// repeatReader serves the same pre-framed payload over and over. It only wraps
// around at an exact frame boundary, so the frame reader always sees well-formed
// frames.
type repeatReader struct {
	data []byte
	off  int
}

func (r *repeatReader) Read(p []byte) (int, error) {
	if r.off >= len(r.data) {
		r.off = 0
	}
	n := copy(p, r.data[r.off:])
	r.off += n
	return n, nil
}

func (r *repeatReader) Close() error { return nil }

// benchPod is a Pod of roughly the shape a real watch stream carries.
func benchPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "benchmark-pod-0123456789",
			Namespace: "benchmark-namespace",
			UID:       "8cbef9f6-e45e-4068-8e1b-f853655d6b39",
			Labels: map[string]string{
				"app":                          "benchmark",
				"app.kubernetes.io/component":  "worker",
				"app.kubernetes.io/managed-by": "benchmark",
			},
			Annotations: map[string]string{
				"kubectl.kubernetes.io/last-applied-configuration": "{\"apiVersion\":\"v1\",\"kind\":\"Pod\"}",
			},
		},
		Spec: v1.PodSpec{
			NodeName: "node-benchmark-0",
			Containers: []v1.Container{{
				Name:  "c1",
				Image: "registry.k8s.io/e2e-test-images/busybox:1.37.0-1",
			}},
		},
		Status: v1.PodStatus{
			Phase: v1.PodRunning,
			ContainerStatuses: []v1.ContainerStatus{{
				Name:  "c1",
				Ready: true,
				Image: "registry.k8s.io/e2e-test-images/busybox:1.37.0-1",
			}},
		},
	}
}

// framedProtobufEvent encodes a single watch event through the same streaming
// encoder + length-delimited framer the apiserver uses for protobuf watches.
func framedProtobufEvent(b *testing.B, rawSerializer runtime.Serializer, pod *v1.Pod) []byte {
	b.Helper()

	// The embedded object is protobuf-encoded (prefixed) just like the apiserver
	// writes it into the watch event's RawExtension.
	objEncoder := scheme.Codecs.WithoutConversion().EncoderForVersion(
		protobuf.NewSerializer(scheme.Scheme, scheme.Scheme), v1.SchemeGroupVersion)
	objData, err := runtime.Encode(objEncoder, pod)
	if err != nil {
		b.Fatalf("encode pod: %v", err)
	}

	var buf bytes.Buffer
	frameWriter := protobuf.LengthDelimitedFramer.NewFrameWriter(&buf)
	encoder := streaming.NewEncoder(frameWriter, rawSerializer)
	event := &metav1.WatchEvent{
		Type:   "MODIFIED",
		Object: runtime.RawExtension{Raw: objData},
	}
	if err := encoder.Encode(event); err != nil {
		b.Fatalf("encode watch event: %v", err)
	}
	return buf.Bytes()
}

// BenchmarkDecoderProtobuf measures the per-event cost of the client-side watch
// decode path (the stack built in rest.Request.newStreamWatcher). The reporter
// on https://github.com/kubernetes/kubernetes/issues/129705 profiled this path
// as the dominant allocator under high pod churn, so this benchmark exists to
// make that cost measurable in-tree.
func BenchmarkDecoderProtobuf(b *testing.B) {
	rawSerializer := protobuf.NewRawSerializer(scheme.Scheme, scheme.Scheme)
	objectDecoder := scheme.Codecs.WithoutConversion().DecoderToVersion(
		protobuf.NewSerializer(scheme.Scheme, scheme.Scheme), v1.SchemeGroupVersion)

	frame := framedProtobufEvent(b, rawSerializer, benchPod())

	reader := &repeatReader{data: frame}
	frameReader := protobuf.LengthDelimitedFramer.NewFrameReader(reader)
	decoder := NewDecoder(streaming.NewDecoder(frameReader, rawSerializer), objectDecoder)
	defer decoder.Close()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		action, obj, err := decoder.Decode()
		if err != nil {
			b.Fatalf("decode: %v", err)
		}
		if action != "MODIFIED" || obj == nil {
			b.Fatalf("unexpected decode result: action=%v obj=%v", action, obj)
		}
	}
}

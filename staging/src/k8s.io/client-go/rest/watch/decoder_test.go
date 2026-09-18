/*
Copyright 2014 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimejson "k8s.io/apimachinery/pkg/runtime/serializer/json"
	"k8s.io/apimachinery/pkg/runtime/serializer/streaming"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/kubernetes/scheme"
)

// getDecoder mimics how k8s.io/client-go/rest.createSerializers creates a decoder
func getDecoder() runtime.Decoder {
	jsonSerializer := runtimejson.NewSerializerWithOptions(runtimejson.DefaultMetaFactory, scheme.Scheme, scheme.Scheme, runtimejson.SerializerOptions{})
	directCodecFactory := scheme.Codecs.WithoutConversion()
	return directCodecFactory.DecoderToVersion(jsonSerializer, v1.SchemeGroupVersion)
}

func TestDecoder(t *testing.T) {
	table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error, watch.Bookmark}

	for _, eventType := range table {
		out, in := io.Pipe()

		decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())
		expect := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
		encoder := json.NewEncoder(in)
		eType := eventType
		errc := make(chan error)

		go func() {
			data, err := runtime.Encode(scheme.Codecs.LegacyCodec(v1.SchemeGroupVersion), expect)
			if err != nil {
				errc <- fmt.Errorf("Unexpected error %v", err)
				return
			}
			event := metav1.WatchEvent{
				Type:   string(eType),
				Object: runtime.RawExtension{Raw: json.RawMessage(data)},
			}
			if err := encoder.Encode(&event); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
			in.Close()
		}()

		done := make(chan struct{})
		go func() {
			action, got, err := decoder.Decode()
			if err != nil {
				errc <- fmt.Errorf("Unexpected error %v", err)
				return
			}
			if e, a := eType, action; e != a {
				t.Errorf("Expected %v, got %v", e, a)
			}
			if e, a := expect, got; !apiequality.Semantic.DeepDerivative(e, a) {
				t.Errorf("Expected %v, got %v", e, a)
			}
			t.Logf("Exited read")
			close(done)
		}()
		select {
		case err := <-errc:
			t.Fatal(err)
		case <-done:
		}

		done = make(chan struct{})
		go func() {
			_, _, err := decoder.Decode()
			if err == nil {
				t.Errorf("Unexpected nil error")
			}
			close(done)
		}()
		<-done

		decoder.Close()
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	out, in := io.Pipe()
	decoder := NewDecoder(streaming.NewDecoder(out, getDecoder()), getDecoder())

	done := make(chan struct{})

	go func() {
		_, _, err := decoder.Decode()
		if err == nil {
			t.Errorf("Unexpected nil error")
		}
		close(done)
	}()

	in.Close()

	select {
	case <-done:
		break
	case <-time.After(wait.ForeverTestTimeout):
		t.Error("Timeout")
	}
}

func TestFrameDecoderPreservesOrderWhenWatchEventDecodeCompletesOutOfOrder(t *testing.T) {
	completed := make(chan string, 3)
	decoder := NewFrameDecoder(
		newFakeFrameReader([][]byte{
			[]byte("pod-0"),
			[]byte("pod-1"),
			[]byte("pod-2"),
		}),
		&delayedWatchEventDecoder{
			delays:    map[string]time.Duration{"pod-0": 200 * time.Millisecond},
			completed: completed,
		},
		&delayedObjectDecoder{},
	)
	defer decoder.Close()

	select {
	case name := <-completed:
		if name != "pod-1" && name != "pod-2" {
			t.Fatalf("expected a later watch event frame to decode first, got %q", name)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for concurrent watch event frame decode")
	}

	for i, expected := range []string{"pod-0", "pod-1", "pod-2"} {
		action, obj, err := decoder.Decode()
		if err != nil {
			t.Fatalf("Decode(%d) returned unexpected error: %v", i, err)
		}
		if action != watch.Modified {
			t.Fatalf("Decode(%d) returned action %q, expected %q", i, action, watch.Modified)
		}
		pod, ok := obj.(*v1.Pod)
		if !ok {
			t.Fatalf("Decode(%d) returned %T, expected *v1.Pod", i, obj)
		}
		if pod.Name != expected {
			t.Fatalf("Decode(%d) returned pod %q, expected %q", i, pod.Name, expected)
		}
	}
}

func TestFrameDecoderReturnsEmbeddedDecodeErrorInOrder(t *testing.T) {
	decoder := NewFrameDecoder(
		newFakeFrameReader([][]byte{
			[]byte("pod-0"),
			[]byte("bad-pod"),
			[]byte("pod-2"),
		}),
		&delayedWatchEventDecoder{},
		&delayedObjectDecoder{
			delays: map[string]time.Duration{"pod-0": 100 * time.Millisecond},
			errors: map[string]error{"bad-pod": fmt.Errorf("decode bad-pod")},
		},
	)
	defer decoder.Close()

	action, obj, err := decoder.Decode()
	if err != nil {
		t.Fatalf("first Decode returned unexpected error: %v", err)
	}
	if action != watch.Modified {
		t.Fatalf("first Decode returned action %q, expected %q", action, watch.Modified)
	}
	if pod := obj.(*v1.Pod); pod.Name != "pod-0" {
		t.Fatalf("first Decode returned pod %q, expected pod-0", pod.Name)
	}

	_, _, err = decoder.Decode()
	if err == nil {
		t.Fatal("second Decode returned nil error")
	}
	if !strings.Contains(err.Error(), "decode bad-pod") {
		t.Fatalf("second Decode returned %q, expected decode bad-pod error", err)
	}
}

func TestFrameDecoderCloseUnblocksDecode(t *testing.T) {
	out, _ := io.Pipe()
	decoder := NewFrameDecoder(out, &delayedWatchEventDecoder{}, &delayedObjectDecoder{})

	done := make(chan error, 1)
	go func() {
		_, _, err := decoder.Decode()
		done <- err
	}()

	decoder.Close()

	select {
	case err := <-done:
		if err == nil {
			t.Fatal("Decode returned nil error after Close")
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatal("timed out waiting for Decode after Close")
	}
}

type fakeFrameReader struct {
	lock   sync.Mutex
	once   sync.Once
	frames [][]byte
	closed chan struct{}
}

func newFakeFrameReader(frames [][]byte) *fakeFrameReader {
	return &fakeFrameReader{
		frames: frames,
		closed: make(chan struct{}),
	}
}

func (r *fakeFrameReader) Read(dst []byte) (int, error) {
	r.lock.Lock()
	defer r.lock.Unlock()
	select {
	case <-r.closed:
		return 0, io.ErrClosedPipe
	default:
	}
	if len(r.frames) == 0 {
		return 0, io.EOF
	}
	frame := r.frames[0]
	if len(frame) > len(dst) {
		copy(dst, frame[:len(dst)])
		r.frames[0] = frame[len(dst):]
		return len(dst), io.ErrShortBuffer
	}
	r.frames = r.frames[1:]
	return copy(dst, frame), nil
}

func (r *fakeFrameReader) Close() error {
	r.once.Do(func() {
		close(r.closed)
	})
	return nil
}

type delayedWatchEventDecoder struct {
	delays    map[string]time.Duration
	errors    map[string]error
	completed chan<- string
}

func (d *delayedWatchEventDecoder) Decode(data []byte, _ *schema.GroupVersionKind, into runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	name := string(data)
	if delay := d.delays[name]; delay > 0 {
		time.Sleep(delay)
	}
	if d.completed != nil {
		d.completed <- name
	}
	if err := d.errors[name]; err != nil {
		return nil, nil, err
	}
	target, ok := into.(*metav1.WatchEvent)
	if !ok {
		return nil, nil, fmt.Errorf("expected *metav1.WatchEvent, got %T", into)
	}
	target.Type = string(watch.Modified)
	target.Object = runtime.RawExtension{Raw: []byte(name)}
	return target, nil, nil
}

type delayedObjectDecoder struct {
	delays map[string]time.Duration
	errors map[string]error
}

func (d *delayedObjectDecoder) Decode(data []byte, _ *schema.GroupVersionKind, _ runtime.Object) (runtime.Object, *schema.GroupVersionKind, error) {
	name := string(data)
	if delay := d.delays[name]; delay > 0 {
		time.Sleep(delay)
	}
	if err := d.errors[name]; err != nil {
		return nil, nil, err
	}
	return &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}}, nil, nil
}

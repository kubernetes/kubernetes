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

package framer

import (
	"bytes"
	"io"
	"strings"
	"testing"
)

// drainAllFrames reads every frame from r into a buffer that starts at
// initBuf bytes and doubles on io.ErrShortBuffer, mirroring
// streaming.Decoder.ReadFrame's resize policy.
func drainAllFrames(b *testing.B, r io.ReadCloser, initBuf int) int {
	buf := make([]byte, initBuf)
	used := 0
	frames := 0
	for {
		n, err := r.Read(buf[used:])
		used += n
		switch err {
		case nil:
			frames++
			used = 0
		case io.ErrShortBuffer:
			if used == len(buf) {
				grown := make([]byte, len(buf)*2)
				copy(grown, buf[:used])
				buf = grown
			}
		case io.EOF:
			return frames
		default:
			b.Fatalf("read: %v", err)
		}
	}
}

// drainAllFixed reads every frame from r into a fixed-size buffer that is
// guaranteed to fit any frame (no resizing).
func drainAllFixed(b *testing.B, r io.ReadCloser, bufSize int) int {
	buf := make([]byte, bufSize)
	frames := 0
	for {
		_, err := r.Read(buf)
		switch err {
		case nil:
			frames++
		case io.EOF:
			return frames
		default:
			b.Fatalf("read: %v", err)
		}
	}
}

// repeatFrames concatenates `frame` `count` times with `\n` separators.
func repeatFrames(frame string, count int) []byte {
	out := bytes.NewBuffer(make([]byte, 0, (len(frame)+1)*count))
	for i := 0; i < count; i++ {
		out.WriteString(frame)
		out.WriteByte('\n')
	}
	return out.Bytes()
}

const smallFrame = `{"type":"MODIFIED","object":{"kind":"Pod","metadata":{"name":"p","resourceVersion":"123"}}}`

func BenchmarkJSONFrameReaderSmall(b *testing.B) {
	stream := repeatFrames(smallFrame, 10000)
	wantFrames := 10000
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewJSONFramedReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFixed(b, r, 16<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

// largeFrame builds a ~50KB Pod-shaped JSON object dominated by nested
// objects/strings, mimicking a watch event for a Pod with long
// managedFields entries.
func largeFrame() string {
	var b strings.Builder
	b.Grow(50 << 10)
	b.WriteString(`{"type":"MODIFIED","object":{"kind":"Pod","apiVersion":"v1","metadata":{"name":"benchpod","namespace":"default","resourceVersion":"99999","uid":"00000000-0000-0000-0000-000000000000","managedFields":[`)
	for i := 0; i < 32; i++ {
		if i > 0 {
			b.WriteByte(',')
		}
		b.WriteString(`{"manager":"kube-controller-manager","operation":"Update","apiVersion":"v1","time":"2026-01-01T00:00:00Z","fieldsType":"FieldsV1","fieldsV1":{"f:metadata":{"f:labels":{".":{},"f:app":{},"f:tier":{},"f:version":{}},"f:annotations":{".":{},"f:checksum":{},"f:owner":{}}},"f:spec":{"f:containers":{"k:{\"name\":\"main\"}":{".":{},"f:image":{},"f:ports":{},"f:resources":{}}}},"f:status":{"f:conditions":{"k:{\"type\":\"Ready\"}":{".":{},"f:lastTransitionTime":{},"f:status":{},"f:type":{}}}}}}`)
	}
	b.WriteString(`]},"spec":{"containers":[{"name":"main","image":"nginx:1.25","ports":[{"containerPort":80}],"resources":{"limits":{"cpu":"500m","memory":"512Mi"},"requests":{"cpu":"100m","memory":"128Mi"}}}]},"status":{"phase":"Running","conditions":[{"type":"Ready","status":"True","lastTransitionTime":"2026-01-01T00:00:00Z"}]}}}`)
	return b.String()
}

func BenchmarkJSONFrameReaderLarge(b *testing.B) {
	frame := largeFrame()
	stream := repeatFrames(frame, 200)
	wantFrames := 200
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewJSONFramedReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFixed(b, r, 64<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

// BenchmarkJSONFrameReaderShortBuffer measures the realistic resize path
// that streaming.Decoder hits on cold start: 1 KB buffer that grows on
// io.ErrShortBuffer until each frame fits.
func BenchmarkJSONFrameReaderShortBuffer(b *testing.B) {
	frame := largeFrame()
	stream := repeatFrames(frame, 50)
	wantFrames := 50
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewJSONFramedReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFrames(b, r, 1<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

// BenchmarkLineDelimitedFrameReaderSmall mirrors
// BenchmarkJSONFrameReaderSmall so callers can compare the fast-path line
// reader against the JSON-aware state machine for the watch-style workload
// it's designed for (compact JSON + `\n`).
func BenchmarkLineDelimitedFrameReaderSmall(b *testing.B) {
	stream := repeatFrames(smallFrame, 10000)
	wantFrames := 10000
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewLineDelimitedFrameReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFixed(b, r, 16<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

func BenchmarkLineDelimitedFrameReaderLarge(b *testing.B) {
	frame := largeFrame()
	stream := repeatFrames(frame, 200)
	wantFrames := 200
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewLineDelimitedFrameReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFixed(b, r, 64<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

func BenchmarkLineDelimitedFrameReaderShortBuffer(b *testing.B) {
	frame := largeFrame()
	stream := repeatFrames(frame, 50)
	wantFrames := 50
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewLineDelimitedFrameReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFrames(b, r, 1<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

// BenchmarkJSONFrameReaderEscapeHeavy exercises the in-string hot loop
// with payloads dominated by escaped quotes (e.g., embedded JSON-in-JSON
// in managedFields.fieldsV1 keys).
func BenchmarkJSONFrameReaderEscapeHeavy(b *testing.B) {
	var sb strings.Builder
	sb.WriteString(`{"key":"`)
	for i := 0; i < 1024; i++ {
		sb.WriteString(`\"escaped\\value\"`)
	}
	sb.WriteString(`"}`)
	stream := repeatFrames(sb.String(), 500)
	wantFrames := 500
	b.SetBytes(int64(len(stream)))
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := NewJSONFramedReader(io.NopCloser(bytes.NewReader(stream)))
		if got := drainAllFixed(b, r, 64<<10); got != wantFrames {
			b.Fatalf("frames: got %d want %d", got, wantFrames)
		}
	}
}

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

package tracing

import (
	"io"
	"time"
)

type TracedReader struct {
	next  io.Reader
	layer *layer
	span  *Span
}

func (r *TracedReader) Read(p []byte) (n int, err error) {
	start := r.span.clock.Now()
	n, err = r.next.Read(p)
	duration := r.span.clock.Since(start)
	r.layer.duration += duration
	r.layer.bytes += int64(n)
	r.layer.count++
	return n, err
}

type ReaderSpan struct {
	span      *Span
	layer     *layer
	start     time.Time
	prevLayer *layer
}

func (s *ReaderSpan) Done() {
	duration := s.span.clock.Since(s.start)
	s.layer.duration = duration
	if s.layer.child != nil {
		s.layer.childDuration = s.layer.child.duration
	}
}

type TracedWriter struct {
	next  io.Writer
	layer *layer
	span  *Span
}

func (w *TracedWriter) Write(p []byte) (n int, err error) {
	start := w.span.clock.Now()
	n, err = w.next.Write(p)
	duration := w.span.clock.Since(start)
	w.layer.duration += duration
	w.layer.bytes += int64(n)
	w.layer.count++
	return n, err
}

type WriterSpan struct {
	span      *Span
	layer     *layer
	start     time.Time
	prevLayer *layer
}

func (s *WriterSpan) End() {
	duration := s.span.clock.Since(s.start)
	s.layer.duration = duration
	if s.layer.child != nil {
		s.layer.childDuration = s.layer.child.duration
	}
}

func (s *WriterSpan) Done() {
	s.End()
}

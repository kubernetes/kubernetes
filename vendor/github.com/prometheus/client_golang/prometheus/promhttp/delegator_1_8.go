// Copyright 2017 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.8

package promhttp

import (
	"io"
	"net/http"
)

type pusherDelegator struct{ *responseWriterDelegator }

func (d pusherDelegator) Push(target string, opts *http.PushOptions) error {
	return d.ResponseWriter.(http.Pusher).Push(target, opts)
}

func init() {
	pickDelegator[pusher] = func(d *responseWriterDelegator) delegator { // 16
		return pusherDelegator{d}
	}
	pickDelegator[pusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 17
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.CloseNotifier
		}{d, pusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+flusher] = func(d *responseWriterDelegator) delegator { // 18
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Flusher
		}{d, pusherDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[pusher+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 19
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Flusher
			http.CloseNotifier
		}{d, pusherDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+hijacker] = func(d *responseWriterDelegator) delegator { // 20
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Hijacker
		}{d, pusherDelegator{d}, hijackerDelegator{d}}
	}
	pickDelegator[pusher+hijacker+closeNotifier] = func(d *responseWriterDelegator) delegator { // 21
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Hijacker
			http.CloseNotifier
		}{d, pusherDelegator{d}, hijackerDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+hijacker+flusher] = func(d *responseWriterDelegator) delegator { // 22
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Hijacker
			http.Flusher
		}{d, pusherDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[pusher+hijacker+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { //23
		return struct {
			*responseWriterDelegator
			http.Pusher
			http.Hijacker
			http.Flusher
			http.CloseNotifier
		}{d, pusherDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+readerFrom] = func(d *responseWriterDelegator) delegator { // 24
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
		}{d, pusherDelegator{d}, readerFromDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+closeNotifier] = func(d *responseWriterDelegator) delegator { // 25
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.CloseNotifier
		}{d, pusherDelegator{d}, readerFromDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+flusher] = func(d *responseWriterDelegator) delegator { // 26
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Flusher
		}{d, pusherDelegator{d}, readerFromDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 27
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Flusher
			http.CloseNotifier
		}{d, pusherDelegator{d}, readerFromDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+hijacker] = func(d *responseWriterDelegator) delegator { // 28
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Hijacker
		}{d, pusherDelegator{d}, readerFromDelegator{d}, hijackerDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+hijacker+closeNotifier] = func(d *responseWriterDelegator) delegator { // 29
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Hijacker
			http.CloseNotifier
		}{d, pusherDelegator{d}, readerFromDelegator{d}, hijackerDelegator{d}, closeNotifierDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+hijacker+flusher] = func(d *responseWriterDelegator) delegator { // 30
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Hijacker
			http.Flusher
		}{d, pusherDelegator{d}, readerFromDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}}
	}
	pickDelegator[pusher+readerFrom+hijacker+flusher+closeNotifier] = func(d *responseWriterDelegator) delegator { // 31
		return struct {
			*responseWriterDelegator
			http.Pusher
			io.ReaderFrom
			http.Hijacker
			http.Flusher
			http.CloseNotifier
		}{d, pusherDelegator{d}, readerFromDelegator{d}, hijackerDelegator{d}, flusherDelegator{d}, closeNotifierDelegator{d}}
	}
}

func newDelegator(w http.ResponseWriter, observeWriteHeaderFunc func(int)) delegator {
	d := &responseWriterDelegator{
		ResponseWriter:     w,
		observeWriteHeader: observeWriteHeaderFunc,
	}

	id := 0
	if _, ok := w.(http.CloseNotifier); ok {
		id += closeNotifier
	}
	if _, ok := w.(http.Flusher); ok {
		id += flusher
	}
	if _, ok := w.(http.Hijacker); ok {
		id += hijacker
	}
	if _, ok := w.(io.ReaderFrom); ok {
		id += readerFrom
	}
	if _, ok := w.(http.Pusher); ok {
		id += pusher
	}

	return pickDelegator[id](d)
}

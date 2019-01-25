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

// +build !go1.8

package promhttp

import (
	"io"
	"net/http"
)

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

	return pickDelegator[id](d)
}

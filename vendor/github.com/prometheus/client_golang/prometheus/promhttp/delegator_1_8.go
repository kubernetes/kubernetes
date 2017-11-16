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

// newDelegator handles the four different methods of upgrading a
// http.ResponseWriter to delegator.
func newDelegator(w http.ResponseWriter, observeWriteHeaderFunc func(int)) delegator {
	d := &responseWriterDelegator{
		ResponseWriter:     w,
		observeWriteHeader: observeWriteHeaderFunc,
	}

	_, cn := w.(http.CloseNotifier)
	_, fl := w.(http.Flusher)
	_, hj := w.(http.Hijacker)
	_, ps := w.(http.Pusher)
	_, rf := w.(io.ReaderFrom)

	// Check for the four most common combination of interfaces a
	// http.ResponseWriter might implement.
	switch {
	case cn && fl && hj && rf && ps:
		// All interfaces.
		return &fancyPushDelegator{
			fancyDelegator: &fancyDelegator{d},
			p:              &pushDelegator{d},
		}
	case cn && fl && hj && rf:
		// All interfaces, except http.Pusher.
		return &fancyDelegator{d}
	case ps:
		// Just http.Pusher.
		return &pushDelegator{d}
	}

	return d
}

type fancyPushDelegator struct {
	p *pushDelegator

	*fancyDelegator
}

func (f *fancyPushDelegator) Push(target string, opts *http.PushOptions) error {
	return f.p.Push(target, opts)
}

type pushDelegator struct {
	*responseWriterDelegator
}

func (f *pushDelegator) Push(target string, opts *http.PushOptions) error {
	return f.ResponseWriter.(http.Pusher).Push(target, opts)
}

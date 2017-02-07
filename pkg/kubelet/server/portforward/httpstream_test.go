/*
Copyright 2016 The Kubernetes Authors.

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

package portforward

import (
	"net/http"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/httpstream"
	"k8s.io/kubernetes/pkg/api"
)

func TestHTTPStreamReceived(t *testing.T) {
	tests := map[string]struct {
		port          string
		streamType    string
		expectedError string
	}{
		"missing port": {
			expectedError: `"port" header is required`,
		},
		"unable to parse port": {
			port:          "abc",
			expectedError: `unable to parse "abc" as a port: strconv.ParseUint: parsing "abc": invalid syntax`,
		},
		"negative port": {
			port:          "-1",
			expectedError: `unable to parse "-1" as a port: strconv.ParseUint: parsing "-1": invalid syntax`,
		},
		"missing stream type": {
			port:          "80",
			expectedError: `"streamType" header is required`,
		},
		"valid port with error stream": {
			port:       "80",
			streamType: "error",
		},
		"valid port with data stream": {
			port:       "80",
			streamType: "data",
		},
		"invalid stream type": {
			port:          "80",
			streamType:    "foo",
			expectedError: `invalid stream type "foo"`,
		},
	}
	for name, test := range tests {
		streams := make(chan httpstream.Stream, 1)
		f := httpStreamReceived(streams)
		stream := newFakeHttpStream()
		if len(test.port) > 0 {
			stream.headers.Set("port", test.port)
		}
		if len(test.streamType) > 0 {
			stream.headers.Set("streamType", test.streamType)
		}
		replySent := make(chan struct{})
		err := f(stream, replySent)
		close(replySent)
		if len(test.expectedError) > 0 {
			if err == nil {
				t.Errorf("%s: expected err=%q, but it was nil", name, test.expectedError)
			}
			if e, a := test.expectedError, err.Error(); e != a {
				t.Errorf("%s: expected err=%q, got %q", name, e, a)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
			continue
		}
		if s := <-streams; s != stream {
			t.Errorf("%s: expected stream %#v, got %#v", name, stream, s)
		}
	}
}

func TestGetStreamPair(t *testing.T) {
	timeout := make(chan time.Time)

	h := &httpStreamHandler{
		streamPairs: make(map[string]*httpStreamPair),
	}

	// test adding a new entry
	p, created := h.getStreamPair("1")
	if p == nil {
		t.Fatalf("unexpected nil pair")
	}
	if !created {
		t.Fatal("expected created=true")
	}
	if p.dataStream != nil {
		t.Errorf("unexpected non-nil data stream")
	}
	if p.errorStream != nil {
		t.Errorf("unexpected non-nil error stream")
	}

	// start the monitor for this pair
	monitorDone := make(chan struct{})
	go func() {
		h.monitorStreamPair(p, timeout)
		close(monitorDone)
	}()

	if !h.hasStreamPair("1") {
		t.Fatal("This should still be true")
	}

	// make sure we can retrieve an existing entry
	p2, created := h.getStreamPair("1")
	if created {
		t.Fatal("expected created=false")
	}
	if p != p2 {
		t.Fatalf("retrieving an existing pair: expected %#v, got %#v", p, p2)
	}

	// removed via complete
	dataStream := newFakeHttpStream()
	dataStream.headers.Set(api.StreamType, api.StreamTypeData)
	complete, err := p.add(dataStream)
	if err != nil {
		t.Fatalf("unexpected error adding data stream to pair: %v", err)
	}
	if complete {
		t.Fatalf("unexpected complete")
	}

	errorStream := newFakeHttpStream()
	errorStream.headers.Set(api.StreamType, api.StreamTypeError)
	complete, err = p.add(errorStream)
	if err != nil {
		t.Fatalf("unexpected error adding error stream to pair: %v", err)
	}
	if !complete {
		t.Fatal("unexpected incomplete")
	}

	// make sure monitorStreamPair completed
	<-monitorDone

	// make sure the pair was removed
	if h.hasStreamPair("1") {
		t.Fatal("expected removal of pair after both data and error streams received")
	}

	// removed via timeout
	p, created = h.getStreamPair("2")
	if !created {
		t.Fatal("expected created=true")
	}
	if p == nil {
		t.Fatal("expected p not to be nil")
	}
	monitorDone = make(chan struct{})
	go func() {
		h.monitorStreamPair(p, timeout)
		close(monitorDone)
	}()
	// cause the timeout
	close(timeout)
	// make sure monitorStreamPair completed
	<-monitorDone
	if h.hasStreamPair("2") {
		t.Fatal("expected stream pair to be removed")
	}
}

func TestRequestID(t *testing.T) {
	h := &httpStreamHandler{}

	s := newFakeHttpStream()
	s.headers.Set(api.StreamType, api.StreamTypeError)
	s.id = 1
	if e, a := "1", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}

	s.headers.Set(api.StreamType, api.StreamTypeData)
	s.id = 3
	if e, a := "1", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}

	s.id = 7
	s.headers.Set(api.PortForwardRequestIDHeader, "2")
	if e, a := "2", h.requestID(s); e != a {
		t.Errorf("expected %q, got %q", e, a)
	}
}

type fakeHttpStream struct {
	headers http.Header
	id      uint32
}

func newFakeHttpStream() *fakeHttpStream {
	return &fakeHttpStream{
		headers: make(http.Header),
	}
}

var _ httpstream.Stream = &fakeHttpStream{}

func (s *fakeHttpStream) Read(data []byte) (int, error) {
	return 0, nil
}

func (s *fakeHttpStream) Write(data []byte) (int, error) {
	return 0, nil
}

func (s *fakeHttpStream) Close() error {
	return nil
}

func (s *fakeHttpStream) Reset() error {
	return nil
}

func (s *fakeHttpStream) Headers() http.Header {
	return s.headers
}

func (s *fakeHttpStream) Identifier() uint32 {
	return s.id
}

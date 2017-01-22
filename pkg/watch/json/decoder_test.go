/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package json

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

func TestDecoder(t *testing.T) {
	table := []watch.EventType{watch.Added, watch.Deleted, watch.Modified, watch.Error}

	for _, eventType := range table {
		out, in := io.Pipe()
		decoder := NewDecoder(out, testapi.Default.Codec())

		expect := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		encoder := json.NewEncoder(in)
		go func() {
			data, err := testapi.Default.Codec().Encode(expect)
			if err != nil {
				t.Fatalf("Unexpected error %v", err)
			}
			if err := encoder.Encode(&WatchEvent{eventType, runtime.RawExtension{RawJSON: json.RawMessage(data)}}); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
			in.Close()
		}()

		done := make(chan struct{})
		go func() {
			action, got, err := decoder.Decode()
			if err != nil {
				t.Fatalf("Unexpected error %v", err)
			}
			if e, a := eventType, action; e != a {
				t.Errorf("Expected %v, got %v", e, a)
			}
			if e, a := expect, got; !api.Semantic.DeepDerivative(e, a) {
				t.Errorf("Expected %v, got %v", e, a)
			}
			t.Logf("Exited read")
			close(done)
		}()
		<-done

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

func pipeViaHTTP() (client io.ReadCloser, server io.Writer, cleanup func()) {
	done := make(chan struct{}, 1)
	serverChan := make(chan http.ResponseWriter)
	s := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		flusher, ok := w.(http.Flusher)
		if !ok {
			panic("not a flusher?")
		}
		w.Header().Set("Transfer-Encoding", "chunked")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
		serverChan <- w
		<-done
	}))

	resp, err := http.Get(s.URL)
	if err != nil {
		panic("unexpected error " + err.Error())
	}

	return resp.Body, <-serverChan, func() {
		close(done)
		s.Close()
	}
}

func TestDecoder_SourceClose(t *testing.T) {
	out, in := io.Pipe()
	decoder := NewDecoder(out, testapi.Default.Codec())
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
	case <-time.After(util.ForeverTestTimeout):
		t.Error("Timeout")
	}
}

func TestDecoder_DestClose(t *testing.T) {
	out, _, cleanup := pipeViaHTTP()
	defer cleanup()
	decoder := NewDecoder(out, testapi.Default.Codec())

	done := make(chan struct{})

	go func() {
		_, _, err := decoder.Decode()
		if err == nil {
			t.Errorf("Unexpected nil error")
		}
		close(done)
	}()

	out.Close()

	select {
	case <-done:
		break
	case <-time.After(util.ForeverTestTimeout):
		t.Error("Timeout")
	}
}

func TestDecoder_Blocked(t *testing.T) {
	out, in, cleanup := pipeViaHTTP()
	inFlusher := in.(http.Flusher)
	inCN := in.(http.CloseNotifier)
	defer cleanup()
	encoder := NewEncoder(in, testapi.Default.Codec())

	wrote := make(chan struct{}, 1)

	go func() {
		defer func() {
			if x := recover(); x != nil {
				close(wrote)
				t.Logf("Got panic: %v", x)
			}
		}()
		for {
			err := encoder.Encode(&watch.Event{watch.Added, &api.Pod{ObjectMeta: api.ObjectMeta{Name: strings.Repeat("a", 1024)}}})
			if err != nil {
				t.Errorf("Write error %v", err)
				return
			}
			select {
			case <-inCN.CloseNotify():
				t.Logf("Got close")
				return
			default:
				inFlusher.Flush()
				wrote <- struct{}{}
			}
		}
	}()

	count := 0
loop:
	for {
		select {
		case _, open := <-wrote:
			if !open {
				t.Logf("'wrote' channel closed")
				break loop
			}
			count++
			t.Logf("wrote %v", count)

		case <-time.After(5 * time.Second):
			t.Error("Timeout")
			break loop
		}
	}
	t.Logf("about to close")
	out.Close()
}

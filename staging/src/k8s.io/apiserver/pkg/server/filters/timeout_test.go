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

package filters

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/runtime"
)

type recorder struct {
	lock  sync.Mutex
	count int
}

func (r *recorder) Record() {
	r.lock.Lock()
	defer r.lock.Unlock()
	r.count++
}

func (r *recorder) Count() int {
	r.lock.Lock()
	defer r.lock.Unlock()
	return r.count
}

func newHandler(responseCh <-chan string, panicCh <-chan interface{}, writeErrCh chan<- error) http.HandlerFunc {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		select {
		case resp := <-responseCh:
			_, err := w.Write([]byte(resp))
			writeErrCh <- err
		case panicReason := <-panicCh:
			panic(panicReason)
		}
	})
}

func TestTimeout(t *testing.T) {
	origReallyCrash := runtime.ReallyCrash
	runtime.ReallyCrash = false
	defer func() {
		runtime.ReallyCrash = origReallyCrash
	}()

	sendResponse := make(chan string, 1)
	doPanic := make(chan interface{}, 1)
	writeErrors := make(chan error, 1)
	gotPanic := make(chan interface{}, 1)
	timeout := make(chan time.Time, 1)
	resp := "test response"
	timeoutErr := apierrors.NewServerTimeout(schema.GroupResource{Group: "foo", Resource: "bar"}, "get", 0)
	record := &recorder{}

	handler := newHandler(sendResponse, doPanic, writeErrors)
	ts := httptest.NewServer(withPanicRecovery(
		WithTimeout(handler, func(req *http.Request) (*http.Request, <-chan time.Time, func(), *apierrors.StatusError) {
			return req, timeout, record.Record, timeoutErr
		}), func(w http.ResponseWriter, req *http.Request, err interface{}) {
			gotPanic <- err
			http.Error(w, "This request caused apiserver to panic. Look in the logs for details.", http.StatusInternalServerError)
		}),
	)
	defer ts.Close()

	// No timeouts
	sendResponse <- resp
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusOK {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusOK)
	}
	body, _ := ioutil.ReadAll(res.Body)
	if string(body) != resp {
		t.Errorf("got body %q; expected %q", string(body), resp)
	}
	if err := <-writeErrors; err != nil {
		t.Errorf("got unexpected Write error on first request: %v", err)
	}
	if record.Count() != 0 {
		t.Errorf("invoked record method: %#v", record)
	}

	// Times out
	timeout <- time.Time{}
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusServiceUnavailable)
	}
	body, _ = ioutil.ReadAll(res.Body)
	status := &metav1.Status{}
	if err := json.Unmarshal(body, status); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(status, &timeoutErr.ErrStatus) {
		t.Errorf("unexpected object: %s", diff.ObjectReflectDiff(&timeoutErr.ErrStatus, status))
	}
	if record.Count() != 1 {
		t.Errorf("did not invoke record method: %#v", record)
	}

	// Now try to send a response
	sendResponse <- resp
	if err := <-writeErrors; err != http.ErrHandlerTimeout {
		t.Errorf("got Write error of %v; expected %v", err, http.ErrHandlerTimeout)
	}

	// Panics
	doPanic <- "inner handler panics"
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusInternalServerError {
		t.Errorf("got res.StatusCode %d; expected %d due to panic", res.StatusCode, http.StatusInternalServerError)
	}
	select {
	case err := <-gotPanic:
		msg := fmt.Sprintf("%v", err)
		if !strings.Contains(msg, "newHandler") {
			t.Errorf("expected line with root cause panic in the stack trace, but didn't: %v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("expected to see a handler panic, but didn't")
	}

	// Panics with http.ErrAbortHandler
	doPanic <- http.ErrAbortHandler
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Fatal(err)
	}
	if res.StatusCode != http.StatusInternalServerError {
		t.Errorf("got res.StatusCode %d; expected %d due to panic", res.StatusCode, http.StatusInternalServerError)
	}
	select {
	case err := <-gotPanic:
		if err != http.ErrAbortHandler {
			t.Errorf("expected unwrapped http.ErrAbortHandler, got %#v", err)
		}
	case <-time.After(30 * time.Second):
		t.Fatalf("expected to see a handler panic, but didn't")
	}
}

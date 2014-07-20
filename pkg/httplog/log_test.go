/*
Copyright 2014 Google Inc. All rights reserved.

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

package httplog

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestHandler(t *testing.T) {
	want := &httptest.ResponseRecorder{
		HeaderMap: make(http.Header),
		Body:      new(bytes.Buffer),
	}
	want.WriteHeader(http.StatusOK)
	mux := http.NewServeMux()
	handler := Handler(mux, DefaultStacktracePred)
	mux.HandleFunc("/kube", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	req, err := http.NewRequest("GET", "http://example.com/kube", nil)
	if err != nil {
		t.Errorf("Unexpected error: %#v", err)
	}
	w := httptest.NewRecorder()
	handler.ServeHTTP(w, req)
	if !reflect.DeepEqual(want, w) {
		t.Errorf("Expected %v, got %v", want, w)
	}
}

var statusTestTable = []struct {
	status   int
	statuses []int
	want     bool
}{
	{http.StatusOK, []int{}, true},
	{http.StatusOK, []int{http.StatusOK}, false},
	{http.StatusCreated, []int{http.StatusOK, http.StatusAccepted}, true},
}

func TestStatusIsNot(t *testing.T) {
	for _, tt := range statusTestTable {
		sp := StatusIsNot(tt.statuses...)
		got := sp(tt.status)
		if got != tt.want {
			t.Errorf("Expected %v, got %v", tt.want, got)
		}
	}
}

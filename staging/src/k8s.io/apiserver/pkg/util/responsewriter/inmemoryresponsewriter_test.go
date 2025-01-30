/*
Copyright 2024 The Kubernetes Authors.

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

package responsewriter

import (
	"net/http"
	"testing"
)

func TestInMemoryResponseWriter(t *testing.T) {
	w := NewInMemoryResponseWriter()

	h := w.Header()
	h.Set("Content-Type", "application/json")

	w.WriteHeader(http.StatusCreated)

	_, err := w.Write([]byte(`{"message": "hello"}`))
	if err != nil {
		t.Errorf("Write() returned an error: %v", err)
	}

	if w.RespCode() != http.StatusCreated {
		t.Errorf("RespCode() returned unexpected code: %d, want %d", w.RespCode(), http.StatusCreated)
	}

	if string(w.Data()) != `{"message": "hello"}` {
		t.Errorf("Data() returned unexpected body: %s, want %s", string(w.Data()), `{"message": "hello"}`)
	}

	expectedString := "ResponseCode: 201, Body: {\"message\": \"hello\"}, Header: map[Content-Type:[application/json]]"
	if w.String() != expectedString {
		t.Errorf("String() returned unexpected output: %s, want %s", w.String(), expectedString)
	}
}

func TestInMemoryResponseWriter_DefaultHeader(t *testing.T) {
	w := NewInMemoryResponseWriter()

	_, err := w.Write([]byte(`{"message": "hello"}`))
	if err != nil {
		t.Errorf("Write() returned an error: %v", err)
	}

	// should be StatusOK (200) by default
	if w.RespCode() != http.StatusOK {
		t.Errorf("RespCode() returned unexpected code: %d, want %d", w.RespCode(), http.StatusOK)
	}
}

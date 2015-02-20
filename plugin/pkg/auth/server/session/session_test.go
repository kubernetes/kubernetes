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

package session

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestSession(t *testing.T) {
	store := NewStore("secret")
	req, err := http.NewRequest("GET", "/", nil)
	if err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}

	// Get session
	session, err := store.Get(req, "session name")
	if err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}
	values := session.Values()
	foo, ok := values["Foo"]
	if ok {
		t.Fatalf("Unexpected success")
	}

	// Set value
	values["Foo"] = "Bar"

	// Get session again
	session, err = store.Get(req, "session name")
	if err != nil {
		t.Fatalf("Unexpected error: %s", err)
	}
	values = session.Values()
	foo, ok = values["Foo"]
	if !ok {
		t.Fatalf("Unexpected error")
	}
	if foo != "Bar" {
		t.Fatalf("Expected \"Bar\", got \"%v\"", foo)
	}

	// Save
	w := httptest.NewRecorder()
	if err := store.Save(w, req); err != nil {
		t.Fatalf("Unexpected error")
	}
	if _, ok := w.HeaderMap["Set-Cookie"]; !ok {
		t.Fatalf("Expected Set-Cookie header, got none")
	}
}

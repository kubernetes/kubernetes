/*
Copyright 2015 The Kubernetes Authors.

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

package configz

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestConfigz(t *testing.T) {
	v, err := New("testing")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	v.Set("blah")

	s := httptest.NewServer(http.HandlerFunc(handle))
	defer s.Close()

	resp, err := http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{"testing":"blah"}` {
		t.Fatalf("unexpected output: %v", err)
	}

	v.Set("bing")
	resp, err = http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{"testing":"bing"}` {
		t.Fatalf("unexpected output: %v", err)
	}

	Delete("testing")
	resp, err = http.Get(s.URL + "/configz")
	if err != nil {
		t.Fatalf("err: %v", err)
	}

	body, err = ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if string(body) != `{}` {
		t.Fatalf("unexpected output: %v", err)
	}
}

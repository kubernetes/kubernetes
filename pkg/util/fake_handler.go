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
package util

import (
	"io/ioutil"
	"log"
	"net/http"
	"testing"
)

// FakeHandler is to assist in testing HTTP requests.
type FakeHandler struct {
	RequestReceived *http.Request
	StatusCode      int
	ResponseBody    string
}

func (f *FakeHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	f.RequestReceived = request
	response.WriteHeader(f.StatusCode)
	response.Write([]byte(f.ResponseBody))

	bodyReceived, err := ioutil.ReadAll(request.Body)
	if err != nil {
		log.Printf("Received read error: %#v", err)
	}
	f.ResponseBody = string(bodyReceived)
}

func (f FakeHandler) ValidateRequest(t *testing.T, expectedPath, expectedMethod string, body *string) {
	if f.RequestReceived.URL.Path != expectedPath {
		t.Errorf("Unexpected request path: %s", f.RequestReceived.URL.Path)
	}
	if f.RequestReceived.Method != expectedMethod {
		t.Errorf("Unexpected method: %s", f.RequestReceived.Method)
	}
	if body != nil {
		if *body != f.ResponseBody {
			t.Errorf("Received body:\n%s\n Doesn't match expected body:\n%s", f.ResponseBody, *body)
		}
	}
}

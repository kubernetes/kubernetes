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

package kubecfg

import (
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestFileServing(t *testing.T) {
	data := "This is test data"
	dir, err := ioutil.TempDir("", "data")
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	err = ioutil.WriteFile(dir+"/test.txt", []byte(data), 0755)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	prefix := "/foo/"
	handler := newFileHandler(prefix, dir)
	server := httptest.NewServer(handler)
	client := http.Client{}
	req, err := http.NewRequest("GET", server.URL+prefix+"test.txt", nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	res, err := client.Do(req)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	defer res.Body.Close()
	b, err := ioutil.ReadAll(res.Body)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if res.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d", res.StatusCode)
	}
	if string(b) != data {
		t.Errorf("Data doesn't match: %s vs %s", string(b), data)
	}
}

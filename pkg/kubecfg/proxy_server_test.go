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
	expectNoError(t, err)
	err = ioutil.WriteFile(dir+"/test.txt", []byte(data), 0755)
	expectNoError(t, err)
	handler := fileServer{
		prefix: "/foo/",
		base:   dir,
	}
	server := httptest.NewServer(&handler)
	client := http.Client{}
	req, err := http.NewRequest("GET", server.URL+handler.prefix+"/test.txt", nil)
	expectNoError(t, err)
	res, err := client.Do(req)
	expectNoError(t, err)
	defer res.Body.Close()
	b, err := ioutil.ReadAll(res.Body)
	expectNoError(t, err)
	if res.StatusCode != http.StatusOK {
		t.Errorf("Unexpected status: %d", res.StatusCode)
	}
	if string(b) != data {
		t.Errorf("Data doesn't match: %s vs %s", string(b), data)
	}
}

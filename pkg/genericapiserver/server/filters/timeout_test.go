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
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"
)

func TestTimeout(t *testing.T) {
	sendResponse := make(chan struct{}, 1)
	writeErrors := make(chan error, 1)
	timeout := make(chan time.Time, 1)
	resp := "test response"
	timeoutResp := "test timeout"

	ts := httptest.NewServer(WithTimeout(http.HandlerFunc(
		func(w http.ResponseWriter, r *http.Request) {
			<-sendResponse
			_, err := w.Write([]byte(resp))
			writeErrors <- err
		}),
		func(*http.Request) (<-chan time.Time, string) {
			return timeout, timeoutResp
		}))
	defer ts.Close()

	// No timeouts
	sendResponse <- struct{}{}
	res, err := http.Get(ts.URL)
	if err != nil {
		t.Error(err)
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

	// Times out
	timeout <- time.Time{}
	res, err = http.Get(ts.URL)
	if err != nil {
		t.Error(err)
	}
	if res.StatusCode != http.StatusGatewayTimeout {
		t.Errorf("got res.StatusCode %d; expected %d", res.StatusCode, http.StatusServiceUnavailable)
	}
	body, _ = ioutil.ReadAll(res.Body)
	if string(body) != timeoutResp {
		t.Errorf("got body %q; expected %q", string(body), timeoutResp)
	}

	// Now try to send a response
	sendResponse <- struct{}{}
	if err := <-writeErrors; err != http.ErrHandlerTimeout {
		t.Errorf("got Write error of %v; expected %v", err, http.ErrHandlerTimeout)
	}
}

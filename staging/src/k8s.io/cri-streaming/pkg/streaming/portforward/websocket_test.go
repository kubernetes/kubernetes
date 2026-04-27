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

package portforward

import (
	"net/http"
	"reflect"
	"testing"
)

func TestV4Options(t *testing.T) {
	tests := map[string]struct {
		url           string
		websocket     bool
		expectedOpts  *V4Options
		expectedError string
	}{
		"non-ws request": {
			url:          "http://example.com",
			expectedOpts: &V4Options{},
		},
		"missing port": {
			url:           "http://example.com",
			websocket:     true,
			expectedError: `query parameter "port" is required`,
		},
		"unable to parse port": {
			url:           "http://example.com?port=abc",
			websocket:     true,
			expectedError: `unable to parse "abc" as a port: strconv.ParseUint: parsing "abc": invalid syntax`,
		},
		"negative port": {
			url:           "http://example.com?port=-1",
			websocket:     true,
			expectedError: `unable to parse "-1" as a port: strconv.ParseUint: parsing "-1": invalid syntax`,
		},
		"one port": {
			url:       "http://example.com?port=80",
			websocket: true,
			expectedOpts: &V4Options{
				Ports: []int32{80},
			},
		},
		"multiple ports": {
			url:       "http://example.com?port=80,90,100",
			websocket: true,
			expectedOpts: &V4Options{
				Ports: []int32{80, 90, 100},
			},
		},
		"multiple port": {
			url:       "http://example.com?port=80&port=90",
			websocket: true,
			expectedOpts: &V4Options{
				Ports: []int32{80, 90},
			},
		},
	}
	for name, test := range tests {
		req, err := http.NewRequest(http.MethodGet, test.url, nil)
		if err != nil {
			t.Errorf("%s: invalid url %q err=%q", name, test.url, err)
			continue
		}
		if test.websocket {
			req.Header.Set("Connection", "Upgrade")
			req.Header.Set("Upgrade", "websocket")
		}
		opts, err := NewV4Options(req)
		if len(test.expectedError) > 0 {
			if err == nil {
				t.Errorf("%s: expected err=%q, but it was nil", name, test.expectedError)
			}
			if e, a := test.expectedError, err.Error(); e != a {
				t.Errorf("%s: expected err=%q, got %q", name, e, a)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: unexpected error %v", name, err)
			continue
		}
		if !reflect.DeepEqual(test.expectedOpts, opts) {
			t.Errorf("%s: expected options %#v, got %#v", name, test.expectedOpts, err)
		}
	}
}

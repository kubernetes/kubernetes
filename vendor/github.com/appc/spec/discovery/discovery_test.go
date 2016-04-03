// Copyright 2015 The appc Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package discovery

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/appc/spec/schema/types"
)

func fakeHTTPGet(filename string, failures int, header http.Header) func(req *http.Request) (*http.Response, error) {
	attempts := 0
	return func(req *http.Request) (*http.Response, error) {
		f, err := os.Open(filename)
		if err != nil {
			return nil, err
		}

		var resp *http.Response

		if header != nil && !reflect.DeepEqual(req.Header, header) {
			err = fmt.Errorf("fakeHTTPGet: wrong header %v. Expected %v", req.Header, header)
			return nil, err
		}

		switch {
		case attempts < failures:
			resp = &http.Response{
				Status:     "404 Not Found",
				StatusCode: http.StatusNotFound,
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				Header: http.Header{
					"Content-Type": []string{"text/html"},
				},
				Body: ioutil.NopCloser(bytes.NewBufferString("")),
			}
		default:
			resp = &http.Response{
				Status:     "200 OK",
				StatusCode: http.StatusOK,
				Proto:      "HTTP/1.1",
				ProtoMajor: 1,
				ProtoMinor: 1,
				Header: http.Header{
					"Content-Type": []string{"text/html"},
				},
				Body: f,
			}
		}

		attempts = attempts + 1
		return resp, nil
	}
}

func TestDiscoverEndpoints(t *testing.T) {
	tests := []struct {
		do                     httpDoer
		expectDiscoverySuccess bool
		app                    App
		expectedACIEndpoints   []ACIEndpoint
		expectedKeys           []string
		authHeader             http.Header
	}{
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp.html", 0, nil),
			},
			true,
			App{
				Name: "example.com/myapp",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
					"os":      "linux",
					"arch":    "amd64",
				},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp-1.0.0.aci?torrent",
					ASC: "https://storage.example.com/example.com/myapp-1.0.0.aci.asc?torrent",
				},
				ACIEndpoint{
					ACI: "hdfs://storage.example.com/example.com/myapp-1.0.0.aci",
					ASC: "hdfs://storage.example.com/example.com/myapp-1.0.0.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			nil,
		},
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp.html", 1, nil),
			},
			true,
			App{
				Name: "example.com/myapp/foobar",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
					"os":      "linux",
					"arch":    "amd64",
				},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp/foobar-1.0.0.aci?torrent",
					ASC: "https://storage.example.com/example.com/myapp/foobar-1.0.0.aci.asc?torrent",
				},
				ACIEndpoint{
					ACI: "hdfs://storage.example.com/example.com/myapp/foobar-1.0.0.aci",
					ASC: "hdfs://storage.example.com/example.com/myapp/foobar-1.0.0.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			nil,
		},
		{
			&mockHttpDoer{
				// always fails
				doer: fakeHTTPGet("myapp.html", 10000, nil),
			},
			false,
			App{
				Name: "example.com/myapp/foobar/bazzer",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
					"os":      "linux",
					"arch":    "amd64",
				},
			},
			[]ACIEndpoint{},
			[]string{},
			nil,
		},
		// Test missing label. Only one ac-discovery template should be
		// returned as the other one cannot be completely rendered due to
		// missing labels.
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp2.html", 0, nil),
			},
			true,
			App{
				Name: "example.com/myapp",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
				},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp-1.0.0.aci",
					ASC: "https://storage.example.com/example.com/myapp-1.0.0.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			nil,
		},
		// Test missing labels. version label should default to
		// "latest" and the first template should be rendered
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp2.html", 0, nil),
			},
			true,
			App{
				Name:   "example.com/myapp",
				Labels: map[types.ACIdentifier]string{},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp-latest.aci",
					ASC: "https://storage.example.com/example.com/myapp-latest.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			nil,
		},
		// Test with a label called "name". It should be ignored.
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp2.html", 0, nil),
			},
			true,
			App{
				Name: "example.com/myapp",
				Labels: map[types.ACIdentifier]string{
					"name":    "labelcalledname",
					"version": "1.0.0",
				},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp-1.0.0.aci",
					ASC: "https://storage.example.com/example.com/myapp-1.0.0.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			nil,
		},
		// Test with an auth header
		{
			&mockHttpDoer{
				doer: fakeHTTPGet("myapp.html", 0, testAuthHeader),
			},
			true,
			App{
				Name: "example.com/myapp",
				Labels: map[types.ACIdentifier]string{
					"version": "1.0.0",
					"os":      "linux",
					"arch":    "amd64",
				},
			},
			[]ACIEndpoint{
				ACIEndpoint{
					ACI: "https://storage.example.com/example.com/myapp-1.0.0.aci?torrent",
					ASC: "https://storage.example.com/example.com/myapp-1.0.0.aci.asc?torrent",
				},
				ACIEndpoint{
					ACI: "hdfs://storage.example.com/example.com/myapp-1.0.0.aci",
					ASC: "hdfs://storage.example.com/example.com/myapp-1.0.0.aci.asc",
				},
			},
			[]string{"https://example.com/pubkeys.gpg"},
			testAuthHeader,
		},
	}

	for i, tt := range tests {
		httpDo = tt.do
		httpDoInsecureTls = tt.do
		var hostHeaders map[string]http.Header
		if tt.authHeader != nil {
			hostHeaders = map[string]http.Header{
				strings.Split(tt.app.String(), "/")[0]: tt.authHeader,
			}
		}
		insecureList := []InsecureOption{
			InsecureNone,
			InsecureTls,
			InsecureHttp,
			InsecureTls | InsecureHttp,
		}
		for _, insecure := range insecureList {
			de, _, err := DiscoverEndpoints(tt.app, hostHeaders, insecure)
			if err != nil && !tt.expectDiscoverySuccess {
				continue
			}
			if err == nil && !tt.expectDiscoverySuccess {
				t.Fatalf("#%d DiscoverEndpoints should have failed but didn't", i)
			}
			if err != nil {
				t.Fatalf("#%d DiscoverEndpoints failed: %v", i, err)
			}

			if len(de.ACIEndpoints) != len(tt.expectedACIEndpoints) {
				t.Errorf("ACIEndpoints array is wrong length want %d got %d", len(tt.expectedACIEndpoints), len(de.ACIEndpoints))
			} else {
				for n, _ := range de.ACIEndpoints {
					if de.ACIEndpoints[n] != tt.expectedACIEndpoints[n] {
						t.Errorf("#%d ACIEndpoints[%d] mismatch: want %v got %v", i, n, tt.expectedACIEndpoints[n], de.ACIEndpoints[n])
					}
				}
			}

			if len(de.Keys) != len(tt.expectedKeys) {
				t.Errorf("Keys array is wrong length want %d got %d", len(tt.expectedKeys), len(de.Keys))
			} else {
				for n, _ := range de.Keys {
					if de.Keys[n] != tt.expectedKeys[n] {
						t.Errorf("#%d sig[%d] mismatch: want %v got %v", i, n, tt.expectedKeys[n], de.Keys[n])
					}
				}
			}
		}
	}
}

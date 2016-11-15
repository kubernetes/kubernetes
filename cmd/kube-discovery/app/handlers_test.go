/*
Copyright 2014 The Kubernetes Authors.

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

package discovery

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/square/go-jose"
)

type mockTokenLoader struct {
	tokenID string
	token   string
}

func (tl *mockTokenLoader) LoadAndLookup(tokenID string) (string, error) {
	if tokenID == tl.tokenID {
		return tl.token, nil
	}
	return "", errors.New(fmt.Sprintf("invalid token: %s", tokenID))
}

const mockEndpoint1 = "https://192.168.1.5:8080"
const mockEndpoint2 = "https://192.168.1.6:8080"

type mockEndpointsLoader struct {
}

func (el *mockEndpointsLoader) LoadList() ([]string, error) {
	return []string{mockEndpoint1, mockEndpoint2}, nil
}

const mockCA = "---BEGIN------END---DUMMYDATA"

type mockCALoader struct {
}

func (cl *mockCALoader) LoadPEM() (string, error) {
	return mockCA, nil
}

const mockTokenID = "AAAAAA"
const mockToken = "9537434E638E4378"

const mockTokenIDCustom = "SHAREDSECRET"
const mockTokenCustom = "VERYSECRETTOKEN"

func TestClusterInfoIndex(t *testing.T) {
	longToken := strings.Repeat("a", 1000)
	tests := map[string]struct {
		tokenID          string // token ID the mock loader will use
		token            string // token the mock loader will use
		reqTokenID       string // token ID the will request with
		reqToken         string // token the caller will validate response with
		expStatus        int
		expVerifyFailure bool
	}{
		"no token": {
			tokenID:    mockTokenID,
			token:      mockToken,
			reqTokenID: "",
			reqToken:   "",
			expStatus:  http.StatusForbidden,
		},
		"valid token ID": {
			tokenID:    mockTokenID,
			token:      mockToken,
			reqTokenID: mockTokenID,
			reqToken:   mockToken,
			expStatus:  http.StatusOK,
		},
		"valid arbitrary string token": {
			tokenID:    mockTokenIDCustom,
			token:      mockTokenCustom,
			reqTokenID: mockTokenIDCustom,
			reqToken:   mockTokenCustom,
			expStatus:  http.StatusOK,
		},
		"valid arbitrary long string token": {
			tokenID:    "LONGTOKENTEST",
			token:      longToken,
			reqTokenID: "LONGTOKENTEST",
			reqToken:   longToken,
			expStatus:  http.StatusOK,
		},
		"invalid token ID": {
			tokenID:    mockTokenID,
			token:      mockToken,
			reqTokenID: "BADTOKENID",
			reqToken:   mockToken,
			expStatus:  http.StatusForbidden,
		},
		"invalid token": {
			tokenID:          mockTokenID,
			token:            mockToken,
			reqTokenID:       mockTokenID,
			reqToken:         "badtoken",
			expStatus:        http.StatusOK,
			expVerifyFailure: true,
		},
	}

	for name, test := range tests {
		t.Logf("Running test: %s", name)
		tokenLoader := &mockTokenLoader{test.tokenID, test.token}
		// Create a request to pass to our handler. We don't have any query parameters for now, so we'll
		// pass 'nil' as the third parameter.
		url := "/cluster-info/v1/"
		if test.tokenID != "" {
			url = fmt.Sprintf("%s?token-id=%s", url, test.reqTokenID)
		}
		req, err := http.NewRequest("GET", url, nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		handler := &ClusterInfoHandler{
			tokenLoader:     tokenLoader,
			caLoader:        &mockCALoader{},
			endpointsLoader: &mockEndpointsLoader{},
		}

		handler.ServeHTTP(rr, req)

		if status := rr.Code; status != test.expStatus {
			t.Errorf("handler returned wrong status code: got %v want %v",
				status, test.expStatus)
			continue
		}

		// If we were expecting valid status validate the body:
		if test.expStatus == http.StatusOK {
			var ci ClusterInfo

			body := string(rr.Body.Bytes())

			// Parse the JSON web signature:
			jws, err := jose.ParseSigned(body)
			if err != nil {
				t.Errorf("Error parsing JWS from request body: %s", err)
				continue
			}

			// Now we can verify the signature on the payload. An error here would
			// indicate the the message failed to verify, e.g. because the signature was
			// broken or the message was tampered with.
			var clusterInfoBytes []byte
			hmacTestKey := []byte(test.reqToken)
			clusterInfoBytes, err = jws.Verify(hmacTestKey)

			if test.expVerifyFailure {
				if err == nil {
					t.Errorf("Signature verification did not fail as expected.")
				}
				// We are done the test here either way.
				continue
			}

			if err != nil {
				t.Errorf("Error verifing signature: %s", err)
				continue
			}

			err = json.Unmarshal(clusterInfoBytes, &ci)
			if err != nil {
				t.Errorf("Unable to unmarshall payload to JSON: error=%s body=%s", err, rr.Body.String())
				continue
			}
			if len(ci.Endpoints) != 2 {
				t.Errorf("Expected 2 endpoints, got: %d", len(ci.Endpoints))
			}
			if mockEndpoint1 != ci.Endpoints[0] {
				t.Errorf("Unexpected endpoint: %s", ci.Endpoints[0])
			}
			if mockEndpoint2 != ci.Endpoints[1] {
				t.Errorf("Unexpected endpoint: %s", ci.Endpoints[1])
			}

			if len(ci.CertificateAuthorities) != 1 {
				t.Errorf("Expected 1 root certificate, got: %d", len(ci.CertificateAuthorities))
			}
			if ci.CertificateAuthorities[0] != mockCA {
				t.Errorf("Expected CA: %s, got: %s", mockCA, ci.CertificateAuthorities[0])
			}
		}
	}
}

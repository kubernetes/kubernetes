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
package kubediscovery

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/square/go-jose"
)

func TestClusterInfoIndex(t *testing.T) {
	tests := map[string]struct {
		url       string
		expStatus int
	}{
		"no token": {
			"/cluster-info/v1/",
			http.StatusForbidden,
		},
		"valid token": {
			fmt.Sprintf("/cluster-info/v1/?token-id=%s", tempTokenId),
			http.StatusOK,
		},
		"invalid token": {
			"/cluster-info/v1/?token-id=JUNK",
			http.StatusForbidden,
		},
	}

	for name, test := range tests {
		t.Logf("Running test: %s", name)
		// Create a request to pass to our handler. We don't have any query parameters for now, so we'll
		// pass 'nil' as the third parameter.
		req, err := http.NewRequest("GET", test.url, nil)
		if err != nil {
			t.Fatal(err)
		}

		rr := httptest.NewRecorder()
		// TODO: mock/stub here
		handler := NewClusterInfoHandler()

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
			hmacTestKey := fromHexBytes(tempToken)
			clusterInfoBytes, err = jws.Verify(hmacTestKey)
			if err != nil {
				t.Errorf("Error verifing signature: %s", err)
				continue
			}

			err = json.Unmarshal(clusterInfoBytes, &ci)
			if err != nil {
				t.Errorf("Unable to unmarshall payload to JSON: error=%s body=%s", err, rr.Body.String())
				continue
			}
			if ci.RootCertificates == "" {
				t.Error("No root certificates in response")
				continue
			}
		}
	}
}

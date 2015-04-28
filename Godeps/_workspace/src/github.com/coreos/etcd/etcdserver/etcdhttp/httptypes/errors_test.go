// Copyright 2015 CoreOS, Inc.
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

package httptypes

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
)

func TestHTTPErrorWriteTo(t *testing.T) {
	err := NewHTTPError(http.StatusBadRequest, "what a bad request you made!")
	rr := httptest.NewRecorder()
	err.WriteTo(rr)

	wcode := http.StatusBadRequest
	wheader := http.Header(map[string][]string{
		"Content-Type": []string{"application/json"},
	})
	wbody := `{"message":"what a bad request you made!"}`

	if wcode != rr.Code {
		t.Errorf("HTTP status code %d, want %d", rr.Code, wcode)
	}

	if !reflect.DeepEqual(wheader, rr.HeaderMap) {
		t.Errorf("HTTP headers %v, want %v", rr.HeaderMap, wheader)
	}

	gbody := rr.Body.String()
	if wbody != gbody {
		t.Errorf("HTTP body %q, want %q", gbody, wbody)
	}
}

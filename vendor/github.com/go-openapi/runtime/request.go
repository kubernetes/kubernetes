// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package runtime

import (
	"io"
	"net/http"
	"strings"

	"github.com/go-openapi/swag"
)

// CanHaveBody returns true if this method can have a body
func CanHaveBody(method string) bool {
	mn := strings.ToUpper(method)
	return mn == "POST" || mn == "PUT" || mn == "PATCH" || mn == "DELETE"
}

// IsSafe returns true if this is a request with a safe method
func IsSafe(r *http.Request) bool {
	mn := strings.ToUpper(r.Method)
	return mn == "GET" || mn == "HEAD"
}

// AllowsBody returns true if the request allows for a body
func AllowsBody(r *http.Request) bool {
	mn := strings.ToUpper(r.Method)
	return mn != "HEAD"
}

// HasBody returns true if this method needs a content-type
func HasBody(r *http.Request) bool {
	return len(r.TransferEncoding) > 0 || r.ContentLength > 0
}

// JSONRequest creates a new http request with json headers set
func JSONRequest(method, urlStr string, body io.Reader) (*http.Request, error) {
	req, err := http.NewRequest(method, urlStr, body)
	if err != nil {
		return nil, err
	}
	req.Header.Add(HeaderContentType, JSONMime)
	req.Header.Add(HeaderAccept, JSONMime)
	return req, nil
}

// Gettable for things with a method GetOK(string) (data string, hasKey bool, hasValue bool)
type Gettable interface {
	GetOK(string) ([]string, bool, bool)
}

// ReadSingleValue reads a single value from the source
func ReadSingleValue(values Gettable, name string) string {
	vv, _, hv := values.GetOK(name)
	if hv {
		return vv[len(vv)-1]
	}
	return ""
}

// ReadCollectionValue reads a collection value from a string data source
func ReadCollectionValue(values Gettable, name, collectionFormat string) []string {
	v := ReadSingleValue(values, name)
	return swag.SplitByFormat(v, collectionFormat)
}

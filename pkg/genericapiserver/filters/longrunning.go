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
	"net/http"
	"regexp"
	"strings"
)

// LongRunningRequestCheck is a predicate which is true for long-running http requests.
type LongRunningRequestCheck func(r *http.Request) bool

// BasicLongRunningRequestCheck pathRegex operates against the url path, the queryParams match is case insensitive.
// Any one match flags the request.
// TODO tighten this check to eliminate the abuse potential by malicious clients that start setting queryParameters
// to bypass the rate limitter.  This could be done using a full parse and special casing the bits we need.
func BasicLongRunningRequestCheck(pathRegex *regexp.Regexp, queryParams map[string]string) LongRunningRequestCheck {
	return func(r *http.Request) bool {
		if pathRegex.MatchString(r.URL.Path) {
			return true
		}

		for key, expectedValue := range queryParams {
			if strings.ToLower(expectedValue) == strings.ToLower(r.URL.Query().Get(key)) {
				return true
			}
		}

		return false
	}
}

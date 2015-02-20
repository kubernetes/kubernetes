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

package csrf

import "net/http"

type emptyCsrf struct{}

// NewEmptyCSRF returns a CSRF object which generates empty CSRF tokens,
// and accepts any token as valid
func NewEmptyCSRF() CSRF {
	return emptyCsrf{}
}

// Generate implements the CSRF interface
func (emptyCsrf) Generate(http.ResponseWriter, *http.Request) (string, error) {
	return "", nil
}

// Check implements the CSRF interface
func (emptyCsrf) Check(*http.Request, string) (bool, error) {
	return true, nil
}

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

// FakeCSRF returns the given token and error for testing purposes
type FakeCSRF struct {
	Token string
	Err   error
}

// Generate implements the CSRF interface
func (c *FakeCSRF) Generate(w http.ResponseWriter, req *http.Request) (string, error) {
	return c.Token, c.Err
}

// Check implements the CSRF interface
func (c *FakeCSRF) Check(req *http.Request, value string) (bool, error) {
	return c.Token == value, c.Err
}

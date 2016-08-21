//
// Copyright (c) 2015 The heketi Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package utils

import (
	"errors"
	"io"
	"io/ioutil"
	"net/http"
)

// Return the body from a response as a string
func GetStringFromResponse(r *http.Response) (string, error) {
	body, err := ioutil.ReadAll(io.LimitReader(r.Body, r.ContentLength))
	if err != nil {
		return "", err
	}
	r.Body.Close()
	return string(body), nil
}

// Return the body from a response as an error
func GetErrorFromResponse(r *http.Response) error {
	s, err := GetStringFromResponse(r)
	if err != nil {
		return err
	}
	return errors.New(s)
}

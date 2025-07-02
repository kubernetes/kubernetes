// Copyright (c) 2025 Bart Venter <bartventer@proton.me>
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package internal

import (
	"net/http"
)

// RequestMethodChecker describes the interface implemented by types that can
// check whether the request method is understood by the cache according to RFC 9111 §3.
type RequestMethodChecker interface {
	IsRequestMethodUnderstood(req *http.Request) bool
}

type RequestMethodCheckerFunc func(req *http.Request) bool

func (f RequestMethodCheckerFunc) IsRequestMethodUnderstood(req *http.Request) bool {
	return f(req)
}

func NewRequestMethodChecker() RequestMethodChecker {
	return RequestMethodCheckerFunc(isRequestMethodUnderstood)
}

func isRequestMethodUnderstood(req *http.Request) bool {
	return req.Method == http.MethodGet && req.Header.Get("Range") == ""
}

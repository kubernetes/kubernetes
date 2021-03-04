// +build !go1.8

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

package internal

import "net/url"

// PathUnescape provides url.PathUnescape(), with seamless
// go version support for pre-go1.8
//
// TODO: this function is currently defined in go-openapi/swag,
// but unexported. We might chose to export it, or simple phase
// out pre-go1.8 support.
func PathUnescape(path string) (string, error) {
	return url.QueryUnescape(path)
}

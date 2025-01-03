/*
Copyright 2024 The Kubernetes Authors.

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

package httputil

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/munnerz/goautoneg"
)

// ErrUnsupportedMediaType is the error returned when the request's
// Accept header does not contain "text/plain".
var ErrUnsupportedMediaType = fmt.Errorf("media type not acceptable, must be: text/plain")

// AcceptableMediaType checks if the request's Accept header contains
// a supported media type with optional "charset=utf-8" parameter.
func AcceptableMediaType(r *http.Request) bool {
	accepts := goautoneg.ParseAccept(r.Header.Get("Accept"))
	for _, accept := range accepts {
		if !mediaTypeMatches(accept) {
			continue
		}
		if len(accept.Params) == 0 {
			return true
		}
		if len(accept.Params) == 1 {
			if charset, ok := accept.Params["charset"]; ok && strings.EqualFold(charset, "utf-8") {
				return true
			}
		}
	}
	return false
}

func mediaTypeMatches(a goautoneg.Accept) bool {
	return (a.Type == "text" || a.Type == "*") &&
		(a.SubType == "plain" || a.SubType == "*")
}

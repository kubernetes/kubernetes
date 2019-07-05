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
	"mime"
	"net/http"

	"github.com/go-openapi/errors"
)

// ContentType parses a content type header
func ContentType(headers http.Header) (string, string, error) {
	ct := headers.Get(HeaderContentType)
	orig := ct
	if ct == "" {
		ct = DefaultMime
	}
	if ct == "" {
		return "", "", nil
	}

	mt, opts, err := mime.ParseMediaType(ct)
	if err != nil {
		return "", "", errors.NewParseError(HeaderContentType, "header", orig, err)
	}

	if cs, ok := opts[charsetKey]; ok {
		return mt, cs, nil
	}

	return mt, "", nil
}

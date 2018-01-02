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
	"testing"

	"github.com/go-openapi/errors"
	"github.com/stretchr/testify/assert"
)

func TestParseContentType(t *testing.T) {
	_, _, reason1 := mime.ParseMediaType("application(")
	_, _, reason2 := mime.ParseMediaType("application/json;char*")
	data := []struct {
		hdr, mt, cs string
		err         *errors.ParseError
	}{
		{"application/json", "application/json", "", nil},
		{"text/html; charset=utf-8", "text/html", "utf-8", nil},
		{"text/html;charset=utf-8", "text/html", "utf-8", nil},
		{"", "application/octet-stream", "", nil},
		{"text/html;           charset=utf-8", "text/html", "utf-8", nil},
		{"application(", "", "", errors.NewParseError("Content-Type", "header", "application(", reason1)},
		{"application/json;char*", "", "", errors.NewParseError("Content-Type", "header", "application/json;char*", reason2)},
	}

	headers := http.Header(map[string][]string{})
	for _, v := range data {
		if v.hdr != "" {
			headers.Set("Content-Type", v.hdr)
		} else {
			headers.Del("Content-Type")
		}
		ct, cs, err := ContentType(headers)
		if v.err == nil {
			assert.NoError(t, err, "input: %q, err: %v", v.hdr, err)
		} else {
			assert.Error(t, err, "input: %q", v.hdr)
			assert.IsType(t, &errors.ParseError{}, err, "input: %q", v.hdr)
			assert.Equal(t, v.err.Error(), err.Error(), "input: %q", v.hdr)
		}
		assert.Equal(t, v.mt, ct, "input: %q", v.hdr)
		assert.Equal(t, v.cs, cs, "input: %q", v.hdr)
	}

}

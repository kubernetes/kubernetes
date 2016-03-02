// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Implements systemd-escape [--unescape] [--path]

package unit

import (
	"fmt"
	"strconv"
	"strings"
)

const (
	allowed = `:_.abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789`
)

// If isPath is true:
//   We remove redundant '/'s, the leading '/', and trailing '/'.
//   If the result is empty, a '/' is inserted.
//
// We always:
//  Replace the following characters with `\x%x`:
//   Leading `.`
//   `-`, `\`, and anything not in this set: `:-_.\[0-9a-zA-Z]`
//  Replace '/' with '-'.
func escape(unescaped string, isPath bool) string {
	e := []byte{}
	inSlashes := false
	start := true
	for i := 0; i < len(unescaped); i++ {
		c := unescaped[i]
		if isPath {
			if c == '/' {
				inSlashes = true
				continue
			} else if inSlashes {
				inSlashes = false
				if !start {
					e = append(e, '-')
				}
			}
		}

		if c == '/' {
			e = append(e, '-')
		} else if start && c == '.' || strings.IndexByte(allowed, c) == -1 {
			e = append(e, []byte(fmt.Sprintf(`\x%x`, c))...)
		} else {
			e = append(e, c)
		}
		start = false
	}
	if isPath && len(e) == 0 {
		e = append(e, '-')
	}
	return string(e)
}

// If isPath is true:
//   We always return a string beginning with '/'.
//
// We always:
//  Replace '-' with '/'.
//  Replace `\x%x` with the value represented in hex.
func unescape(escaped string, isPath bool) string {
	u := []byte{}
	for i := 0; i < len(escaped); i++ {
		c := escaped[i]
		if c == '-' {
			c = '/'
		} else if c == '\\' && len(escaped)-i >= 4 && escaped[i+1] == 'x' {
			n, err := strconv.ParseInt(escaped[i+2:i+4], 16, 8)
			if err == nil {
				c = byte(n)
				i += 3
			}
		}
		u = append(u, c)
	}
	if isPath && (len(u) == 0 || u[0] != '/') {
		u = append([]byte("/"), u...)
	}
	return string(u)
}

// UnitNameEscape escapes a string as `systemd-escape` would
func UnitNameEscape(unescaped string) string {
	return escape(unescaped, false)
}

// UnitNameUnescape unescapes a string as `systemd-escape --unescape` would
func UnitNameUnescape(escaped string) string {
	return unescape(escaped, false)
}

// UnitNamePathEscape escapes a string as `systemd-escape --path` would
func UnitNamePathEscape(unescaped string) string {
	return escape(unescaped, true)
}

// UnitNamePathUnescape unescapes a string as `systemd-escape --path --unescape` would
func UnitNamePathUnescape(escaped string) string {
	return unescape(escaped, true)
}

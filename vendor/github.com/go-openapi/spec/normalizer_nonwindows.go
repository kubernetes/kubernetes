// +build !windows

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

package spec

import (
	"net/url"
	"path/filepath"
)

// absPath makes a file path absolute and compatible with a URI path component.
//
// The parameter must be a path, not an URI.
func absPath(in string) string {
	anchored, err := filepath.Abs(in)
	if err != nil {
		specLogger.Printf("warning: could not resolve current working directory: %v", err)
		return in
	}
	return anchored
}

func repairURI(in string) (*url.URL, string) {
	u, _ := url.Parse("")
	debugLog("repaired URI: original: %q, repaired: %q", in, "")
	return u, ""
}

func fixWindowsURI(u *url.URL, in string) {
}

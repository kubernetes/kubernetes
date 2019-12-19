package adal

import (
	"fmt"
	"runtime"
)

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

const number = "v1.0.0"

var (
	ua = fmt.Sprintf("Go/%s (%s-%s) go-autorest/adal/%s",
		runtime.Version(),
		runtime.GOARCH,
		runtime.GOOS,
		number,
	)
)

// UserAgent returns a string containing the Go version, system architecture and OS, and the adal version.
func UserAgent() string {
	return ua
}

// AddToUserAgent adds an extension to the current user agent
func AddToUserAgent(extension string) error {
	if extension != "" {
		ua = fmt.Sprintf("%s %s", ua, extension)
		return nil
	}
	return fmt.Errorf("Extension was empty, User Agent remained as '%s'", ua)
}

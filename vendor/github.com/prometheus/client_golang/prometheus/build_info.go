// Copyright 2019 The Prometheus Authors
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

// +build go1.12

package prometheus

import "runtime/debug"

// readBuildInfo is a wrapper around debug.ReadBuildInfo for Go 1.12+.
func readBuildInfo() (path, version, sum string) {
	path, version, sum = "unknown", "unknown", "unknown"
	if bi, ok := debug.ReadBuildInfo(); ok {
		path = bi.Main.Path
		version = bi.Main.Version
		sum = bi.Main.Sum
	}
	return
}

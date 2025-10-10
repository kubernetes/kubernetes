// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build !windows && !linux

package metadata

// systemInfoSuggestsGCE reports whether the local system (without
// doing network requests) suggests that we're running on GCE. If this
// returns true, testOnGCE tries a bit harder to reach its metadata
// server.
//
// NOTE: systemInfoSuggestsGCE is assigned to a varible for test stubbing purposes.
var systemInfoSuggestsGCE = func() bool {
	// We don't currently have checks for other GOOS
	return false
}

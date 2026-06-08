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

package filesystem

import (
	"path/filepath"
)

// IsPathClean will replace slashes to Separator (which is OS-specific).
// This will make sure that all slashes are the same before comparing.
func IsPathClean(path string) bool {
	return filepath.ToSlash(filepath.Clean(path)) == filepath.ToSlash(path)
}

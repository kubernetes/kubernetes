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

package invoke

import (
	"os"
	"path/filepath"
	"strings"
)

func FindInPath(plugin string, path []string) string {
	for _, p := range path {
		fullname := filepath.Join(p, plugin)
		if fi, err := os.Stat(fullname); err == nil && fi.Mode().IsRegular() {
			return fullname
		}
	}
	return ""
}

// Find returns the full path of the plugin by searching in CNI_PATH
func Find(plugin string) string {
	paths := strings.Split(os.Getenv("CNI_PATH"), ":")
	return FindInPath(plugin, paths)
}

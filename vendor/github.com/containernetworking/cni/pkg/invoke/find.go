// Copyright 2015 CNI authors
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
	"fmt"
	"os"
	"path/filepath"
)

// FindInPath returns the full path of the plugin by searching in the provided path
func FindInPath(plugin string, paths []string) (string, error) {
	if plugin == "" {
		return "", fmt.Errorf("no plugin name provided")
	}

	if len(paths) == 0 {
		return "", fmt.Errorf("no paths provided")
	}

	var fullpath string
	for _, path := range paths {
		full := filepath.Join(path, plugin)
		if fi, err := os.Stat(full); err == nil && fi.Mode().IsRegular() {
			fullpath = full
			break
		}
	}

	if fullpath == "" {
		return "", fmt.Errorf("failed to find plugin %q in path %s", plugin, paths)
	}

	return fullpath, nil
}

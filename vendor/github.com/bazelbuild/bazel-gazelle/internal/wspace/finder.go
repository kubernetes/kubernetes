/* Copyright 2016 The Bazel Authors. All rights reserved.
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

// Package wspace provides functions to locate and modify a bazel WORKSPACE file.
package wspace

import (
	"os"
	"path/filepath"
	"strings"
)

const workspaceFile = "WORKSPACE"

// Find searches from the given dir and up for the WORKSPACE file
// returning the directory containing it, or an error if none found in the tree.
func Find(dir string) (string, error) {
	dir, err := filepath.Abs(dir)
	if err != nil {
		return "", err
	}

	for {
		_, err = os.Stat(filepath.Join(dir, workspaceFile))
		if err == nil {
			return dir, nil
		}
		if !os.IsNotExist(err) {
			return "", err
		}
		if strings.HasSuffix(dir, string(os.PathSeparator)) { // stop at root dir
			return "", os.ErrNotExist
		}
		dir = filepath.Dir(dir)
	}
}

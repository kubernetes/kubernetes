// Copyright 2018 Microsoft Corporation
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

package dirs

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

// GetSubdirs returns all of the subdirectories under current.
// Returns an empty slice if current contains no subdirectories.
func GetSubdirs(current string) ([]string, error) {
	children, err := ioutil.ReadDir(current)
	if err != nil {
		return nil, err
	}
	dirs := []string{}
	for _, info := range children {
		if info.IsDir() {
			dirs = append(dirs, info.Name())
		}
	}
	return dirs, nil
}

// DeleteChildDirs deletes all child directories in the specified directory.
func DeleteChildDirs(dir string) error {
	children, err := GetSubdirs(dir)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	for _, child := range children {
		childPath := filepath.Join(dir, child)
		err = os.RemoveAll(childPath)
		if err != nil {
			return err
		}
	}
	return nil
}

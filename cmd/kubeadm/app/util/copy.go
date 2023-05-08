/*
Copyright 2023 The Kubernetes Authors.

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

package util

import (
	"os"
)

// CopyFile copies a file from src to dest.
func CopyFile(src, dest string) error {
	fileInfo, err := os.Stat(src)
	if err != nil {
		return err
	}
	contents, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	err = os.WriteFile(dest, contents, fileInfo.Mode())
	return err
}

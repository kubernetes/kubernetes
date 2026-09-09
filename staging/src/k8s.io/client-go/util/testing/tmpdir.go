/*
Copyright 2016 The Kubernetes Authors.

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

package testing

import (
	"os"
)

// MkTmpdir creates a temporary directory based upon the prefix passed in.
// If successful, it returns the temporary directory path. The directory can be
// deleted with a call to "os.RemoveAll(...)".
// In case of error, it'll return an empty string and the error.
func MkTmpdir(prefix string) (string, error) {
	tmpDir, err := os.MkdirTemp(os.TempDir(), prefix)
	if err != nil {
		return "", err
	}
	return tmpDir, nil
}

// MkTmpdirOrDie does the same work as "MkTmpdir", except in case of
// errors, it'll trigger a panic.
func MkTmpdirOrDie(prefix string) string {
	tmpDir, err := MkTmpdir(prefix)
	if err != nil {
		panic(err)
	}
	return tmpDir
}

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

package fs

import "os"

// WithinDirectory changes the current working directory to dir, and then
// executes fn. It then changes the current working directory back to the
// original directory.
func WithinDirectory(dir string, fn func() error) error {
	if wd, err := os.Getwd(); err != nil {
		return err
	} else {
		defer func() {
			_ = os.Chdir(wd)
		}()
	}
	if err := os.Chdir(dir); err != nil {
		return err
	}
	return fn()
}

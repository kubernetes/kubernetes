/*
Copyright 2020 The Kubernetes Authors.

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

package main

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
)

func copyFile(source, dest string) error {
	sf, err := os.Open(source)
	if err != nil {
		return fmt.Errorf("unable to open source file [%s]: %q", source, err)
	}
	defer sf.Close()
	fi, err := sf.Stat()
	if err != nil {
		return fmt.Errorf("unable to stat source file [%s]: %q", source, err)
	}

	dir := filepath.Dir(dest)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("unable to create directory [%s]: %q", dir, err)
	}
	df, err := os.Create(dest)
	if err != nil {
		return fmt.Errorf("unable to create destination file [%s]: %q", dest, err)
	}
	defer df.Close()

	_, err = io.Copy(df, sf)
	if err != nil {
		return fmt.Errorf("unable to copy [%s] to [%s]: %q", source, dest, err)
	}

	if err := os.Chmod(dest, fi.Mode()); err != nil {
		return fmt.Errorf("unable to close destination file: %q", err)
	}
	return nil
}

/*
Copyright 2016 Google Inc. All Rights Reserved.
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

// Package file provides utility file operations.
package file

import (
	"fmt"
	"io"
	"io/ioutil"
	"os"
)

// ReadFile is like ioutil.ReadFile.
func ReadFile(name string) ([]byte, os.FileInfo, error) {
	fi, err := os.Stat(name)
	if err != nil {
		return nil, nil, err
	}

	data, err := ioutil.ReadFile(name)
	return data, fi, err
}

// WriteFile is like ioutil.WriteFile
func WriteFile(name string, data []byte) error {
	return ioutil.WriteFile(name, data, 0644)
}

// OpenReadFile is like os.Open.
func OpenReadFile(name string) io.ReadCloser {
	f, err := os.Open(name)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Could not open %s\n", name)
		os.Exit(1)
	}
	return f
}

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

package main

import (
	"bytes"
	"io/ioutil"
	"os"

	"testing"
)

func TestCopyFile(t *testing.T) {

	// create testing file

	_, err := os.Create("test1.txt")
	if err != nil {
		t.Errorf("got error: %v", err)
	}
	defer os.Remove("test1.txt")

	// call copyFile function
	err = copyFile("test1.txt", "test2.txt")
	if err != nil {
		t.Errorf("got error: %v", err)
	}

	// verify copy file completed
	if _, err := os.Stat("test2.txt"); os.IsNotExist(err) {
		t.Errorf("file test2.txt does not exist")
	}

	// verify copy file content is same
	content1, _ := ioutil.ReadFile("test1.txt")
	content2, _ := ioutil.ReadFile("test2.txt")
	if !bytes.Equal(content1, content2) {
		t.Errorf("file content is not the same")
	}

	// clean up testing file
	os.Remove("test2.txt")
}

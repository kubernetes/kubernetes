/*
Copyright 2019 The Kubernetes Authors.

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

import (
	"fmt"
	"io/ioutil"
	"os"

	"k8s.io/apimachinery/pkg/api/resource"
	"testing"
)

func TestDiskUsage(t *testing.T) {

	dir_1, err := ioutil.TempDir("", "dir_1")
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	defer os.RemoveAll(dir_1)

	tmpfile_1, err := ioutil.TempFile(dir_1, "test")
	if _, err = tmpfile_1.WriteString("just for testing"); err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	dir_2, err := ioutil.TempDir(dir_1, "dir_2")
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	tmpfile_2, err := ioutil.TempFile(dir_2, "test")
	if _, err = tmpfile_2.WriteString("just for testing"); err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}

	dirInfo1, err := os.Lstat(dir_1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	dirInfo2, err := os.Lstat(dir_2)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	file1 := dir_1 + "/" + "test"
	file2 := dir_2 + "/" + "test"
	fileInfo1, err := os.Lstat(file1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	fileInfo2, err := os.Lstat(file2)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	total := dirInfo1.Size() + dirInfo2.Size() + fileInfo1.Size() + fileInfo2.Size()

	size, err := DiskUsage(dir_1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	used, err := resource.ParseQuantity(fmt.Sprintf("%d", total))
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	if size.Cmp(used) != 1 {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}

}

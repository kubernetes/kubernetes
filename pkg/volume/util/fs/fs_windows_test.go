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
	"os"

	"testing"

	"k8s.io/apimachinery/pkg/api/resource"
)

func TestDiskUsage(t *testing.T) {

	dir1, err := os.MkdirTemp("", "dir_1")
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	defer os.RemoveAll(dir1)

	tmpfile1, err := os.CreateTemp(dir1, "test")
	if _, err = tmpfile1.WriteString("just for testing"); err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	dir2, err := os.MkdirTemp(dir1, "dir_2")
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	tmpfile2, err := os.CreateTemp(dir2, "test")
	if _, err = tmpfile2.WriteString("just for testing"); err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}

	// File creation is not atomic. If we're calculating the DiskUsage before the data is flushed,
	// we'd get zeroes for sizes, and fail with this error:
	// TestDiskUsage failed: expected 0, got -1
	tmpfile1.Sync()
	tmpfile2.Sync()

	dirInfo1, err := os.Lstat(dir1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	dirInfo2, err := os.Lstat(dir2)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	file1 := tmpfile1.Name()
	file2 := tmpfile2.Name()
	fileInfo1, err := os.Lstat(file1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	fileInfo2, err := os.Lstat(file2)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	total := dirInfo1.Size() + dirInfo2.Size() + fileInfo1.Size() + fileInfo2.Size()

	usage, err := DiskUsage(dir1)
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	size, err := resource.ParseQuantity(fmt.Sprintf("%d", usage.Bytes))
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}

	used, err := resource.ParseQuantity(fmt.Sprintf("%d", total))
	if err != nil {
		t.Fatalf("TestDiskUsage failed: %s", err.Error())
	}
	if size.Cmp(used) != 0 {
		t.Fatalf("TestDiskUsage failed: expected 0, got %d", size.Cmp(used))
	}
}

func TestInfo(t *testing.T) {
	dir1, err := os.MkdirTemp("", "dir_1")
	if err != nil {
		t.Fatalf("TestInfo failed: %s", err.Error())
	}
	defer os.RemoveAll(dir1)

	// should pass for folder path
	availablebytes, capacity, usage, inodesTotal, inodesFree, inodeUsage, err := Info(dir1)
	if err != nil {
		t.Errorf("Info() should not error = %v", err)
		return
	}
	validateInfo(t, availablebytes, capacity, usage, inodesTotal, inodeUsage, inodesFree)

	// should pass for file
	tmpfile1, err := os.CreateTemp(dir1, "test")
	if _, err = tmpfile1.WriteString("just for testing"); err != nil {
		t.Fatalf("TestInfo failed: %s", err.Error())
	}
	availablebytes, capacity, usage, inodesTotal, inodesFree, inodeUsage, err = Info(tmpfile1.Name())
	validateInfo(t, availablebytes, capacity, usage, inodesTotal, inodeUsage, inodesFree)
}

func validateInfo(t *testing.T, availablebytes int64, capacity int64, usage int64, inodesTotal int64, inodeUsage int64, inodesFree int64) {
	// All of these should be greater than zero on a real system
	if availablebytes <= 0 {
		t.Errorf("Info() availablebytes should be greater than 0, got %v", availablebytes)
	}
	if capacity <= 0 {
		t.Errorf("Info() capacity should be greater than 0, got %v", capacity)
	}
	if usage <= 0 {
		t.Errorf("Info() got usage should be greater than 0, got %v", usage)
	}

	// inodes will always be zero for Windows
	if inodesTotal != 0 {
		t.Errorf("Info() inodesTotal = %v, want %v", inodeUsage, 0)
	}
	if inodesFree != 0 {
		t.Errorf("Info() inodesFree = %v, want %v", inodesFree, 0)
	}
	if inodeUsage != 0 {
		t.Errorf("Info() inodeUsage = %v, want %v", inodeUsage, 0)
	}
}

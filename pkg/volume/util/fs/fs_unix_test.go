//go:build unix

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

import (
	"os"
	"testing"
)

func TestDiskUsage(t *testing.T) {
	dir1, err := os.MkdirTemp("", "dir_1")
	if err != nil {
		t.Fatalf("creating temp directory, %s", err)
	}
	defer os.RemoveAll(dir1)

	initialUsage, err := DiskUsage(dir1)
	if err != nil {
		t.Errorf("DiskUsage should not error, got %s", err)
		return
	}

	tmpfile1, err := os.CreateTemp(dir1, "test")
	if err != nil {
		t.Fatalf("creating temp file, %s", err)
	}
	if _, err = tmpfile1.WriteString("just for testing"); err != nil {
		t.Fatalf("writing to file, %s", err)
	}

	dir2, err := os.MkdirTemp(dir1, "dir_2")
	if err != nil {
		t.Fatalf("creating temp directory, %s", err)
	}
	tmpfile2, err := os.CreateTemp(dir2, "test")
	if err != nil {
		t.Fatalf("creating temp file, %s", err)
	}
	if _, err = tmpfile2.WriteString("just for testing"); err != nil {
		t.Fatalf("writing to file, %s", err)
	}

	usage, err := DiskUsage(dir1)
	if err != nil {
		t.Errorf("DiskUsage should not error, got %s", err)
	}

	if initialUsage.Bytes >= usage.Bytes {
		t.Fatalf("expected directory disk usage to increase after creating files, had initial of %d and post-create of %d",
			initialUsage.Bytes, usage.Bytes)
	}
	if initialUsage.Inodes >= usage.Inodes {
		t.Fatalf("expected directory inode usage to increase after creating files, had initial of %d and post-create of %d",
			initialUsage.Bytes, usage.Bytes)
	}
}

func TestInfo(t *testing.T) {
	dir1, err := os.MkdirTemp("", "dir_1")
	if err != nil {
		t.Fatalf("creating temp directory, %s", err)
	}
	defer os.RemoveAll(dir1)

	// should pass for folder path
	info, err := Info(dir1)
	if err != nil {
		t.Errorf("Info() should not error, got %s", err)
		return
	}
	validateInfo(t, info)

	// should pass for file
	tmpfile1, err := os.CreateTemp(dir1, "test")
	if err != nil {
		t.Fatalf("creating temp file, %s", err)
	}
	if _, err = tmpfile1.WriteString("just for testing"); err != nil {
		t.Fatalf("writing to file, %s", err)
	}
	info, err = Info(tmpfile1.Name())
	if err != nil {
		t.Errorf("Info() should not error = %v", err)
	}
	validateInfo(t, info)
}

func validateInfo(t *testing.T, info FSInfo) {
	t.Helper()
	// All of these should be greater than zero on a real system
	if info.Available <= 0 {
		t.Errorf("Info() availablebytes should be greater than 0, got %v", info.Available)
	}
	if info.Capacity <= 0 {
		t.Errorf("Info() capacity should be greater than 0, got %v", info.Capacity)
	}
	if info.Usage <= 0 {
		t.Errorf("Info() got usage should be greater than 0, got %v", info.Usage)
	}

	// inodes will always be non-zero for Linux
	if info.Inodes <= 0 {
		t.Errorf("Info().InodesTotal should be greater than 0, got %v", info.Inodes)
	}
	if info.InodesFree <= 0 {
		t.Errorf("Info().InodesFree should be greater than 0, got %v", info.InodesFree)
	}
	if info.InodesUsed <= 0 {
		t.Errorf("Info().InodesUsed should be greater than 0, got %v", info.InodesUsed)
	}
	// tmp shouldn't be on a read-only filesystem
	if info.ReadOnly {
		t.Errorf("Info().ReadOnly should be false")
	}
}

//go:build linux

/*
Copyright The Kubernetes Authors.

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

package mounttest

import (
	"bytes"
	"fmt"
	"io"
	"os"
	"strings"
	"testing"

	"golang.org/x/sys/unix"
)

func captureStdout(f func()) string {
	old := os.Stdout
	r, w, _ := os.Pipe()
	os.Stdout = w

	f()

	w.Close()
	os.Stdout = old

	var buf bytes.Buffer
	io.Copy(&buf, r)
	return buf.String()
}

func TestFsType_EmptyPath(t *testing.T) {
	output := captureStdout(func() {
		err := fsType("")
		if err != nil {
			t.Errorf("fsType(\"\") returned error: %v", err)
		}
	})

	if output != "" {
		t.Errorf("fsType(\"\") should not print anything, got: %q", output)
	}
}

func TestFsType_ValidPath(t *testing.T) {
	// Use a path that exists on the system
	testPath := os.TempDir()

	output := captureStdout(func() {
		err := fsType(testPath)
		if err != nil {
			t.Errorf("fsType(%q) returned error: %v", testPath, err)
		}
	})

	// Verify the output format
	expectedPrefix := fmt.Sprintf("mount type of %q:", testPath)
	if !strings.HasPrefix(output, expectedPrefix) {
		t.Errorf("fsType(%q) output should start with %q, got: %q", testPath, expectedPrefix, output)
	}

	// Verify the output is printed exactly once (not duplicated)
	lines := strings.Split(strings.TrimSpace(output), "\n")
	if len(lines) != 1 {
		t.Errorf("fsType(%q) should print exactly one line, got %d lines: %q", testPath, len(lines), output)
	}
}

func TestFsType_NonExistentPath(t *testing.T) {
	testPath := "/nonexistent/path/that/does/not/exist"

	output := captureStdout(func() {
		err := fsType(testPath)
		if err == nil {
			t.Errorf("fsType(%q) should return error for non-existent path", testPath)
		}
	})

	// Should print an error message
	if !strings.Contains(output, "error from statfs") {
		t.Errorf("fsType(%q) should print error message, got: %q", testPath, output)
	}
}

func TestFormatFsType_Tmpfs(t *testing.T) {
	result := formatFsType(unix.TMPFS_MAGIC)

	expected := "tmpfs"
	if result != expected {
		t.Errorf("formatFsType(TMPFS_MAGIC) = %q, want %q", result, expected)
	}
}

func TestFormatFsType_OtherType(t *testing.T) {
	otherType := int64(0x12345678)
	result := formatFsType(otherType)

	expected := "305419896"
	if result != expected {
		t.Errorf("formatFsType(%v) = %q, want %q", otherType, result, expected)
	}
}

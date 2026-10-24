/*
Copyright 2021 The Kubernetes Authors.

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

package routes

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestPreCheckLogFileNameLength(t *testing.T) {
	// in linux file paths can be up to 4096 bytes
	// and one filename 255 bytes

	// in macos file paths can be up to 1024 bytes
	// and one filename 255 bytes

	// in windows realtive file paths can have up to 260 bytes
	// and one filename 255 bytes
	// https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation

	// endpoint in logs.go takes relative path to log
	// logFileNameIsTooLong uses os.Stat which on os level adds cwd to relative path ( cwd + relative )

	// oversizeFileName won't work on any os
	oversizeFileName := fmt.Sprintf("%256s", "a")

	// oversizeFilePath is a relative path, that is 2097 itself + pwd also can't work on any system
	oversizeFilePath := "." + strings.Repeat("/a", 2048)
	oversizeFilePath, err := filepath.Abs(oversizeFilePath)
	if err != nil {
		t.Error("error getting abs path")
	}

	normalFileName := fmt.Sprintf("%255s", "a")

	// check file with oversize name.
	if !logFileNameIsTooLong(oversizeFileName) {
		t.Error("failed to check oversize filename")
	}

	if !logFileNameIsTooLong(oversizeFilePath) {
		t.Error("failed to check oversize for relative path")
	}

	// check file with normal name which doesn't exist.
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}

	// check file with normal name which does exist.
	dir, err := os.MkdirTemp("", "logs")
	if err != nil {
		t.Fatal("failed to create temp dir")
	}
	defer os.RemoveAll(dir)

	normalFileName = filepath.Join(dir, normalFileName)
	f, err := os.Create(normalFileName)
	if err != nil {
		t.Error("failed to create test file")
	}
	defer os.Remove(normalFileName)
	defer f.Close()
	if logFileNameIsTooLong(normalFileName) {
		t.Error("failed to check normal filename")
	}
}

//go:build windows
// +build windows

/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const npipeProtocol = "npipe"

// LocalEndpoint returns the full path to a named pipe at the given endpoint - unlike on unix, we can't use sockets.
func LocalEndpoint(path, file string) (string, error) {
	// extract the podresources config name from the path. We only need this on windows because the preferred layout of pipes,
	// this is why we have the extra logic in here instead of changing the function signature. Join the file to make sure the
	// last path component is a file, so the operation chain works..
	podResourcesDir := filepath.Base(filepath.Dir(filepath.Join(path, file)))
	if podResourcesDir == "" {
		// should not happen because the user can configure a root directory, and we expected a subdirectory inside
		// the user supplied root directory named like "pod-resources" or so.
		return "", fmt.Errorf("cannot infer the podresources directory from path %q", path)
	}
	// windows pipes are expected to use forward slashes: https://learn.microsoft.com/windows/win32/ipc/pipe-names
	// so using `url` like we do on unix gives us unclear benefits - see https://github.com/kubernetes/kubernetes/issues/78628
	// So we just construct the path from scratch.
	// Format: \\ServerName\pipe\PipeName
	// Where where ServerName is either the name of a remote computer or a period, to specify the local computer.
	// We only consider PipeName as regular windows path, while the pipe path components are fixed, hence we use constants.
	serverPart := `\\.`
	pipePart := "pipe"
	pipeName := "kubelet-" + podResourcesDir
	return npipeProtocol + "://" + filepath.Join(serverPart, pipePart, pipeName), nil
}

var tickCount = syscall.NewLazyDLL("kernel32.dll").NewProc("GetTickCount64")

// GetBootTime returns the time at which the machine was started, truncated to the nearest second
func GetBootTime() (time.Time, error) {
	currentTime := time.Now()
	output, _, err := tickCount.Call()
	if errno, ok := err.(syscall.Errno); !ok || errno != 0 {
		return time.Time{}, err
	}
	return currentTime.Add(-time.Duration(output) * time.Millisecond).Truncate(time.Second), nil
}

// NormalizePath converts FS paths returned by certain go frameworks (like fsnotify)
// to native Windows paths that can be passed to Windows specific code
func NormalizePath(path string) string {
	path = strings.ReplaceAll(path, "/", "\\")
	if strings.HasPrefix(path, "\\") {
		path = "c:" + path
	}
	return path
}

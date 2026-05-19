//go:build !freebsd && !linux && !windows && !darwin
// +build !freebsd,!linux,!windows,!darwin

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
	"time"
)

// LockAndCheckSubPath empty implementation
func LockAndCheckSubPath(volumePath, subPath string) ([]uintptr, error) {
	return []uintptr{}, nil
}

// UnlockPath empty implementation
func UnlockPath(fileHandles []uintptr) {
}

// LocalEndpoint empty implementation
func LocalEndpoint(path, file string) (string, error) {
	return "", fmt.Errorf("LocalEndpoints are unsupported in this build")
}

// GetBootTime empty implementation
func GetBootTime() (time.Time, error) {
	return time.Time{}, fmt.Errorf("GetBootTime is unsupported in this build")
}

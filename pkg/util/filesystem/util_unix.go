//go:build freebsd || linux || darwin
// +build freebsd linux darwin

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

package filesystem

import (
	"os"
)

// Chmod is the same as os.Chmod on Linux.
func Chmod(name string, mode os.FileMode) error {
	return os.Chmod(name, mode)
}

// MkdirAll is the same as os.MkdirAll on Linux.
func MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

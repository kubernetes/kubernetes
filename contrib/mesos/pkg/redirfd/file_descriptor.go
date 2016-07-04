/*
Copyright 2015 The Kubernetes Authors.

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

package redirfd

import (
	"fmt"
	"strconv"
)

// FileDescriptor mirrors unix-specific indexes for cross-platform use
type FileDescriptor int

const (
	InvalidFD FileDescriptor = -1
	Stdin     FileDescriptor = 0
	Stdout    FileDescriptor = 1
	Stderr    FileDescriptor = 2
)

// ParseFileDescriptor parses a string formatted file descriptor
func ParseFileDescriptor(fdstr string) (FileDescriptor, error) {
	fdint, err := strconv.Atoi(fdstr)
	if err != nil {
		return InvalidFD, fmt.Errorf("file descriptor must be an integer: %q", fdstr)
	}
	return FileDescriptor(fdint), nil
}

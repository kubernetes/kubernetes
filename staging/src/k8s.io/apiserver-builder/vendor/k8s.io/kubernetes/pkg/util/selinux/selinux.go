/*
Copyright 2014 The Kubernetes Authors.

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

package selinux

// Note: the libcontainer SELinux package is only built for Linux, so it is
// necessary to have a NOP wrapper which is built for non-Linux platforms to
// allow code that links to this package not to differentiate its own methods
// for Linux and non-Linux platforms.
//
// SELinuxRunner wraps certain libcontainer SELinux calls. For more
// information, see:
//
// https://github.com/opencontainers/runc/blob/master/libcontainer/selinux/selinux.go
type SELinuxRunner interface {
	// Getfilecon returns the SELinux context for the given path or returns an
	// error.
	Getfilecon(path string) (string, error)
}

// NewSELinuxRunner returns a new SELinuxRunner appropriate for the platform.
// On Linux, all methods short-circuit and return NOP values if SELinux is
// disabled. On non-Linux platforms, a NOP implementation is returned.
func NewSELinuxRunner() SELinuxRunner {
	return &realSELinuxRunner{}
}

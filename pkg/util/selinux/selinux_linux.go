// +build linux

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

import (
	"fmt"

	"github.com/opencontainers/runc/libcontainer/selinux"
)

type realSelinuxContextRunner struct{}

func (_ *realSelinuxContextRunner) SetContext(dir, context string) error {
	// If SELinux is not enabled, return an empty string
	if !selinux.SelinuxEnabled() {
		return nil
	}

	return selinux.Setfilecon(dir, context)
}

func (_ *realSelinuxContextRunner) Getfilecon(path string) (string, error) {
	if !selinux.SelinuxEnabled() {
		return "", fmt.Errorf("SELinux is not enabled")
	}
	return selinux.Getfilecon(path)
}

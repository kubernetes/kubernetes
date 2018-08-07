// +build !windows

/*
Copyright 2018 The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"syscall"
)

// Chroot chroot()s to the new path.
// NB: All file paths after this call are effectively relative to
// `rootfs`
func Chroot(rootfs string) error {
	if err := syscall.Chroot(rootfs); err != nil {
		return fmt.Errorf("unable to chroot to %s: %v", rootfs, err)
	}
	root := filepath.FromSlash("/")
	if err := os.Chdir(root); err != nil {
		return fmt.Errorf("unable to chdir to %s: %v", root, err)
	}
	return nil
}

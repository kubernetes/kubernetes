//go:build !windows
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
	"os"
	"path/filepath"
	"syscall"

	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
)

// Chroot chroot()s to the new path.
// NB: All file paths after this call are effectively relative to
// `rootfs`
func Chroot(rootfs string) error {
	if err := syscall.Chroot(rootfs); err != nil {
		return errors.Wrapf(err, "unable to chroot to %s", rootfs)
	}
	root := filepath.FromSlash("/")
	if err := os.Chdir(root); err != nil {
		return errors.Wrapf(err, "unable to chdir to %s", root)
	}
	return nil
}

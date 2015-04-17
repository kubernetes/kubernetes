// +build linux

/*
Copyright 2015 Google Inc. All rights reserved.

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

package kubelet

import (
	"github.com/docker/libcontainer/selinux"
)

// getRootContext gets the SELinux context of the kubelet rootDir
// or returns an error.
func (kl *Kubelet) getRootDirContext() (string, error) {
	// If SELinux is not enabled, return an empty string
	if !selinux.SelinuxEnabled() {
		return "", nil
	}

	// Get the SELinux context of the rootDir.
	rootContext, err := selinux.Getfilecon(kl.getRootDir())
	if err != nil {
		return "", err
	}

	// There is a libcontainer bug where the null byte is not stripped from
	// the result of reading some selinux xattrs; strip it.
	//
	// TODO: remove when https://github.com/docker/libcontainer/issues/499
	// is fixed
	rootContext = rootContext[:len(rootContext)-1]

	return rootContext, nil
}

//go:build !linux && !windows
// +build !linux,!windows

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

package subpath

import (
	"errors"
	"os"

	"k8s.io/mount-utils"
)

type subpath struct{}

var errUnsupported = errors.New("util/subpath on this platform is not supported")

// New returns a subpath.Interface for the current system.
func New(mount.Interface) Interface {
	return &subpath{}
}

func (sp *subpath) PrepareSafeSubpath(subPath Subpath) (newHostPath string, cleanupAction func(), err error) {
	return subPath.Path, nil, errUnsupported
}

func (sp *subpath) CleanSubPaths(podDir string, volumeName string) error {
	return errUnsupported
}

func (sp *subpath) SafeMakeDir(pathname string, base string, perm os.FileMode) error {
	return errUnsupported
}

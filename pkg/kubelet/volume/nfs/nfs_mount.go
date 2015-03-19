/*
Copyright 2014 Google Inc. All rights reserved.

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

package nfs

import (
	"os/exec"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/golang/glog"
)

type nfsMountInterface interface {
	// Mount takes an NFS host ip or hostname, a source directory (the exported directory), a target directory where the source directory will be mounted, and a boolean readOnly
	Mount(server string, source string, target string, readOnly bool) error

	// Umount wraps syscall.Mount().
	Unmount(target string) error

	List() ([]mount.MountPoint, error)

	IsMountPoint(dir string) (bool, error)
}

// newNFSMounter returns an nfsMountInterface for the current system.
func newNFSMounter() nfsMountInterface {
	return &nfsMounter{}
}

type nfsMounter struct{}

func (mounter *nfsMounter) Mount(server string, exportDir string, mountDir string, readOnly bool) error {
	mountOptions := "rw"
	if readOnly {
		mountOptions = "ro"
	}
	mountArgs := []string{"-t", "nfs", server + ":" + exportDir, mountDir, "-o", mountOptions}
	command := exec.Command("mount", mountArgs...)
	output, errs := command.CombinedOutput()
	if errs != nil {
		glog.Errorf("NFS mounting failed: %v\n\tMount args are: %v\n\texportDir is: %v\n\tmountDir is: %v\n\tserver is: %v\n\tmount output is: %v", errs, mountArgs, exportDir, mountDir, server, string(output))
		return errs
	}
	return nil
}

func (mounter *nfsMounter) Unmount(target string) error {
	unmounter := mount.New()
	return unmounter.Unmount(target, 0)
}

func (mounter *nfsMounter) List() ([]mount.MountPoint, error) {
	return nil, nil
}

func (mounter *nfsMounter) IsMountPoint(dir string) (bool, error) {
	return mount.IsMountPoint(dir)
}

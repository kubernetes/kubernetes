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
	"fmt"
	"syscall"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/mount"
	"github.com/docker/libcontainer/selinux"
	"github.com/golang/glog"
)

const (
	mountFlags          = uint(syscall.MS_NOEXEC | syscall.MS_NOSUID | syscall.MS_NODEV)
	defaultMountOptions = "mode=0755,size=10g"
)

// mountTmpfs mounts the node's tmpfs storage, if necessary
func (kl *Kubelet) mountTmpfsStorage() error {
	isMount, err := mount.IsMountPoint(kl.getTmpfsRootDir())
	if err != nil {
		return fmt.Errorf("error determining whether tmpfs root dir is a mountpoint: %v", err)
	}

	if isMount {
		glog.V(2).Infof("Tmpfs root dir is already a mountpoint, skipping mount: %v", kl.getTmpfsRootDir())
		return nil
	}

	// FUTURE: determine mount size from kubelet property
	options, err := kl.getTmpfsMountOptions()
	if err != nil {
		return fmt.Errorf("Error determining tmpfs mount options: %v", err)
	}

	glog.Infof("Mounting tmpfs root (%v) with options: %v", kl.getTmpfsRootDir(), options)

	err = kl.mounter.Mount("tmpfs", kl.getTmpfsRootDir(), "tmpfs", uintptr(mountFlags), options)
	if err != nil {
		return fmt.Errorf("error mounting tmpfs root directory (%v): %v", kl.getTmpfsRootDir(), err)
	}

	return nil
}

func (kl *Kubelet) getTmpfsMountOptions() (string, error) {
	if !selinux.SelinuxEnabled() {
		return defaultMountOptions, nil
	}

	rootContext, err := selinux.Getfilecon(kl.getRootDir())
	if err != nil {
		return "", err
	}

	// TODO: check out unexpected null char (or something)
	// in result of selinux.Getfilecon
	rootContext = rootContext[:len(rootContext)-1]

	return fmt.Sprintf("%v,rootcontext=\"%v\"", defaultMountOptions, rootContext), nil
}

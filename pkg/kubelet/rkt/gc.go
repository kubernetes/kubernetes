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

package rkt

import (
	"os/exec"

	"github.com/golang/glog"
)

const (
	// TODO(yifan): Merge with ContainerGCPolicy, i.e., derive
	// the grace period from MinAge in ContainerGCPolicy.
	//
	// Duration to wait before discarding inactive pods from garbage
	defaultGracePeriod = "1m"
	// Duration to wait before expiring prepared pods.
	defaultExpirePrepared = "1m"
)

// GarbageCollect collects the pods/containers. TODO(yifan): Enforce the gc policy.
func (r *Runtime) GarbageCollect() error {
	if err := exec.Command("systemctl", "reset-failed").Run(); err != nil {
		glog.Errorf("rkt: Failed to reset failed systemd services: %v", err)
	}
	if _, err := r.runCommand("gc", "--grace-period="+defaultGracePeriod, "--expire-prepared="+defaultExpirePrepared); err != nil {
		glog.Errorf("rkt: Failed to gc: %v", err)
		return err
	}
	return nil
}

// ImageManager manages and garbage collects the container images for rkt.
type ImageManager struct {
	runtime *Runtime
}

func NewImageManager(r *Runtime) *ImageManager {
	return &ImageManager{runtime: r}
}

// GarbageCollect collects the images. It is not implemented by rkt yet.
func (im *ImageManager) GarbageCollect() error {
	return nil
}

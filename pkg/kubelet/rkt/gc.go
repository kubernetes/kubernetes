/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

import "github.com/golang/glog"

// ImageManager manages and garbage collects the container images for rkt.
type ImageManager struct {
	runtime *Runtime
}

func NewImageManager(r *Runtime) *ImageManager {
	return &ImageManager{runtime: r}
}

// GarbageCollect collects the images.
// TODO(yifan): Enforce ImageGCPolicy.
func (im *ImageManager) GarbageCollect() error {
	if _, err := im.runtime.runCommand("image", "gc"); err != nil {
		glog.Errorf("rkt: Failed to gc image: %v", err)
		return err
	}
	return nil
}

// Start is a no-op for rkt as we don't need to mark unused images in kubelet.
func (im *ImageManager) Start() error {
	return nil
}

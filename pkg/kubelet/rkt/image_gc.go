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
import "k8s.io/kubernetes/pkg/kubelet/im"

// ImageGC manages and garbage collects the container images for rkt.
type realImageGC struct {
	runtime *Runtime
}

func NewImageGC(r *Runtime) im.ImageGC {
	return &realImageGC{runtime: r}
}

// GarbageCollect collects the images.
func (im *realImageGC) GarbageCollect() error {
	if _, err := im.runtime.runCommand("image", "gc"); err != nil {
		glog.Errorf("rkt: Failed to gc image: %v", err)
		return err
	}
	return nil
}

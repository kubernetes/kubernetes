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

// ImageManager manages and garbage collects the container images for rkt.
type ImageManager struct {
	runtime *runtime
}

func NewImageManager(r *runtime) *ImageManager {
	return &ImageManager{runtime: r}
}

// GarbageCollect collects the images. It is not implemented by rkt yet.
func (im *ImageManager) GarbageCollect() error {
	return nil
}

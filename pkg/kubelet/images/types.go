/*
Copyright 2016 The Kubernetes Authors All.
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

package images

import "k8s.io/kubernetes/pkg/api"

// ImageManager provides an interface to manage the lifecycle of images.
// Implementations of this interface are expected to deal with pulling (downloading),
// managing, and deleting container images.
// Implementations are expected to abstract the underlying runtimes.
// Implementations are expected to be thread safe.
type ImageManager interface {
	// EnsureImageExists ensures that image specified in `container` exists.
	EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string)

	// TODO(ronl): consolidating image managing and deleting operation in this interface
}

// ImagePuller wraps Runtime.PullImage() to pull a container image.
// It will check the presence of the image, and report the 'image pulling',
// 'image pulled' events correspondingly.
type imagePuller interface {
	pullImage(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string)
}

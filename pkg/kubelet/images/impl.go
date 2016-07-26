/*
Copyright 2016 The Kubernetes Authors.

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

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/util/flowcontrol"
)

type imageManager struct {
	recorder    record.EventRecorder
	runtime     kubecontainer.Runtime
	backOff     *flowcontrol.Backoff
	imagePuller imagePuller
}

var _ ImageManager = &imageManager{}

func NewImageManager(recorder record.EventRecorder, runtime kubecontainer.Runtime, imageBackOff *flowcontrol.Backoff, serialized bool) ImageManager {
	var imagePuller imagePuller
	if serialized {
		imagePuller = newSerializedImagePuller(recorder, runtime, imageBackOff)
	} else {
		imagePuller = newParallelImagePuller(recorder, runtime, imageBackOff)
	}
	return &imageManager{
		recorder:    recorder,
		runtime:     runtime,
		backOff:     imageBackOff,
		imagePuller: imagePuller,
	}
}

func (im *imageManager) EnsureImageExists(pod *api.Pod, container *api.Container, pullSecrets []api.Secret) (error, string) {
	return im.imagePuller.pullImage(pod, container, pullSecrets)
}

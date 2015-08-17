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

package kubelet

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/record"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"github.com/golang/glog"
)

// Kubelet-specific runtime hooks.
type kubeletRuntimeHooks struct {
	recorder record.EventRecorder
}

var _ kubecontainer.RuntimeHooks = &kubeletRuntimeHooks{}

func newKubeletRuntimeHooks(recorder record.EventRecorder) kubecontainer.RuntimeHooks {
	return &kubeletRuntimeHooks{
		recorder: recorder,
	}
}

func (kr *kubeletRuntimeHooks) ShouldPullImage(pod *api.Pod, container *api.Container, imagePresent bool) bool {
	if container.ImagePullPolicy == api.PullNever {
		return false
	}

	if container.ImagePullPolicy == api.PullAlways ||
		(container.ImagePullPolicy == api.PullIfNotPresent && (!imagePresent)) {
		return true
	}

	return false
}

func (kr *kubeletRuntimeHooks) ReportImagePull(pod *api.Pod, container *api.Container, pullError error) {
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		glog.Errorf("Couldn't make a ref to pod %q, container %q: '%v'", pod.Name, container.Name, err)
		return
	}

	if pullError != nil {
		kr.recorder.Eventf(ref, "failed", "Failed to pull image %q: %v", container.Image, pullError)
	} else {
		kr.recorder.Eventf(ref, "pulled", "Successfully pulled image %q", container.Image)
	}
}

/*
Copyright 2025 The Kubernetes Authors.

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

package prober

import (
	"context"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// RecordContainerEvent should be used by the prober for all container related events.
func RecordContainerEvent(recorder record.EventRecorder, ctx context.Context, pod *v1.Pod, container *v1.Container, eventType, reason, message string, args ...interface{}) {
	logger := klog.FromContext(ctx)
	ref, err := kubecontainer.GenerateContainerRef(pod, container)
	if err != nil {
		logger.Error(err, "Can't make a ref to pod and container", "pod", klog.KObj(pod), "containerName", container.Name)
		return
	}
	recorder.Eventf(ref, eventType, reason, message, args...)
}

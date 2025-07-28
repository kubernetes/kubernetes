//go:build windows
// +build windows

/*
Copyright 2024 The Kubernetes Authors.

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

package eviction

import (
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

func makeMemoryAvailableSignalObservation(logger klog.Logger, summary *statsapi.Summary) *signalObservation {
	logger.V(4).Info("Eviction manager: building memory signal observations for windows")
	sysContainer, err := getSysContainer(summary.Node.SystemContainers, statsapi.SystemContainerWindowsGlobalCommitMemory)
	if err != nil {
		logger.Error(err, "Eviction manager: failed to construct signal", "signal", evictionapi.SignalMemoryAvailable)
		return nil
	}
	if memory := sysContainer.Memory; memory != nil && memory.AvailableBytes != nil && memory.UsageBytes != nil {
		logger.V(4).Info(
			"Eviction manager: memory signal observations for windows",
			"Available", *memory.AvailableBytes,
			"Usage", *memory.UsageBytes)
		return &signalObservation{
			available: resource.NewQuantity(int64(*memory.AvailableBytes), resource.BinarySI),
			capacity:  resource.NewQuantity(int64(*memory.AvailableBytes+*memory.UsageBytes), resource.BinarySI),
			time:      memory.Time,
		}
	} else {
		return nil
	}
}

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
	"context"
	"fmt"
	"time"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/klog/v2"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
)

type windowsMemoryThresholdNotifier struct {
	threshold evictionapi.Threshold
	events    chan struct{}
	handler   func(string)
}

var _ ThresholdNotifier = &windowsMemoryThresholdNotifier{}

func NewMemoryThresholdNotifier(logger klog.Logger, threshold evictionapi.Threshold, cgroupRoot string, factory NotifierFactory, handler func(string)) (ThresholdNotifier, error) {
	logger.Info("Eviction manager: creating new WindowsMemoryThresholdNotifier")
	return &windowsMemoryThresholdNotifier{
		threshold: threshold,
		events:    make(chan struct{}),
		handler:   handler,
	}, nil
}

func (m *windowsMemoryThresholdNotifier) Start(ctx context.Context) {
	logger := klog.FromContext(ctx)
	logger.Info("Eviction manager: starting windowsMemoryThresholdNotifier", "notifier", m.Description())
	go func() {
		for true {
			time.Sleep(notifierRefreshInterval)
			m.checkMemoryUsage(logger)
		}
	}()

	for range m.events {
		m.handler(fmt.Sprintf("eviction manager: %s crossed", m.Description()))
	}
}

func (m *windowsMemoryThresholdNotifier) checkMemoryUsage(logger klog.Logger) {
	// Get global commit limit
	perfInfo, err := winstats.GetPerformanceInfo()
	if err != nil {
		logger.Error(err, "Eviction manager: error getting global memory status for node")
	}

	commmiLimitBytes := perfInfo.CommitLimitPages * perfInfo.PageSize
	capacity := resource.NewQuantity(int64(commmiLimitBytes), resource.BinarySI)
	evictionThresholdQuantity := evictionapi.GetThresholdQuantity(m.threshold.Value, capacity)

	commitTotalBytes := perfInfo.CommitTotalPages * perfInfo.PageSize
	commitAvailableBytes := commmiLimitBytes - commitTotalBytes

	if commitAvailableBytes <= uint64(evictionThresholdQuantity.Value()) {
		m.events <- struct{}{}
	}
}

func (m *windowsMemoryThresholdNotifier) UpdateThreshold(ctx context.Context, summary *statsapi.Summary) error {
	// Windows doesn't use cgroup notifiers to trigger eviction, so this function is a no-op.
	// Instead the go-routine set up in Start() will poll the system for memory usage and
	// trigger eviction when the threshold is crossed.
	return nil
}

func (m *windowsMemoryThresholdNotifier) Description() string {
	var hard, allocatable string
	if isHardEvictionThreshold(m.threshold) {
		hard = "hard"
	} else {
		hard = "soft"
	}
	if isAllocatableEvictionThreshold(m.threshold) {
		allocatable = "allocatable"
	}
	return fmt.Sprintf("%s %s memory eviction threshold", hard, allocatable)
}

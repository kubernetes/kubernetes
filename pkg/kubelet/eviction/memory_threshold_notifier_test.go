/*
Copyright 2018 The Kubernetes Authors.

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
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	gomock "go.uber.org/mock/gomock"
	"k8s.io/apimachinery/pkg/api/resource"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

const testCgroupPath = "/sys/fs/cgroups/memory"

func nodeSummary(available, workingSet, usage resource.Quantity, allocatable bool) *statsapi.Summary {
	availableBytes := uint64(available.Value())
	workingSetBytes := uint64(workingSet.Value())
	usageBytes := uint64(usage.Value())
	memoryStats := statsapi.MemoryStats{
		AvailableBytes:  &availableBytes,
		WorkingSetBytes: &workingSetBytes,
		UsageBytes:      &usageBytes,
	}
	if allocatable {
		return &statsapi.Summary{
			Node: statsapi.NodeStats{
				SystemContainers: []statsapi.ContainerStats{
					{
						Name:   statsapi.SystemContainerPods,
						Memory: &memoryStats,
					},
				},
			},
		}
	}
	return &statsapi.Summary{
		Node: statsapi.NodeStats{
			Memory: &memoryStats,
		},
	}
}

func newTestMemoryThresholdNotifier(threshold evictionapi.Threshold, factory NotifierFactory, handler func(string)) *memoryThresholdNotifier {
	return &memoryThresholdNotifier{
		threshold:  threshold,
		cgroupPath: testCgroupPath,
		events:     make(chan struct{}),
		factory:    factory,
		handler:    handler,
	}
}

func TestUpdateThreshold(t *testing.T) {
	testCases := []struct {
		description        string
		available          resource.Quantity
		workingSet         resource.Quantity
		usage              resource.Quantity
		evictionThreshold  evictionapi.Threshold
		expectedThreshold  resource.Quantity
		updateThresholdErr error
		expectErr          bool
	}{
		{
			description: "node level threshold",
			available:   resource.MustParse("3Gi"),
			usage:       resource.MustParse("2Gi"),
			workingSet:  resource.MustParse("1Gi"),
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			expectedThreshold:  resource.MustParse("4Gi"),
			updateThresholdErr: nil,
			expectErr:          false,
		},
		{
			description: "allocatable threshold",
			available:   resource.MustParse("4Gi"),
			usage:       resource.MustParse("3Gi"),
			workingSet:  resource.MustParse("1Gi"),
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalAllocatableMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			expectedThreshold:  resource.MustParse("6Gi"),
			updateThresholdErr: nil,
			expectErr:          false,
		},
		{
			description: "error updating node level threshold",
			available:   resource.MustParse("3Gi"),
			usage:       resource.MustParse("2Gi"),
			workingSet:  resource.MustParse("1Gi"),
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("1Gi"),
				},
			},
			expectedThreshold:  resource.MustParse("4Gi"),
			updateThresholdErr: fmt.Errorf("unexpected error"),
			expectErr:          true,
		},
	}

	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			notifierFactory := NewMockNotifierFactory(mockCtrl)
			notifier := NewMockCgroupNotifier(mockCtrl)

			m := newTestMemoryThresholdNotifier(tc.evictionThreshold, notifierFactory, nil)
			notifierFactory.EXPECT().NewCgroupNotifier(testCgroupPath, memoryUsageAttribute, tc.expectedThreshold.Value()).Return(notifier, tc.updateThresholdErr).Times(1)
			var events chan<- struct{} = m.events
			notifier.EXPECT().Start(events).Return().AnyTimes()
			err := m.UpdateThreshold(nodeSummary(tc.available, tc.workingSet, tc.usage, isAllocatableEvictionThreshold(tc.evictionThreshold)))
			if err != nil && !tc.expectErr {
				t.Errorf("Unexpected error updating threshold: %v", err)
			} else if err == nil && tc.expectErr {
				t.Errorf("Expected error updating threshold, but got nil")
			}
		})
	}
}

func TestStart(t *testing.T) {
	noResources := resource.MustParse("0")
	threshold := evictionapi.Threshold{
		Signal:   evictionapi.SignalMemoryAvailable,
		Operator: evictionapi.OpLessThan,
		Value: evictionapi.ThresholdValue{
			Quantity: &noResources,
		},
	}
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()
	notifierFactory := NewMockNotifierFactory(mockCtrl)
	notifier := NewMockCgroupNotifier(mockCtrl)

	var wg sync.WaitGroup
	wg.Add(4)
	m := newTestMemoryThresholdNotifier(threshold, notifierFactory, func(string) {
		wg.Done()
	})
	notifierFactory.EXPECT().NewCgroupNotifier(testCgroupPath, memoryUsageAttribute, int64(0)).Return(notifier, nil).Times(1)

	var events chan<- struct{} = m.events
	notifier.EXPECT().Start(events).DoAndReturn(func(events chan<- struct{}) {
		for i := 0; i < 4; i++ {
			events <- struct{}{}
		}
	})

	err := m.UpdateThreshold(nodeSummary(noResources, noResources, noResources, isAllocatableEvictionThreshold(threshold)))
	if err != nil {
		t.Errorf("Unexpected error updating threshold: %v", err)
	}

	go m.Start()

	wg.Wait()
}

func TestThresholdDescription(t *testing.T) {
	testCases := []struct {
		description        string
		evictionThreshold  evictionapi.Threshold
		expectedSubstrings []string
		omittedSubstrings  []string
	}{
		{
			description: "hard node level threshold",
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
			},
			expectedSubstrings: []string{"hard"},
			omittedSubstrings:  []string{"allocatable", "soft"},
		},
		{
			description: "soft node level threshold",
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
				GracePeriod: time.Minute * 2,
			},
			expectedSubstrings: []string{"soft"},
			omittedSubstrings:  []string{"allocatable", "hard"},
		},
		{
			description: "hard allocatable threshold",
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalAllocatableMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
				GracePeriod: time.Minute * 2,
			},
			expectedSubstrings: []string{"soft", "allocatable"},
			omittedSubstrings:  []string{"hard"},
		},
		{
			description: "soft allocatable threshold",
			evictionThreshold: evictionapi.Threshold{
				Signal:   evictionapi.SignalAllocatableMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: quantityMustParse("2Gi"),
				},
			},
			expectedSubstrings: []string{"hard", "allocatable"},
			omittedSubstrings:  []string{"soft"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			m := &memoryThresholdNotifier{
				notifier:   &MockCgroupNotifier{},
				threshold:  tc.evictionThreshold,
				cgroupPath: testCgroupPath,
			}
			desc := m.Description()
			for _, expected := range tc.expectedSubstrings {
				if !strings.Contains(desc, expected) {
					t.Errorf("expected description for notifier with threshold %+v to contain %s, but it did not", tc.evictionThreshold, expected)
				}
			}
			for _, omitted := range tc.omittedSubstrings {
				if strings.Contains(desc, omitted) {
					t.Errorf("expected description for notifier with threshold %+v NOT to contain %s, but it did", tc.evictionThreshold, omitted)
				}
			}
		})
	}
}

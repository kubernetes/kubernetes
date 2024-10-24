
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

package eviction

import (
	"context"
	"fmt"
	"runtime"
	"sort"
	"sync"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	corev1helpers "k8s.io/component-helpers/scheduling/corev1"
	statsapi "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/utils/clock"

	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	resourcehelper "k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/server/stats"
	kubelettypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	podCleanupTimeout  = 30 * time.Second
	podCleanupPollFreq = time.Second
)

const (
	// signalEphemeralContainerFsLimit is amount of storage available on filesystem requested by the container
	signalEphemeralContainerFsLimit string = "ephemeralcontainerfs.limit"
	// signalEphemeralPodFsLimit is amount of storage available on filesystem requested by the pod
	signalEphemeralPodFsLimit string = "ephemeralpodfs.limit"
	// signalEmptyDirFsLimit is amount of storage available on filesystem requested by an emptyDir
	signalEmptyDirFsLimit string = "emptydirfs.limit"
	// immediateEvictionGracePeriodSeconds is how long we give pods to shut down when we
	// need them to evict quickly due to resource pressure
	immediateEvictionGracePeriodSeconds = 1
)

// managerImpl implements Manager
type managerImpl struct {
	clock                         clock.WithTicker
	config                        Config
	killPodFunc                   KillPodFunc
	imageGC                       ImageGC
	containerGC                   ContainerGC
	sync.RWMutex
	nodeConditions                []v1.NodeConditionType
	nodeConditionsLastObservedAt  nodeConditionsObservedAt
	nodeRef                       *v1.ObjectReference
	recorder                      record.EventRecorder
	summaryProvider               stats.SummaryProvider
	thresholdsFirstObservedAt      thresholdsObservedAt
	thresholdsMet                 []evictionapi.Threshold
	signalToRankFunc              map[evictionapi.Signal]rankFunc
	signalToNodeReclaimFuncs      map[evictionapi.Signal]nodeReclaimFuncs
	lastObservations              signalObservations
	dedicatedImageFs              *bool
	splitContainerImageFs         *bool
	thresholdNotifiers            []ThresholdNotifier
	thresholdsLastUpdated         time.Time
	localStorageCapacityIsolation bool
}

// ensure it implements the required interface
var _ Manager = &managerImpl{}

// NewManager returns a configured Manager and an associated admission handler to enforce eviction configuration.
func NewManager(
	summaryProvider stats.SummaryProvider,
	config Config,
	killPodFunc KillPodFunc,
	imageGC ImageGC,
	containerGC ContainerGC,
	recorder record.EventRecorder,
	nodeRef *v1.ObjectReference,
	clock clock.WithTicker,
	localStorageCapacityIsolation bool,
) (Manager, lifecycle.PodAdmitHandler) {
	manager := &managerImpl{
		clock:                         clock,
		killPodFunc:                   killPodFunc,
		imageGC:                       imageGC,
		containerGC:                   containerGC,
		config:                        config,
		recorder:                      recorder,
		summaryProvider:               summaryProvider,
		nodeRef:                       nodeRef,
		nodeConditionsLastObservedAt:  nodeConditionsObservedAt{},
		thresholdsFirstObservedAt:     thresholdsObservedAt{},
		dedicatedImageFs:              nil,
		splitContainerImageFs:         nil,
		thresholdNotifiers:            []ThresholdNotifier{},
		localStorageCapacityIsolation: localStorageCapacityIsolation,
	}
	return manager, manager
}

// ThresholdsMet checks if the system metrics (memory, disk usage, etc.) meet the eviction thresholds.
func (m *managerImpl) thresholdsMet() ([]evictionapi.Threshold, bool) {
	metrics := m.summaryProvider.GetStats()
	memoryPressure := metrics.Memory.WorkingSetBytes > m.config.MemoryEvictionThreshold
	diskPressure := metrics.FsInfo.AvailableBytes < m.config.DiskEvictionThreshold

	if memoryPressure || diskPressure {
		thresholds := []evictionapi.Threshold{
			{
				Signal: evictionapi.SignalMemoryAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: m.config.MemoryEvictionThreshold,
				},
			},
			{
				Signal: evictionapi.SignalNodeFsAvailable,
				Operator: evictionapi.OpLessThan,
				Value: evictionapi.ThresholdValue{
					Quantity: m.config.DiskEvictionThreshold,
				},
			},
		}
		return thresholds, true
	}
	return nil, false
}

// Evict pods based on the updated memory and disk pressure handling
func (m *managerImpl) StartManager(ctx context.Context) {
	for {
		select {
		case <-ctx.Done():
			return
		case <-time.After(10 * time.Second):
			thresholds, shouldEvict := m.thresholdsMet()
			if shouldEvict {
				fmt.Println("Eviction triggered based on thresholds:", thresholds)

				// Implement the eviction logic here
				activePods := podFunc()  // Function to get active pods
				statsFunc := m.summaryProvider.GetStats

				// Rank the pods for eviction
				rank, ok := m.signalToRankFunc[thresholds[0].Signal]
				if ok {
					rank(activePods, statsFunc)
				}

				// Evict the pod with the lowest priority
				for _, pod := range activePods {
					gracePeriodOverride := int64(immediateEvictionGracePeriodSeconds)
					if m.evictPod(pod, gracePeriodOverride, "Evicting due to resource pressure", nil, nil) {
						fmt.Printf("Evicted pod: %s
", pod.Name)
						break
					}
				}
			}
		}
	}
}

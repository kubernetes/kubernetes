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
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	statsapi "k8s.io/kubernetes/pkg/kubelet/apis/stats/v1alpha1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
)

// fsStatsType defines the types of filesystem stats to collect.
type fsStatsType string

const (
	// fsStatsLocalVolumeSource identifies stats for pod local volume sources.
	fsStatsLocalVolumeSource fsStatsType = "localVolumeSource"
	// fsStatsLogs identifies stats for pod logs.
	fsStatsLogs fsStatsType = "logs"
	// fsStatsRoot identifies stats for pod container writable layers.
	fsStatsRoot fsStatsType = "root"
)

// Config holds information about how eviction is configured.
type Config struct {
	// PressureTransitionPeriod is duration the kubelet has to wait before transititioning out of a pressure condition.
	PressureTransitionPeriod time.Duration
	// Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
	MaxPodGracePeriodSeconds int64
	// Thresholds define the set of conditions monitored to trigger eviction.
	Thresholds []evictionapi.Threshold
	// KernelMemcgNotification if true will integrate with the kernel memcg notification to determine if memory thresholds are crossed.
	KernelMemcgNotification bool
}

// Manager evaluates when an eviction threshold for node stability has been met on the node.
type Manager interface {
	// Start starts the control loop to monitor eviction thresholds at specified interval.
	Start(diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc, podCleanedUpFunc PodCleanedUpFunc, capacityProvider CapacityProvider, monitoringInterval time.Duration)

	// IsUnderMemoryPressure returns true if the node is under memory pressure.
	IsUnderMemoryPressure() bool

	// IsUnderDiskPressure returns true if the node is under disk pressure.
	IsUnderDiskPressure() bool
}

// DiskInfoProvider is responsible for informing the manager how disk is configured.
type DiskInfoProvider interface {
	// HasDedicatedImageFs returns true if the imagefs is on a separate device from the rootfs.
	HasDedicatedImageFs() (bool, error)
}

// CapacityProvider is responsible for providing the resource capacity and reservation information
type CapacityProvider interface {
	// GetCapacity returns the amount of compute resources tracked by container manager available on the node.
	GetCapacity() v1.ResourceList
	// GetNodeAllocatable returns the amount of compute resources that have to be reserved from scheduling.
	GetNodeAllocatableReservation() v1.ResourceList
}

// ImageGC is responsible for performing garbage collection of unused images.
type ImageGC interface {
	// DeleteUnusedImages deletes unused images and returns the number of bytes freed, and an error.
	// This returns the bytes freed even if an error is returned.
	DeleteUnusedImages() (int64, error)
}

// ContainerGC is responsible for performing garbage collection of unused containers.
type ContainerGC interface {
	// DeleteAllUnusedContainers deletes all unused containers, even those that belong to pods that are terminated, but not deleted.
	// It returns an error if it is unsuccessful.
	DeleteAllUnusedContainers() error
}

// KillPodFunc kills a pod.
// The pod status is updated, and then it is killed with the specified grace period.
// This function must block until either the pod is killed or an error is encountered.
// Arguments:
// pod - the pod to kill
// status - the desired status to associate with the pod (i.e. why its killed)
// gracePeriodOverride - the grace period override to use instead of what is on the pod spec
type KillPodFunc func(pod *v1.Pod, status v1.PodStatus, gracePeriodOverride *int64) error

// ActivePodsFunc returns pods bound to the kubelet that are active (i.e. non-terminal state)
type ActivePodsFunc func() []*v1.Pod

// PodCleanedUpFunc returns true if all resources associated with a pod have been reclaimed.
type PodCleanedUpFunc func(*v1.Pod) bool

// statsFunc returns the usage stats if known for an input pod.
type statsFunc func(pod *v1.Pod) (statsapi.PodStats, bool)

// rankFunc sorts the pods in eviction order
type rankFunc func(pods []*v1.Pod, stats statsFunc)

// signalObservation is the observed resource usage
type signalObservation struct {
	// The resource capacity
	capacity *resource.Quantity
	// The available resource
	available *resource.Quantity
	// Time at which the observation was taken
	time metav1.Time
}

// signalObservations maps a signal to an observed quantity
type signalObservations map[evictionapi.Signal]signalObservation

// thresholdsObservedAt maps a threshold to a time that it was observed
type thresholdsObservedAt map[evictionapi.Threshold]time.Time

// nodeConditionsObservedAt maps a node condition to a time that it was observed
type nodeConditionsObservedAt map[v1.NodeConditionType]time.Time

// nodeReclaimFunc is a function that knows how to reclaim a resource from the node without impacting pods.
// Returns the quantity of resources reclaimed and an error, if applicable.
// nodeReclaimFunc return the resources reclaimed even if an error occurs.
type nodeReclaimFunc func() (*resource.Quantity, error)

// nodeReclaimFuncs is an ordered list of nodeReclaimFunc
type nodeReclaimFuncs []nodeReclaimFunc

// thresholdNotifierHandlerFunc is a function that takes action in response to a crossed threshold
type thresholdNotifierHandlerFunc func(thresholdDescription string)

// ThresholdNotifier notifies the user when an attribute crosses a threshold value
type ThresholdNotifier interface {
	Start(stopCh <-chan struct{})
}

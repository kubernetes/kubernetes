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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	statsapi "k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/stats"
)

// Signal defines a signal that can trigger eviction of pods on a node.
type Signal string

const (
	// SignalMemoryAvailable is memory available (i.e. capacity - workingSet), in bytes.
	SignalMemoryAvailable Signal = "memory.available"
	// SignalNodeFsAvailable is amount of storage available on filesystem that kubelet uses for volumes, daemon logs, etc.
	SignalNodeFsAvailable Signal = "nodefs.available"
	// SignalNodeFsInodesFree is amount of inodes available on filesystem that kubelet uses for volumes, daemon logs, etc.
	SignalNodeFsInodesFree Signal = "nodefs.inodesFree"
	// SignalImageFsAvailable is amount of storage available on filesystem that container runtime uses for storing images and container writable layers.
	SignalImageFsAvailable Signal = "imagefs.available"
	// SignalImageFsInodesFree is amount of inodes available on filesystem that container runtime uses for storing images and container writeable layers.
	SignalImageFsInodesFree Signal = "imagefs.inodesFree"
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

// ThresholdOperator is the operator used to express a Threshold.
type ThresholdOperator string

const (
	// OpLessThan is the operator that expresses a less than operator.
	OpLessThan ThresholdOperator = "LessThan"
)

// Config holds information about how eviction is configured.
type Config struct {
	// PressureTransitionPeriod is duration the kubelet has to wait before transititioning out of a pressure condition.
	PressureTransitionPeriod time.Duration
	// Maximum allowed grace period (in seconds) to use when terminating pods in response to a soft eviction threshold being met.
	MaxPodGracePeriodSeconds int64
	// Thresholds define the set of conditions monitored to trigger eviction.
	Thresholds []Threshold
	// KernelMemcgNotification if true will integrate with the kernel memcg notification to determine if memory thresholds are crossed.
	KernelMemcgNotification bool
}

// ThresholdValue is a value holder that abstracts literal versus percentage based quantity
type ThresholdValue struct {
	// The following fields are exclusive. Only the topmost non-zero field is used.

	// Quantity is a quantity associated with the signal that is evaluated against the specified operator.
	Quantity *resource.Quantity
	// Percentage represents the usage percentage over the total resource that is evaluated against the specified operator.
	Percentage float32
}

// Threshold defines a metric for when eviction should occur.
type Threshold struct {
	// Signal defines the entity that was measured.
	Signal Signal
	// Operator represents a relationship of a signal to a value.
	Operator ThresholdOperator
	// Value is the threshold the resource is evaluated against.
	Value ThresholdValue
	// GracePeriod represents the amount of time that a threshold must be met before eviction is triggered.
	GracePeriod time.Duration
	// MinReclaim represents the minimum amount of resource to reclaim if the threshold is met.
	MinReclaim *ThresholdValue
}

// Manager evaluates when an eviction threshold for node stability has been met on the node.
type Manager interface {
	// Start starts the control loop to monitor eviction thresholds at specified interval.
	Start(diskInfoProvider DiskInfoProvider, podFunc ActivePodsFunc, monitoringInterval time.Duration) error

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

// ImageGC is responsible for performing garbage collection of unused images.
type ImageGC interface {
	// DeleteUnusedImages deletes unused images and returns the number of bytes freed, or an error.
	DeleteUnusedImages() (int64, error)
}

// KillPodFunc kills a pod.
// The pod status is updated, and then it is killed with the specified grace period.
// This function must block until either the pod is killed or an error is encountered.
// Arguments:
// pod - the pod to kill
// status - the desired status to associate with the pod (i.e. why its killed)
// gracePeriodOverride - the grace period override to use instead of what is on the pod spec
type KillPodFunc func(pod *api.Pod, status api.PodStatus, gracePeriodOverride *int64) error

// ActivePodsFunc returns pods bound to the kubelet that are active (i.e. non-terminal state)
type ActivePodsFunc func() []*api.Pod

// statsFunc returns the usage stats if known for an input pod.
type statsFunc func(pod *api.Pod) (statsapi.PodStats, bool)

// rankFunc sorts the pods in eviction order
type rankFunc func(pods []*api.Pod, stats statsFunc)

// signalObservation is the observed resource usage
type signalObservation struct {
	// The resource capacity
	capacity *resource.Quantity
	// The available resource
	available *resource.Quantity
	// Time at which the observation was taken
	time unversioned.Time
}

// signalObservations maps a signal to an observed quantity
type signalObservations map[Signal]signalObservation

// thresholdsObservedAt maps a threshold to a time that it was observed
type thresholdsObservedAt map[Threshold]time.Time

// nodeConditionsObservedAt maps a node condition to a time that it was observed
type nodeConditionsObservedAt map[api.NodeConditionType]time.Time

// nodeReclaimFunc is a function that knows how to reclaim a resource from the node without impacting pods.
type nodeReclaimFunc func() (*resource.Quantity, error)

// nodeReclaimFuncs is an ordered list of nodeReclaimFunc
type nodeReclaimFuncs []nodeReclaimFunc

// thresholdNotifierHandlerFunc is a function that takes action in response to a crossed threshold
type thresholdNotifierHandlerFunc func(thresholdDescription string)

// ThresholdNotifier notifies the user when an attribute crosses a threshold value
type ThresholdNotifier interface {
	Start(stopCh <-chan struct{})
}

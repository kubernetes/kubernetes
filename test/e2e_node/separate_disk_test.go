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

package e2enode

import (
	"context"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	kubeletstatsv1alpha1 "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/design-proposals-archive/blob/main/node/kubelet-eviction.md
// Stats is best effort and we evict based on stats being successful

// Container runtime filesystem should display different stats for imagefs and nodefs
var _ = SIGDescribe("Summary", feature.SeparateDisk, func() {
	f := framework.NewDefaultFramework("summary-test")
	f.It("should display different stats for imagefs and nodefs", func(ctx context.Context) {
		summary := eventuallyGetSummary(ctx)
		// Available and Capacity are the most useful to tell difference
		gomega.Expect(summary.Node.Fs.AvailableBytes).ToNot(gomega.Equal(summary.Node.Runtime.ImageFs.AvailableBytes))
		gomega.Expect(summary.Node.Fs.CapacityBytes).ToNot(gomega.Equal(summary.Node.Runtime.ImageFs.CapacityBytes))

	})
})

// Node disk pressure is induced by consuming all inodes on the Writeable Layer (imageFS).
var _ = SIGDescribe("InodeEviction", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.SeparateDisk, func() {
	testRunner(
		framework.NewDefaultFramework("inode-eviction-test"),
		EvictionTestConfig{
			Signal:                  string(evictionapi.SignalImageFsInodesFree),
			PressureTimeout:         15 * time.Minute,
			ExpectedNodeCondition:   v1.NodeDiskPressure,
			ExpectedStarvedResource: resourceInodes,
			ResourceThreshold:       uint64(200000), // Inodes consumed
			IsHardEviction:          true,
			MetricsLogger:           logInodeMetrics,
			ResourceGetter: func(summary *kubeletstatsv1alpha1.Summary) uint64 {
				return *(summary.Node.Runtime.ImageFs.InodesFree)
			},
		},
		[]podEvictSpec{
			{
				evictionPriority: 1,
				pod:              inodeConsumingPod("container-inode-hog", lotsOfFiles, nil),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
})

// LocalStorageEviction tests that the node responds to node disk pressure by evicting only responsible pods
// Disk pressure is induced by running pods which consume disk space, which exceed the soft eviction threshold.
// Note: This test's purpose is to test Soft Evictions.  Local storage was chosen since it is the least costly to run.
var _ = SIGDescribe("LocalStorageSoftEviction", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.SeparateDisk, func() {
	diskConsumed := resource.MustParse("4Gi")
	testRunner(
		framework.NewDefaultFramework("local-storage-imagefs-soft-test"),
		EvictionTestConfig{
			Signal:                  string(evictionapi.SignalImageFsAvailable),
			PressureTimeout:         10 * time.Minute,
			ExpectedNodeCondition:   v1.NodeDiskPressure,
			ExpectedStarvedResource: v1.ResourceEphemeralStorage,
			ResourceThreshold:       uint64(diskConsumed.Value()), // local storage
			IsHardEviction:          false,
			EvictionGracePeriod:     "1m",
			MetricsLogger:           logDiskMetrics,
			ResourceGetter: func(summary *kubeletstatsv1alpha1.Summary) uint64 {
				return *summary.Node.Runtime.ImageFs.AvailableBytes
			},
		},
		[]podEvictSpec{
			{
				evictionPriority: 1,
				pod:              diskConsumingPod("best-effort-disk", lotsOfDisk, nil, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
})

// LocalStorageCapacityIsolationEviction tests that container and volume local storage limits are enforced through evictions
// removed localstoragecapacityisolation feature gate here as its not a feature gate anymore
var _ = SIGDescribe("LocalStorageCapacityIsolationEviction", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.SeparateDisk, func() {
	sizeLimit := resource.MustParse("40Mi")
	useOverLimit := 41  /* Mb */
	useUnderLimit := 39 /* Mb */
	containerLimit := v1.ResourceList{v1.ResourceEphemeralStorage: sizeLimit}

	testRunner(
		framework.NewDefaultFramework("localstorage-eviction-test"),
		EvictionTestConfig{
			Signal:                  string(evictionapi.SignalMemoryAvailable),
			PressureTimeout:         10 * time.Minute,
			ExpectedNodeCondition:   noPressure,
			ExpectedStarvedResource: noStarvedResource,
			IsHardEviction:          true,
			ThresholdPercentage:     "0%", // Disabling this threshold to focus on pod-level limits
			MetricsLogger:           logDiskMetrics,
			ResourceGetter: func(summary *kubeletstatsv1alpha1.Summary) uint64 {
				// We're not using node-level resource checks for this test
				// Just need a non-zero value to pass the resource check
				return 1024 * 1024 * 1024 // 1 GB (arbitrary non-zero value)
			},
		},
		[]podEvictSpec{
			{
				evictionPriority: 1, // This pod should be evicted because emptyDir (default storage type) usage violation
				pod: diskConsumingPod("emptydir-disk-sizelimit", useOverLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 1, // This pod should cross the container limit by writing to its writable layer.
				pod:              diskConsumingPod("container-disk-limit", useOverLimit, nil, v1.ResourceRequirements{Limits: containerLimit}),
			},
			{
				evictionPriority: 1, // This pod should hit the container limit by writing to an emptydir
				pod: diskConsumingPod("container-emptydir-disk-limit", useOverLimit, &v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
					v1.ResourceRequirements{Limits: containerLimit}),
			},
			{
				evictionPriority: 0, // This pod should not be evicted because MemoryBackedVolumes cannot use more space than is allocated to them since SizeMemoryBackedVolumes was enabled
				pod: diskConsumingPod("emptydir-memory-sizelimit", useOverLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{Medium: "Memory", SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0, // This pod should not be evicted because it uses less than its limit
				pod: diskConsumingPod("emptydir-disk-below-sizelimit", useUnderLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0, // This pod should not be evicted because it uses less than its limit
				pod:              diskConsumingPod("container-disk-below-sizelimit", useUnderLimit, nil, v1.ResourceRequirements{Limits: containerLimit}),
			},
		})
})

// ImageStorageVolumeEviction tests that the node responds to node disk pressure by evicting pods.
// Volumes write to the node filesystem so we are testing eviction on nodefs even if it
// exceeds imagefs limits.
var _ = SIGDescribe("ImageStorageVolumeEviction", framework.WithSlow(), framework.WithSerial(), framework.WithDisruptive(), feature.SeparateDisk, func() {
	testRunner(
		framework.NewDefaultFramework("exceed-nodefs-test"),
		EvictionTestConfig{
			Signal:                  string(evictionapi.SignalNodeFsAvailable),
			PressureTimeout:         15 * time.Minute,
			ExpectedNodeCondition:   v1.NodeDiskPressure,
			ExpectedStarvedResource: v1.ResourceEphemeralStorage,
			IsHardEviction:          true,
			ThresholdPercentage:     "50%", // Use percentage instead of absolute threshold
			MetricsLogger:           logDiskMetrics,
			ResourceGetter: func(summary *kubeletstatsv1alpha1.Summary) uint64 {
				return *summary.Node.Fs.AvailableBytes
			},
		},
		[]podEvictSpec{
			{
				evictionPriority: 1, // This pod should exceed disk capacity on nodefs since writing to a volume
				pod: diskConsumingPod("container-emptydir-disk-limit", 16000, &v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
					v1.ResourceRequirements{}),
			},
		})
})

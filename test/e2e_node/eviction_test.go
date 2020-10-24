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

package e2enode

import (
	"context"
	"fmt"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1 "k8s.io/api/scheduling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	kubeletstatsv1alpha1 "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	evictionapi "k8s.io/kubernetes/pkg/kubelet/eviction/api"
	kubeletmetrics "k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	testutils "k8s.io/kubernetes/test/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

// Eviction Policy is described here:
// https://github.com/kubernetes/community/blob/master/contributors/design-proposals/node/kubelet-eviction.md

const (
	postTestConditionMonitoringPeriod = 1 * time.Minute
	evictionPollInterval              = 2 * time.Second
	pressureDisappearTimeout          = 1 * time.Minute
	// pressure conditions often surface after evictions because the kubelet only updates
	// node conditions periodically.
	// we wait this period after evictions to make sure that we wait out this delay
	pressureDelay     = 20 * time.Second
	testContextFmt    = "when we run containers that should cause %s"
	noPressure        = v1.NodeConditionType("NoPressure")
	lotsOfDisk        = 10240      // 10 Gb in Mb
	lotsOfFiles       = 1000000000 // 1 billion
	resourceInodes    = v1.ResourceName("inodes")
	noStarvedResource = v1.ResourceName("none")
)

// InodeEviction tests that the node responds to node disk pressure by evicting only responsible pods.
// Node disk pressure is induced by consuming all inodes on the node.
var _ = framework.KubeDescribe("InodeEviction [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("inode-eviction-test")
	expectedNodeCondition := v1.NodeDiskPressure
	expectedStarvedResource := resourceInodes
	pressureTimeout := 15 * time.Minute
	inodesConsumed := uint64(200000)
	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			// Set the eviction threshold to inodesFree - inodesConsumed, so that using inodesConsumed causes an eviction.
			summary := eventuallyGetSummary()
			inodesFree := *summary.Node.Fs.InodesFree
			if inodesFree <= inodesConsumed {
				e2eskipper.Skipf("Too few inodes free on the host for the InodeEviction test to run")
			}
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsInodesFree): fmt.Sprintf("%d", inodesFree-inodesConsumed)}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logInodeMetrics, []podEvictSpec{
			{
				evictionPriority: 1,
				pod:              inodeConsumingPod("container-inode-hog", lotsOfFiles, nil),
			},
			{
				evictionPriority: 1,
				pod:              inodeConsumingPod("volume-inode-hog", lotsOfFiles, &v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}}),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
	})
})

// ImageGCNoEviction tests that the node does not evict pods when inodes are consumed by images
// Disk pressure is induced by pulling large images
var _ = framework.KubeDescribe("ImageGCNoEviction [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("image-gc-eviction-test")
	pressureTimeout := 10 * time.Minute
	expectedNodeCondition := v1.NodeDiskPressure
	expectedStarvedResource := resourceInodes
	inodesConsumed := uint64(100000)
	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			// Set the eviction threshold to inodesFree - inodesConsumed, so that using inodesConsumed causes an eviction.
			summary := eventuallyGetSummary()
			inodesFree := *summary.Node.Fs.InodesFree
			if inodesFree <= inodesConsumed {
				e2eskipper.Skipf("Too few inodes free on the host for the InodeEviction test to run")
			}
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsInodesFree): fmt.Sprintf("%d", inodesFree-inodesConsumed)}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		// Consume enough inodes to induce disk pressure,
		// but expect that image garbage collection can reduce it enough to avoid an eviction
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logDiskMetrics, []podEvictSpec{
			{
				evictionPriority: 0,
				pod:              inodeConsumingPod("container-inode", 110000, nil),
			},
		})
	})
})

// MemoryAllocatableEviction tests that the node responds to node memory pressure by evicting only responsible pods.
// Node memory pressure is only encountered because we reserve the majority of the node's capacity via kube-reserved.
var _ = framework.KubeDescribe("MemoryAllocatableEviction [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("memory-allocatable-eviction-test")
	expectedNodeCondition := v1.NodeMemoryPressure
	expectedStarvedResource := v1.ResourceMemory
	pressureTimeout := 10 * time.Minute
	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			// Set large system and kube reserved values to trigger allocatable thresholds far before hard eviction thresholds.
			kubeReserved := getNodeCPUAndMemoryCapacity(f)[v1.ResourceMemory]
			// The default hard eviction threshold is 250Mb, so Allocatable = Capacity - Reserved - 250Mb
			// We want Allocatable = 50Mb, so set Reserved = Capacity - Allocatable - 250Mb = Capacity - 300Mb
			kubeReserved.Sub(resource.MustParse("300Mi"))
			initialConfig.KubeReserved = map[string]string{
				string(v1.ResourceMemory): kubeReserved.String(),
			}
			initialConfig.EnforceNodeAllocatable = []string{kubetypes.NodeAllocatableEnforcementKey}
			initialConfig.CgroupsPerQOS = true
		})
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logMemoryMetrics, []podEvictSpec{
			{
				evictionPriority: 1,
				pod:              getMemhogPod("memory-hog-pod", "memory-hog", v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
	})
})

// LocalStorageEviction tests that the node responds to node disk pressure by evicting only responsible pods
// Disk pressure is induced by running pods which consume disk space.
var _ = framework.KubeDescribe("LocalStorageEviction [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("localstorage-eviction-test")
	pressureTimeout := 10 * time.Minute
	expectedNodeCondition := v1.NodeDiskPressure
	expectedStarvedResource := v1.ResourceEphemeralStorage
	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			diskConsumed := resource.MustParse("200Mi")
			summary := eventuallyGetSummary()
			availableBytes := *(summary.Node.Fs.AvailableBytes)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsAvailable): fmt.Sprintf("%d", availableBytes-uint64(diskConsumed.Value()))}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logDiskMetrics, []podEvictSpec{
			{
				evictionPriority: 1,
				pod:              diskConsumingPod("container-disk-hog", lotsOfDisk, nil, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
	})
})

// LocalStorageEviction tests that the node responds to node disk pressure by evicting only responsible pods
// Disk pressure is induced by running pods which consume disk space, which exceed the soft eviction threshold.
// Note: This test's purpose is to test Soft Evictions.  Local storage was chosen since it is the least costly to run.
var _ = framework.KubeDescribe("LocalStorageSoftEviction [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("localstorage-eviction-test")
	pressureTimeout := 10 * time.Minute
	expectedNodeCondition := v1.NodeDiskPressure
	expectedStarvedResource := v1.ResourceEphemeralStorage
	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			diskConsumed := resource.MustParse("200Mi")
			summary := eventuallyGetSummary()
			availableBytes := *(summary.Node.Fs.AvailableBytes)
			if availableBytes <= uint64(diskConsumed.Value()) {
				e2eskipper.Skipf("Too little disk free on the host for the LocalStorageSoftEviction test to run")
			}
			initialConfig.EvictionSoft = map[string]string{string(evictionapi.SignalNodeFsAvailable): fmt.Sprintf("%d", availableBytes-uint64(diskConsumed.Value()))}
			initialConfig.EvictionSoftGracePeriod = map[string]string{string(evictionapi.SignalNodeFsAvailable): "1m"}
			// Defer to the pod default grace period
			initialConfig.EvictionMaxPodGracePeriod = 30
			initialConfig.EvictionMinimumReclaim = map[string]string{}
			// Ensure that pods are not evicted because of the eviction-hard threshold
			// setting a threshold to 0% disables; non-empty map overrides default value (necessary due to omitempty)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%"}
		})
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logDiskMetrics, []podEvictSpec{
			{
				evictionPriority: 1,
				pod:              diskConsumingPod("container-disk-hog", lotsOfDisk, nil, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
		})
	})
})

// LocalStorageCapacityIsolationEviction tests that container and volume local storage limits are enforced through evictions
var _ = framework.KubeDescribe("LocalStorageCapacityIsolationEviction [Slow] [Serial] [Disruptive] [Feature:LocalStorageCapacityIsolation][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("localstorage-eviction-test")
	evictionTestTimeout := 10 * time.Minute
	ginkgo.Context(fmt.Sprintf(testContextFmt, "evictions due to pod local storage violations"), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			// setting a threshold to 0% disables; non-empty map overrides default value (necessary due to omitempty)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalMemoryAvailable): "0%"}
		})
		sizeLimit := resource.MustParse("100Mi")
		useOverLimit := 101 /* Mb */
		useUnderLimit := 99 /* Mb */
		containerLimit := v1.ResourceList{v1.ResourceEphemeralStorage: sizeLimit}

		runEvictionTest(f, evictionTestTimeout, noPressure, noStarvedResource, logDiskMetrics, []podEvictSpec{
			{
				evictionPriority: 1, // This pod should be evicted because emptyDir (default storage type) usage violation
				pod: diskConsumingPod("emptydir-disk-sizelimit", useOverLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{SizeLimit: &sizeLimit},
				}, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 1, // This pod should be evicted because of memory emptyDir usage violation
				pod: diskConsumingPod("emptydir-memory-sizelimit", useOverLimit, &v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{Medium: "Memory", SizeLimit: &sizeLimit},
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
})

// PriorityMemoryEvictionOrdering tests that the node responds to node memory pressure by evicting pods.
// This test tests that the guaranteed pod is never evicted, and that the lower-priority pod is evicted before
// the higher priority pod.
var _ = framework.KubeDescribe("PriorityMemoryEvictionOrdering [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("priority-memory-eviction-ordering-test")
	expectedNodeCondition := v1.NodeMemoryPressure
	expectedStarvedResource := v1.ResourceMemory
	pressureTimeout := 10 * time.Minute

	highPriorityClassName := f.BaseName + "-high-priority"
	highPriority := int32(999999999)

	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			memoryConsumed := resource.MustParse("600Mi")
			summary := eventuallyGetSummary()
			availableBytes := *(summary.Node.Memory.AvailableBytes)
			if availableBytes <= uint64(memoryConsumed.Value()) {
				e2eskipper.Skipf("Too little memory free on the host for the PriorityMemoryEvictionOrdering test to run")
			}
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalMemoryAvailable): fmt.Sprintf("%d", availableBytes-uint64(memoryConsumed.Value()))}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		ginkgo.BeforeEach(func() {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: highPriorityClassName}, Value: highPriority}, metav1.CreateOptions{})
			framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		})
		ginkgo.AfterEach(func() {
			err := f.ClientSet.SchedulingV1().PriorityClasses().Delete(context.TODO(), highPriorityClassName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})
		specs := []podEvictSpec{
			{
				evictionPriority: 2,
				pod:              getMemhogPod("memory-hog-pod", "memory-hog", v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 1,
				pod:              getMemhogPod("high-priority-memory-hog-pod", "high-priority-memory-hog", v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				pod: getMemhogPod("guaranteed-pod", "guaranteed-pod", v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("300Mi"),
					},
					Limits: v1.ResourceList{
						v1.ResourceMemory: resource.MustParse("300Mi"),
					},
				}),
			},
		}
		specs[1].pod.Spec.PriorityClassName = highPriorityClassName
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logMemoryMetrics, specs)
	})
})

// PriorityLocalStorageEvictionOrdering tests that the node responds to node disk pressure by evicting pods.
// This test tests that the guaranteed pod is never evicted, and that the lower-priority pod is evicted before
// the higher priority pod.
var _ = framework.KubeDescribe("PriorityLocalStorageEvictionOrdering [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("priority-disk-eviction-ordering-test")
	expectedNodeCondition := v1.NodeDiskPressure
	expectedStarvedResource := v1.ResourceEphemeralStorage
	pressureTimeout := 10 * time.Minute

	highPriorityClassName := f.BaseName + "-high-priority"
	highPriority := int32(999999999)

	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			diskConsumed := resource.MustParse("350Mi")
			summary := eventuallyGetSummary()
			availableBytes := *(summary.Node.Fs.AvailableBytes)
			if availableBytes <= uint64(diskConsumed.Value()) {
				e2eskipper.Skipf("Too little disk free on the host for the PriorityLocalStorageEvictionOrdering test to run")
			}
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalNodeFsAvailable): fmt.Sprintf("%d", availableBytes-uint64(diskConsumed.Value()))}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		ginkgo.BeforeEach(func() {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: highPriorityClassName}, Value: highPriority}, metav1.CreateOptions{})
			framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		})
		ginkgo.AfterEach(func() {
			err := f.ClientSet.SchedulingV1().PriorityClasses().Delete(context.TODO(), highPriorityClassName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})
		specs := []podEvictSpec{
			{
				evictionPriority: 2,
				pod:              diskConsumingPod("best-effort-disk", lotsOfDisk, nil, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 1,
				pod:              diskConsumingPod("high-priority-disk", lotsOfDisk, nil, v1.ResourceRequirements{}),
			},
			{
				evictionPriority: 0,
				// Only require 99% accuracy (297/300 Mb) because on some OS distributions, the file itself (excluding contents), consumes disk space.
				pod: diskConsumingPod("guaranteed-disk", 297 /* Mb */, nil, v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("300Mi"),
					},
					Limits: v1.ResourceList{
						v1.ResourceEphemeralStorage: resource.MustParse("300Mi"),
					},
				}),
			},
		}
		specs[1].pod.Spec.PriorityClassName = highPriorityClassName
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logDiskMetrics, specs)
	})
})

// PriorityPidEvictionOrdering tests that the node emits pid pressure in response to a fork bomb, and evicts pods by priority
var _ = framework.KubeDescribe("PriorityPidEvictionOrdering [Slow] [Serial] [Disruptive][NodeFeature:Eviction]", func() {
	f := framework.NewDefaultFramework("pidpressure-eviction-test")
	pressureTimeout := 2 * time.Minute
	expectedNodeCondition := v1.NodePIDPressure
	expectedStarvedResource := noStarvedResource

	highPriorityClassName := f.BaseName + "-high-priority"
	highPriority := int32(999999999)

	ginkgo.Context(fmt.Sprintf(testContextFmt, expectedNodeCondition), func() {
		tempSetCurrentKubeletConfig(f, func(initialConfig *kubeletconfig.KubeletConfiguration) {
			pidsConsumed := int64(10000)
			summary := eventuallyGetSummary()
			availablePids := *(summary.Node.Rlimit.MaxPID) - *(summary.Node.Rlimit.NumOfRunningProcesses)
			initialConfig.EvictionHard = map[string]string{string(evictionapi.SignalPIDAvailable): fmt.Sprintf("%d", availablePids-pidsConsumed)}
			initialConfig.EvictionMinimumReclaim = map[string]string{}
		})
		ginkgo.BeforeEach(func() {
			_, err := f.ClientSet.SchedulingV1().PriorityClasses().Create(context.TODO(), &schedulingv1.PriorityClass{ObjectMeta: metav1.ObjectMeta{Name: highPriorityClassName}, Value: highPriority}, metav1.CreateOptions{})
			framework.ExpectEqual(err == nil || apierrors.IsAlreadyExists(err), true)
		})
		ginkgo.AfterEach(func() {
			err := f.ClientSet.SchedulingV1().PriorityClasses().Delete(context.TODO(), highPriorityClassName, metav1.DeleteOptions{})
			framework.ExpectNoError(err)
		})
		specs := []podEvictSpec{
			{
				evictionPriority: 2,
				pod:              pidConsumingPod("fork-bomb-container-with-low-priority", 12000),
			},
			{
				evictionPriority: 0,
				pod:              innocentPod(),
			},
			{
				evictionPriority: 1,
				pod:              pidConsumingPod("fork-bomb-container-with-high-priority", 12000),
			},
		}
		specs[1].pod.Spec.PriorityClassName = highPriorityClassName
		specs[2].pod.Spec.PriorityClassName = highPriorityClassName
		runEvictionTest(f, pressureTimeout, expectedNodeCondition, expectedStarvedResource, logPidMetrics, specs)
	})
})

// Struct used by runEvictionTest that specifies the pod, and when that pod should be evicted, relative to other pods
type podEvictSpec struct {
	// P0 should never be evicted, P1 shouldn't evict before P2, etc.
	// If two are ranked at P1, either is permitted to fail before the other.
	// The test ends when all pods other than p0 have been evicted
	evictionPriority int
	pod              *v1.Pod
}

// runEvictionTest sets up a testing environment given the provided pods, and checks a few things:
//		It ensures that the desired expectedNodeCondition is actually triggered.
//		It ensures that evictionPriority 0 pods are not evicted
//		It ensures that lower evictionPriority pods are always evicted before higher evictionPriority pods (2 evicted before 1, etc.)
//		It ensures that all pods with non-zero evictionPriority are eventually evicted.
// runEvictionTest then cleans up the testing environment by deleting provided pods, and ensures that expectedNodeCondition no longer exists
func runEvictionTest(f *framework.Framework, pressureTimeout time.Duration, expectedNodeCondition v1.NodeConditionType, expectedStarvedResource v1.ResourceName, logFunc func(), testSpecs []podEvictSpec) {
	// Place the remainder of the test within a context so that the kubelet config is set before and after the test.
	ginkgo.Context("", func() {
		ginkgo.BeforeEach(func() {
			// reduce memory usage in the allocatable cgroup to ensure we do not have MemoryPressure
			reduceAllocatableMemoryUsage()
			// Nodes do not immediately report local storage capacity
			// Sleep so that pods requesting local storage do not fail to schedule
			time.Sleep(30 * time.Second)
			ginkgo.By("setting up pods to be used by tests")
			pods := []*v1.Pod{}
			for _, spec := range testSpecs {
				pods = append(pods, spec.pod)
			}
			f.PodClient().CreateBatch(pods)
		})

		ginkgo.It("should eventually evict all of the correct pods", func() {
			ginkgo.By(fmt.Sprintf("Waiting for node to have NodeCondition: %s", expectedNodeCondition))
			gomega.Eventually(func() error {
				logFunc()
				if expectedNodeCondition == noPressure || hasNodeCondition(f, expectedNodeCondition) {
					return nil
				}
				return fmt.Errorf("NodeCondition: %s not encountered", expectedNodeCondition)
			}, pressureTimeout, evictionPollInterval).Should(gomega.BeNil())

			ginkgo.By("Waiting for evictions to occur")
			gomega.Eventually(func() error {
				if expectedNodeCondition != noPressure {
					if hasNodeCondition(f, expectedNodeCondition) {
						framework.Logf("Node has %s", expectedNodeCondition)
					} else {
						framework.Logf("Node does NOT have %s", expectedNodeCondition)
					}
				}
				logKubeletLatencyMetrics(kubeletmetrics.EvictionStatsAgeKey)
				logFunc()
				return verifyEvictionOrdering(f, testSpecs)
			}, pressureTimeout, evictionPollInterval).Should(gomega.BeNil())

			// We observe pressure from the API server.  The eviction manager observes pressure from the kubelet internal stats.
			// This means the eviction manager will observe pressure before we will, creating a delay between when the eviction manager
			// evicts a pod, and when we observe the pressure by querying the API server.  Add a delay here to account for this delay
			ginkgo.By("making sure pressure from test has surfaced before continuing")
			time.Sleep(pressureDelay)

			ginkgo.By(fmt.Sprintf("Waiting for NodeCondition: %s to no longer exist on the node", expectedNodeCondition))
			gomega.Eventually(func() error {
				logFunc()
				logKubeletLatencyMetrics(kubeletmetrics.EvictionStatsAgeKey)
				if expectedNodeCondition != noPressure && hasNodeCondition(f, expectedNodeCondition) {
					return fmt.Errorf("Conditions haven't returned to normal, node still has %s", expectedNodeCondition)
				}
				return nil
			}, pressureDisappearTimeout, evictionPollInterval).Should(gomega.BeNil())

			ginkgo.By("checking for stable, pressure-free condition without unexpected pod failures")
			gomega.Consistently(func() error {
				if expectedNodeCondition != noPressure && hasNodeCondition(f, expectedNodeCondition) {
					return fmt.Errorf("%s disappeared and then reappeared", expectedNodeCondition)
				}
				logFunc()
				logKubeletLatencyMetrics(kubeletmetrics.EvictionStatsAgeKey)
				return verifyEvictionOrdering(f, testSpecs)
			}, postTestConditionMonitoringPeriod, evictionPollInterval).Should(gomega.BeNil())

			ginkgo.By("checking for correctly formatted eviction events")
			verifyEvictionEvents(f, testSpecs, expectedStarvedResource)
		})

		ginkgo.AfterEach(func() {
			prePullImagesIfNeccecary := func() {
				if expectedNodeCondition == v1.NodeDiskPressure && framework.TestContext.PrepullImages {
					// The disk eviction test may cause the prepulled images to be evicted,
					// prepull those images again to ensure this test not affect following tests.
					PrePullAllImages()
				}
			}
			// Run prePull using a defer to make sure it is executed even when the assertions below fails
			defer prePullImagesIfNeccecary()

			ginkgo.By("deleting pods")
			for _, spec := range testSpecs {
				ginkgo.By(fmt.Sprintf("deleting pod: %s", spec.pod.Name))
				f.PodClient().DeleteSync(spec.pod.Name, metav1.DeleteOptions{}, 10*time.Minute)
			}

			// In case a test fails before verifying that NodeCondition no longer exist on the node,
			// we should wait for the NodeCondition to disappear
			ginkgo.By(fmt.Sprintf("making sure NodeCondition %s no longer exist on the node", expectedNodeCondition))
			gomega.Eventually(func() error {
				if expectedNodeCondition != noPressure && hasNodeCondition(f, expectedNodeCondition) {
					return fmt.Errorf("Conditions haven't returned to normal, node still has %s", expectedNodeCondition)
				}
				return nil
			}, pressureDisappearTimeout, evictionPollInterval).Should(gomega.BeNil())

			reduceAllocatableMemoryUsage()
			ginkgo.By("making sure we have all the required images for testing")
			prePullImagesIfNeccecary()

			ginkgo.By("making sure we can start a new pod after the test")
			podName := "test-admit-pod"
			f.PodClient().CreateSync(&v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: podName,
				},
				Spec: v1.PodSpec{
					RestartPolicy: v1.RestartPolicyNever,
					Containers: []v1.Container{
						{
							Image: imageutils.GetPauseImageName(),
							Name:  podName,
						},
					},
				},
			})

			if ginkgo.CurrentGinkgoTestDescription().Failed {
				if framework.TestContext.DumpLogsOnFailure {
					logPodEvents(f)
					logNodeEvents(f)
				}
			}
		})
	})
}

// verifyEvictionOrdering returns an error if all non-zero priority pods have not been evicted, nil otherwise
// This function panics (via Expect) if eviction ordering is violated, or if a priority-zero pod fails.
func verifyEvictionOrdering(f *framework.Framework, testSpecs []podEvictSpec) error {
	// Gather current information
	updatedPodList, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{})
	if err != nil {
		return err
	}
	updatedPods := updatedPodList.Items
	for _, p := range updatedPods {
		framework.Logf("fetching pod %s; phase= %v", p.Name, p.Status.Phase)
	}

	ginkgo.By("checking eviction ordering and ensuring important pods dont fail")
	done := true
	for _, priorityPodSpec := range testSpecs {
		var priorityPod v1.Pod
		for _, p := range updatedPods {
			if p.Name == priorityPodSpec.pod.Name {
				priorityPod = p
			}
		}
		gomega.Expect(priorityPod).NotTo(gomega.BeNil())
		framework.ExpectNotEqual(priorityPod.Status.Phase, v1.PodSucceeded,
			fmt.Sprintf("pod: %s succeeded unexpectedly", priorityPod.Name))

		// Check eviction ordering.
		// Note: it is alright for a priority 1 and priority 2 pod (for example) to fail in the same round,
		// but never alright for a priority 1 pod to fail while the priority 2 pod is still running
		for _, lowPriorityPodSpec := range testSpecs {
			var lowPriorityPod v1.Pod
			for _, p := range updatedPods {
				if p.Name == lowPriorityPodSpec.pod.Name {
					lowPriorityPod = p
				}
			}
			gomega.Expect(lowPriorityPod).NotTo(gomega.BeNil())
			if priorityPodSpec.evictionPriority < lowPriorityPodSpec.evictionPriority && lowPriorityPod.Status.Phase == v1.PodRunning {
				framework.ExpectNotEqual(priorityPod.Status.Phase, v1.PodFailed,
					fmt.Sprintf("priority %d pod: %s failed before priority %d pod: %s",
						priorityPodSpec.evictionPriority, priorityPodSpec.pod.Name, lowPriorityPodSpec.evictionPriority, lowPriorityPodSpec.pod.Name))
			}
		}

		if priorityPod.Status.Phase == v1.PodFailed {
			framework.ExpectEqual(priorityPod.Status.Reason, eviction.Reason, "pod %s failed; expected Status.Reason to be %s, but got %s",
				priorityPod.Name, eviction.Reason, priorityPod.Status.Reason)
		}

		// EvictionPriority 0 pods should not fail
		if priorityPodSpec.evictionPriority == 0 {
			framework.ExpectNotEqual(priorityPod.Status.Phase, v1.PodFailed,
				fmt.Sprintf("priority 0 pod: %s failed", priorityPod.Name))
		}

		// If a pod that is not evictionPriority 0 has not been evicted, we are not done
		if priorityPodSpec.evictionPriority != 0 && priorityPod.Status.Phase != v1.PodFailed {
			done = false
		}
	}
	if done {
		return nil
	}
	return fmt.Errorf("pods that should be evicted are still running")
}

func verifyEvictionEvents(f *framework.Framework, testSpecs []podEvictSpec, expectedStarvedResource v1.ResourceName) {
	for _, spec := range testSpecs {
		pod := spec.pod
		if spec.evictionPriority != 0 {
			selector := fields.Set{
				"involvedObject.kind":      "Pod",
				"involvedObject.name":      pod.Name,
				"involvedObject.namespace": f.Namespace.Name,
				"reason":                   eviction.Reason,
			}.AsSelector().String()
			podEvictEvents, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(context.TODO(), metav1.ListOptions{FieldSelector: selector})
			gomega.Expect(err).To(gomega.BeNil(), "Unexpected error getting events during eviction test: %v", err)
			framework.ExpectEqual(len(podEvictEvents.Items), 1, "Expected to find 1 eviction event for pod %s, got %d", pod.Name, len(podEvictEvents.Items))
			event := podEvictEvents.Items[0]

			if expectedStarvedResource != noStarvedResource {
				// Check the eviction.StarvedResourceKey
				starved, found := event.Annotations[eviction.StarvedResourceKey]
				framework.ExpectEqual(found, true, "Expected to find an annotation on the eviction event for pod %s containing the starved resource %s, but it was not found",
					pod.Name, expectedStarvedResource)
				starvedResource := v1.ResourceName(starved)
				framework.ExpectEqual(starvedResource, expectedStarvedResource, "Expected to the starved_resource annotation on pod %s to contain %s, but got %s instead",
					pod.Name, expectedStarvedResource, starvedResource)

				// We only check these keys for memory, because ephemeral storage evictions may be due to volume usage, in which case these values are not present
				if expectedStarvedResource == v1.ResourceMemory {
					// Check the eviction.OffendingContainersKey
					offendersString, found := event.Annotations[eviction.OffendingContainersKey]
					framework.ExpectEqual(found, true, "Expected to find an annotation on the eviction event for pod %s containing the offending containers, but it was not found",
						pod.Name)
					offendingContainers := strings.Split(offendersString, ",")
					framework.ExpectEqual(len(offendingContainers), 1, "Expected to find the offending container's usage in the %s annotation, but no container was found",
						eviction.OffendingContainersKey)
					framework.ExpectEqual(offendingContainers[0], pod.Spec.Containers[0].Name, "Expected to find the offending container: %s's usage in the %s annotation, but found %s instead",
						pod.Spec.Containers[0].Name, eviction.OffendingContainersKey, offendingContainers[0])

					// Check the eviction.OffendingContainersUsageKey
					offendingUsageString, found := event.Annotations[eviction.OffendingContainersUsageKey]
					framework.ExpectEqual(found, true, "Expected to find an annotation on the eviction event for pod %s containing the offending containers' usage, but it was not found",
						pod.Name)
					offendingContainersUsage := strings.Split(offendingUsageString, ",")
					framework.ExpectEqual(len(offendingContainersUsage), 1, "Expected to find the offending container's usage in the %s annotation, but found %+v",
						eviction.OffendingContainersUsageKey, offendingContainersUsage)
					usageQuantity, err := resource.ParseQuantity(offendingContainersUsage[0])
					gomega.Expect(err).To(gomega.BeNil(), "Expected to be able to parse pod %s's %s annotation as a quantity, but got err: %v", pod.Name, eviction.OffendingContainersUsageKey, err)
					request := pod.Spec.Containers[0].Resources.Requests[starvedResource]
					framework.ExpectEqual(usageQuantity.Cmp(request), 1, "Expected usage of offending container: %s in pod %s to exceed its request %s",
						usageQuantity.String(), pod.Name, request.String())
				}
			}
		}
	}
}

// Returns TRUE if the node has the node condition, FALSE otherwise
func hasNodeCondition(f *framework.Framework, expectedNodeCondition v1.NodeConditionType) bool {
	localNodeStatus := getLocalNode(f).Status
	_, actualNodeCondition := testutils.GetNodeCondition(&localNodeStatus, expectedNodeCondition)
	gomega.Expect(actualNodeCondition).NotTo(gomega.BeNil())
	return actualNodeCondition.Status == v1.ConditionTrue
}

func logInodeMetrics() {
	summary, err := getNodeSummary()
	if err != nil {
		framework.Logf("Error getting summary: %v", err)
		return
	}
	if summary.Node.Runtime != nil && summary.Node.Runtime.ImageFs != nil && summary.Node.Runtime.ImageFs.Inodes != nil && summary.Node.Runtime.ImageFs.InodesFree != nil {
		framework.Logf("imageFsInfo.Inodes: %d, imageFsInfo.InodesFree: %d", *summary.Node.Runtime.ImageFs.Inodes, *summary.Node.Runtime.ImageFs.InodesFree)
	}
	if summary.Node.Fs != nil && summary.Node.Fs.Inodes != nil && summary.Node.Fs.InodesFree != nil {
		framework.Logf("rootFsInfo.Inodes: %d, rootFsInfo.InodesFree: %d", *summary.Node.Fs.Inodes, *summary.Node.Fs.InodesFree)
	}
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Rootfs != nil && container.Rootfs.InodesUsed != nil {
				framework.Logf("--- summary Container: %s inodeUsage: %d", container.Name, *container.Rootfs.InodesUsed)
			}
		}
		for _, volume := range pod.VolumeStats {
			if volume.FsStats.InodesUsed != nil {
				framework.Logf("--- summary Volume: %s inodeUsage: %d", volume.Name, *volume.FsStats.InodesUsed)
			}
		}
	}
}

func logDiskMetrics() {
	summary, err := getNodeSummary()
	if err != nil {
		framework.Logf("Error getting summary: %v", err)
		return
	}
	if summary.Node.Runtime != nil && summary.Node.Runtime.ImageFs != nil && summary.Node.Runtime.ImageFs.CapacityBytes != nil && summary.Node.Runtime.ImageFs.AvailableBytes != nil {
		framework.Logf("imageFsInfo.CapacityBytes: %d, imageFsInfo.AvailableBytes: %d", *summary.Node.Runtime.ImageFs.CapacityBytes, *summary.Node.Runtime.ImageFs.AvailableBytes)
	}
	if summary.Node.Fs != nil && summary.Node.Fs.CapacityBytes != nil && summary.Node.Fs.AvailableBytes != nil {
		framework.Logf("rootFsInfo.CapacityBytes: %d, rootFsInfo.AvailableBytes: %d", *summary.Node.Fs.CapacityBytes, *summary.Node.Fs.AvailableBytes)
	}
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Rootfs != nil && container.Rootfs.UsedBytes != nil {
				framework.Logf("--- summary Container: %s UsedBytes: %d", container.Name, *container.Rootfs.UsedBytes)
			}
		}
		for _, volume := range pod.VolumeStats {
			if volume.FsStats.InodesUsed != nil {
				framework.Logf("--- summary Volume: %s UsedBytes: %d", volume.Name, *volume.FsStats.UsedBytes)
			}
		}
	}
}

func logMemoryMetrics() {
	summary, err := getNodeSummary()
	if err != nil {
		framework.Logf("Error getting summary: %v", err)
		return
	}
	if summary.Node.Memory != nil && summary.Node.Memory.WorkingSetBytes != nil && summary.Node.Memory.AvailableBytes != nil {
		framework.Logf("Node.Memory.WorkingSetBytes: %d, Node.Memory.AvailableBytes: %d", *summary.Node.Memory.WorkingSetBytes, *summary.Node.Memory.AvailableBytes)
	}
	for _, sysContainer := range summary.Node.SystemContainers {
		if sysContainer.Name == kubeletstatsv1alpha1.SystemContainerPods && sysContainer.Memory != nil && sysContainer.Memory.WorkingSetBytes != nil && sysContainer.Memory.AvailableBytes != nil {
			framework.Logf("Allocatable.Memory.WorkingSetBytes: %d, Allocatable.Memory.AvailableBytes: %d", *sysContainer.Memory.WorkingSetBytes, *sysContainer.Memory.AvailableBytes)
		}
	}
	for _, pod := range summary.Pods {
		framework.Logf("Pod: %s", pod.PodRef.Name)
		for _, container := range pod.Containers {
			if container.Memory != nil && container.Memory.WorkingSetBytes != nil {
				framework.Logf("--- summary Container: %s WorkingSetBytes: %d", container.Name, *container.Memory.WorkingSetBytes)
			}
		}
	}
}

func logPidMetrics() {
	summary, err := getNodeSummary()
	if err != nil {
		framework.Logf("Error getting summary: %v", err)
		return
	}
	if summary.Node.Rlimit != nil && summary.Node.Rlimit.MaxPID != nil && summary.Node.Rlimit.NumOfRunningProcesses != nil {
		framework.Logf("Node.Rlimit.MaxPID: %d, Node.Rlimit.RunningProcesses: %d", *summary.Node.Rlimit.MaxPID, *summary.Node.Rlimit.NumOfRunningProcesses)
	}
}

func eventuallyGetSummary() (s *kubeletstatsv1alpha1.Summary) {
	gomega.Eventually(func() error {
		summary, err := getNodeSummary()
		if err != nil {
			return err
		}
		if summary == nil || summary.Node.Fs == nil || summary.Node.Fs.InodesFree == nil || summary.Node.Fs.AvailableBytes == nil {
			return fmt.Errorf("some part of data is nil")
		}
		s = summary
		return nil
	}, time.Minute, evictionPollInterval).Should(gomega.BeNil())
	return
}

// returns a pod that does not use any resources
func innocentPod() *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "innocent-pod"},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image: busyboxImage,
					Name:  "innocent-container",
					Command: []string{
						"sh",
						"-c",
						"while true; do sleep 5; done",
					},
				},
			},
		},
	}
}

const (
	volumeMountPath = "/test-mnt"
	volumeName      = "test-volume"
)

func inodeConsumingPod(name string, numFiles int, volumeSource *v1.VolumeSource) *v1.Pod {
	path := ""
	if volumeSource != nil {
		path = volumeMountPath
	}
	// Each iteration creates an empty file
	return podWithCommand(volumeSource, v1.ResourceRequirements{}, numFiles, name, fmt.Sprintf("touch %s${i}.txt; sleep 0.001;", filepath.Join(path, "file")))
}

func diskConsumingPod(name string, diskConsumedMB int, volumeSource *v1.VolumeSource, resources v1.ResourceRequirements) *v1.Pod {
	path := ""
	if volumeSource != nil {
		path = volumeMountPath
	}
	// Each iteration writes 1 Mb, so do diskConsumedMB iterations.
	return podWithCommand(volumeSource, resources, diskConsumedMB, name, fmt.Sprintf("dd if=/dev/urandom of=%s${i} bs=1048576 count=1 2>/dev/null; sleep .1;", filepath.Join(path, "file")))
}

func pidConsumingPod(name string, numProcesses int) *v1.Pod {
	// Each iteration forks once, but creates two processes
	return podWithCommand(nil, v1.ResourceRequirements{}, numProcesses/2, name, "(while true; do sleep 5; done)&")
}

// podWithCommand returns a pod with the provided volumeSource and resourceRequirements.
func podWithCommand(volumeSource *v1.VolumeSource, resources v1.ResourceRequirements, iterations int, name, command string) *v1.Pod {
	volumeMounts := []v1.VolumeMount{}
	volumes := []v1.Volume{}
	if volumeSource != nil {
		volumeMounts = []v1.VolumeMount{{MountPath: volumeMountPath, Name: volumeName}}
		volumes = []v1.Volume{{Name: volumeName, VolumeSource: *volumeSource}}
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("%s-pod", name)},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Image: busyboxImage,
					Name:  fmt.Sprintf("%s-container", name),
					Command: []string{
						"sh",
						"-c",
						fmt.Sprintf("i=0; while [ $i -lt %d ]; do %s i=$(($i+1)); done; while true; do sleep 5; done", iterations, command),
					},
					Resources:    resources,
					VolumeMounts: volumeMounts,
				},
			},
			Volumes: volumes,
		},
	}
}

func getMemhogPod(podName string, ctnName string, res v1.ResourceRequirements) *v1.Pod {
	env := []v1.EnvVar{
		{
			Name: "MEMORY_LIMIT",
			ValueFrom: &v1.EnvVarSource{
				ResourceFieldRef: &v1.ResourceFieldSelector{
					Resource: "limits.memory",
				},
			},
		},
	}

	// If there is a limit specified, pass 80% of it for -mem-total, otherwise use the downward API
	// to pass limits.memory, which will be the total memory available.
	// This helps prevent a guaranteed pod from triggering an OOM kill due to it's low memory limit,
	// which will cause the test to fail inappropriately.
	var memLimit string
	if limit, ok := res.Limits[v1.ResourceMemory]; ok {
		memLimit = strconv.Itoa(int(
			float64(limit.Value()) * 0.8))
	} else {
		memLimit = "$(MEMORY_LIMIT)"
	}

	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:            ctnName,
					Image:           "k8s.gcr.io/stress:v1",
					ImagePullPolicy: "Always",
					Env:             env,
					// 60 min timeout * 60s / tick per 10s = 360 ticks before timeout => ~11.11Mi/tick
					// to fill ~4Gi of memory, so initial ballpark 12Mi/tick.
					// We might see flakes due to timeout if the total memory on the nodes increases.
					Args:      []string{"-mem-alloc-size", "12Mi", "-mem-alloc-sleep", "10s", "-mem-total", memLimit},
					Resources: res,
				},
			},
		},
	}
}

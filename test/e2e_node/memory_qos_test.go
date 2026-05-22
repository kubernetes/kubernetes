/*
Copyright The Kubernetes Authors.

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
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	cgroupRoot         = "/sys/fs/cgroup"
	cgroupMemoryMin    = "memory.min"
	cgroupMemoryLow    = "memory.low"
	cgroupMemoryHigh   = "memory.high"
	cgroupMemoryMax    = "memory.max"
	cgroupMemoryEvents = "memory.events"
)

// memqosReadCgroupFile reads a cgroup file and returns its content as a string.
func memqosReadCgroupFile(cgroupPath, fileName string) (string, error) {
	data, err := os.ReadFile(filepath.Join(cgroupPath, fileName))
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

// memqosReadCgroupInt64 reads a cgroup file and returns its content as int64.
// Returns -1 for "max".
func memqosReadCgroupInt64(cgroupPath, fileName string) (int64, error) {
	val, err := memqosReadCgroupFile(cgroupPath, fileName)
	if err != nil {
		return 0, err
	}
	if val == "max" {
		return -1, nil
	}
	return strconv.ParseInt(val, 10, 64)
}

// memqosReadMemoryEvents reads memory.events and returns a map of counter names to values.
func memqosReadMemoryEvents(cgroupPath string) (map[string]int64, error) {
	data, err := memqosReadCgroupFile(cgroupPath, cgroupMemoryEvents)
	if err != nil {
		return nil, err
	}
	events := make(map[string]int64)
	for line := range strings.SplitSeq(data, "\n") {
		parts := strings.Fields(line)
		if len(parts) == 2 {
			val, err := strconv.ParseInt(parts[1], 10, 64)
			if err == nil {
				events[parts[0]] = val
			}
		}
	}
	return events, nil
}

// memqosGetPodCgroupPath returns the cgroup path for a pod.
func memqosGetPodCgroupPath(pod *v1.Pod, cgroupDriver string) string {
	uid := string(pod.UID)
	qosClass := pod.Status.QOSClass

	if cgroupDriver == "systemd" {
		uid = strings.ReplaceAll(uid, "-", "_")
		switch qosClass {
		case v1.PodQOSGuaranteed:
			return filepath.Join(cgroupRoot, "kubepods.slice",
				fmt.Sprintf("kubepods-pod%s.slice", uid))
		case v1.PodQOSBurstable:
			return filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice",
				fmt.Sprintf("kubepods-burstable-pod%s.slice", uid))
		case v1.PodQOSBestEffort:
			return filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-besteffort.slice",
				fmt.Sprintf("kubepods-besteffort-pod%s.slice", uid))
		}
	}

	// cgroupfs driver
	switch qosClass {
	case v1.PodQOSGuaranteed:
		return filepath.Join(cgroupRoot, "kubepods", fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBurstable:
		return filepath.Join(cgroupRoot, "kubepods", "burstable", fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBestEffort:
		return filepath.Join(cgroupRoot, "kubepods", "besteffort", fmt.Sprintf("pod%s", uid))
	}
	return ""
}

// memqosGetContainerCgroupPath returns the cgroup path for a container within a pod.
func memqosGetContainerCgroupPath(podCgroupPath, containerID, cgroupDriver string) string {
	// containerID format: "containerd://abc123" or "cri-o://abc123"
	parts := strings.SplitN(containerID, "://", 2)
	if len(parts) != 2 {
		return ""
	}
	runtime := parts[0]
	id := parts[1]

	if cgroupDriver == "systemd" {
		switch runtime {
		case "containerd":
			return filepath.Join(podCgroupPath, fmt.Sprintf("cri-containerd-%s.scope", id))
		case "cri-o":
			return filepath.Join(podCgroupPath, fmt.Sprintf("crio-%s.scope", id))
		}
	}

	// cgroupfs driver
	return filepath.Join(podCgroupPath, id)
}

// memqosExpectedMemoryHigh calculates the expected memory.high value.
func memqosExpectedMemoryHigh(requestBytes, limitBytes int64, throttlingFactor float64) int64 {
	pageSize := int64(os.Getpagesize())
	raw := float64(requestBytes) + throttlingFactor*float64(limitBytes-requestBytes)
	return int64(math.Floor(raw/float64(pageSize))) * pageSize
}

// memqosMakePod creates a test pod with specified resources.
func memqosMakePod(name, namespace string, requests, limits v1.ResourceList) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "test",
					Image:   "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
					Command: []string{"sleep", "infinity"},
					Resources: v1.ResourceRequirements{
						Requests: requests,
						Limits:   limits,
					},
				},
			},
		},
	}
}

var _ = SIGDescribe("MemoryQoS", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("memory-qos")
	addAfterEachForCleaningUpPods(f)
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		oldCfg       *kubeletconfig.KubeletConfiguration
		cgroupDriver string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("MemoryQoS requires cgroups v2")
		}

		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)
		cgroupDriver = oldCfg.CgroupDriver
	})

	// configureMemoryQoS restarts kubelet with MemoryQoS enabled and specified settings.
	configureMemoryQoS := func(ctx context.Context, throttlingFactor float64) {
		newCfg := oldCfg.DeepCopy()
		if newCfg.FeatureGates == nil {
			newCfg.FeatureGates = make(map[string]bool)
		}
		newCfg.FeatureGates["MemoryQoS"] = true
		newCfg.MemoryThrottlingFactor = &throttlingFactor
		newCfg.CgroupsPerQOS = true
		newCfg.EnforceNodeAllocatable = []string{"pods"}

		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
	}

	// configureMemoryQoSWithPolicy restarts kubelet with MemoryQoS and a specific memoryReservationPolicy.
	configureMemoryQoSWithPolicy := func(ctx context.Context, throttlingFactor float64, policy kubeletconfig.MemoryReservationPolicy) {
		newCfg := oldCfg.DeepCopy()
		if newCfg.FeatureGates == nil {
			newCfg.FeatureGates = make(map[string]bool)
		}
		newCfg.FeatureGates["MemoryQoS"] = true
		newCfg.MemoryThrottlingFactor = &throttlingFactor
		newCfg.MemoryReservationPolicy = policy
		newCfg.CgroupsPerQOS = true
		newCfg.EnforceNodeAllocatable = []string{"pods"}

		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
	}

	restoreConfig := func(ctx context.Context) {
		if oldCfg != nil {
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))
			restartKubelet(ctx, true)
			waitForKubeletToStart(ctx, f)
		}
	}

	f.Describe("memory protection [memoryReservationPolicy=TieredReservation]", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should set memory.low = requests.memory for Burstable pod containers", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-burstable", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)

			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			gomega.Expect(podCgroupPath).NotTo(gomega.BeEmpty(), "pod cgroup path should not be empty")

			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err, "reading pod memory.low")

			expectedPodMin := requestsMem.Value()
			framework.Logf("Pod memory.low: got=%d, expected=%d", podMemMin, expectedPodMin)
			gomega.Expect(podMemMin).To(gomega.Equal(expectedPodMin),
				"pod memory.low should equal sum of container requests")

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)
				gomega.Expect(containerCgroupPath).NotTo(gomega.BeEmpty())

				containerMemMin, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
				framework.ExpectNoError(err, "reading container memory.low")

				framework.Logf("Container %s memory.low: got=%d, expected=%d",
					cs.Name, containerMemMin, requestsMem.Value())
				gomega.Expect(containerMemMin).To(gomega.Equal(requestsMem.Value()),
					"container memory.low should equal container request")
			}
		})

		ginkgo.It("should set memory.min for Guaranteed pod containers", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			mem := resource.MustParse("256Mi")
			cpu := resource.MustParse("100m")
			resources := v1.ResourceList{
				v1.ResourceMemory: mem,
				v1.ResourceCPU:    cpu,
			}

			pod := memqosMakePod("memqos-guaranteed", f.Namespace.Name, resources, resources)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryMin)
			framework.ExpectNoError(err)

			framework.Logf("Guaranteed pod memory.min: got=%d, expected=%d", podMemMin, mem.Value())
			gomega.Expect(podMemMin).To(gomega.Equal(mem.Value()),
				"Guaranteed pod memory.min should equal requests")
		})

		ginkgo.It("should NOT set memory.min for BestEffort pod", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "memqos-besteffort",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test",
							Image:   "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							Command: []string{"sleep", "infinity"},
						},
					},
				},
			}

			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryMin)
			framework.ExpectNoError(err)

			framework.Logf("BestEffort pod memory.min: got=%d, expected=0", podMemMin)
			gomega.Expect(podMemMin).To(gomega.Equal(int64(0)),
				"BestEffort pod memory.min should be 0")
		})

		ginkgo.It("should set pod-level memory.low = sum(container requests) for multi-container pod", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			req1 := resource.MustParse("128Mi")
			req2 := resource.MustParse("256Mi")
			limit := resource.MustParse("512Mi")

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "memqos-multi-container",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "container-1",
							Image:   "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							Command: []string{"sleep", "infinity"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: req1},
								Limits:   v1.ResourceList{v1.ResourceMemory: limit},
							},
						},
						{
							Name:    "container-2",
							Image:   "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							Command: []string{"sleep", "infinity"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: req2},
								Limits:   v1.ResourceList{v1.ResourceMemory: limit},
							},
						},
					},
				},
			}

			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)

			expectedPodMin := req1.Value() + req2.Value()
			framework.Logf("Multi-container pod memory.low: got=%d, expected=%d", podMemMin, expectedPodMin)
			gomega.Expect(podMemMin).To(gomega.Equal(expectedPodMin),
				"pod memory.low should equal sum of all container requests")
		})

		ginkgo.It("should propagate memory protection through QoS cgroup hierarchy", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			requestsMem := resource.MustParse("128Mi")
			limitsMem := resource.MustParse("256Mi")

			pod := memqosMakePod("memqos-hierarchy", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			e2epod.NewPodClient(f).CreateSync(ctx, pod)

			var burstableCgroupPath string
			var kubepodsCgroupPath string
			if cgroupDriver == "systemd" {
				kubepodsCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice")
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice")
			} else {
				kubepodsCgroupPath = filepath.Join(cgroupRoot, "kubepods")
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods", "burstable")
			}

			kubepodsMemMin, err := memqosReadCgroupInt64(kubepodsCgroupPath, cgroupMemoryMin)
			framework.ExpectNoError(err, "reading kubepods memory.min")
			framework.Logf("kubepods memory.min: %d", kubepodsMemMin)
			gomega.Expect(kubepodsMemMin).To(gomega.BeNumerically(">", 0),
				"kubepods cgroup must have memory.min > 0 for hierarchy protection to work")

			burstableMemMin, err := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err, "reading burstable QoS memory.low")
			framework.Logf("burstable QoS memory.low: %d", burstableMemMin)
			gomega.Expect(burstableMemMin).To(gomega.BeNumerically(">", 0),
				"burstable QoS cgroup must have memory.low > 0")
		})
	})

	f.Describe("memory.high throttling", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should set memory.high for Burstable pod using formula", func(ctx context.Context) {
			throttlingFactor := 0.9
			configureMemoryQoS(ctx, throttlingFactor)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-high-burstable", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
				framework.ExpectNoError(err, "reading container memory.high")

				expected := memqosExpectedMemoryHigh(requestsMem.Value(), limitsMem.Value(), throttlingFactor)
				framework.Logf("Container %s memory.high: got=%d, expected=%d", cs.Name, containerMemHigh, expected)
				gomega.Expect(containerMemHigh).To(gomega.Equal(expected),
					"memory.high should match formula: floor[(req + factor * (limit - req)) / pageSize] * pageSize")
			}
		})

		ginkgo.It("should NOT set memory.high for Guaranteed pod", func(ctx context.Context) {
			configureMemoryQoS(ctx, 0.9)

			mem := resource.MustParse("256Mi")
			cpu := resource.MustParse("100m")
			resources := v1.ResourceList{
				v1.ResourceMemory: mem,
				v1.ResourceCPU:    cpu,
			}

			pod := memqosMakePod("memqos-high-guaranteed", f.Namespace.Name, resources, resources)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
				framework.ExpectNoError(err, "reading container memory.high")

				framework.Logf("Guaranteed container %s memory.high: got=%d (expect max/-1)", cs.Name, containerMemHigh)
				gomega.Expect(containerMemHigh).To(gomega.Equal(int64(-1)),
					"Guaranteed pod should NOT have memory.high set (should be max)")
			}
		})

		ginkgo.It("should NOT set memory.high on pod-level cgroup", func(ctx context.Context) {
			configureMemoryQoS(ctx, 0.9)

			pod := memqosMakePod("memqos-high-pod-level", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			podMemHigh, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryHigh)
			framework.ExpectNoError(err, "reading pod memory.high")

			framework.Logf("Pod-level memory.high: got=%d (expect max/-1)", podMemHigh)
			gomega.Expect(podMemHigh).To(gomega.Equal(int64(-1)),
				"memory.high should NOT be set at pod level, only container level")
		})

		ginkgo.It("should set memory.high = limit when memoryThrottlingFactor=1.0", func(ctx context.Context) {
			configureMemoryQoS(ctx, 1.0)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-factor-1", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
				framework.ExpectNoError(err)

				expected := memqosExpectedMemoryHigh(requestsMem.Value(), limitsMem.Value(), 1.0)
				framework.Logf("Container %s memory.high with factor=1.0: got=%d, expected=%d (limit=%d)",
					cs.Name, containerMemHigh, expected, limitsMem.Value())
				// With factor=1.0: high = floor[(req + 1.0*(limit-req))/pageSize]*pageSize = limit (page-aligned)
				gomega.Expect(containerMemHigh).To(gomega.Equal(expected),
					"memory.high should equal limit when factor=1.0")
			}
		})

		ginkgo.It("should set memory.high using node allocatable when no limit is set", func(ctx context.Context) {
			throttlingFactor := 0.9
			configureMemoryQoS(ctx, throttlingFactor)

			requestsMem := resource.MustParse("128Mi")

			pod := memqosMakePod("memqos-no-limit", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{}, // no limits
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
				framework.ExpectNoError(err)

				framework.Logf("Container %s memory.high (no limit): got=%d", cs.Name, containerMemHigh)
				// memory.high should be set using node allocatable instead of limit
				gomega.Expect(containerMemHigh).To(gomega.BeNumerically(">", requestsMem.Value()),
					"memory.high without limit should use node allocatable and be > requests")
				gomega.Expect(containerMemHigh).NotTo(gomega.Equal(int64(-1)),
					"memory.high should not be max when MemoryQoS is enabled for Burstable")
			}
		})
	})

	f.Describe("memory.events observability", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should have memory.events file with expected counters", func(ctx context.Context) {
			configureMemoryQoS(ctx, 0.9)

			pod := memqosMakePod("memqos-events", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				events, err := memqosReadMemoryEvents(containerCgroupPath)
				framework.ExpectNoError(err, "reading memory.events")

				// Verify expected counters exist
				expectedCounters := []string{"low", "high", "max", "oom", "oom_kill"}
				for _, counter := range expectedCounters {
					_, exists := events[counter]
					framework.Logf("Container %s memory.events[%s] = %d, exists=%v",
						cs.Name, counter, events[counter], exists)
					gomega.Expect(exists).To(gomega.BeTrueBecause("memory.events should contain %q counter", counter))
				}

				// Initially all counters should be 0
				gomega.Expect(events["high"]).To(gomega.Equal(int64(0)),
					"high counter should be 0 initially")
			}
		})
	})

	f.Describe("feature rollback", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should reset memory protection to 0 when MemoryQoS is disabled", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			pod := memqosMakePod("memqos-rollback", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			memMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			gomega.Expect(memMin).To(gomega.BeNumerically(">", 0),
				"memory.low should be set when MemoryQoS is enabled")

			ginkgo.By("Disabling MemoryQoS feature gate")
			newCfg := oldCfg.DeepCopy()
			if newCfg.FeatureGates == nil {
				newCfg.FeatureGates = make(map[string]bool)
			}
			newCfg.FeatureGates["MemoryQoS"] = false
			newCfg.MemoryReservationPolicy = kubeletconfig.NoneMemoryReservationPolicy
			updateKubeletConfig(ctx, f, newCfg, true)

			// Stale QoS-class cgroup memory.low is cleared at kubelet startup.
			var burstableCgroupPath string
			if cgroupDriver == "systemd" {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice")
			} else {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods", "burstable")
			}
			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(gomega.Equal(int64(0)),
				"burstable QoS memory.low should reset to 0 when MemoryQoS is disabled")

			// Pod and container memory.low values persist but have no effect
			// since cgroup v2 memory protection is hierarchical (parent=0 wins).
		})

		ginkgo.It("should not clobber other cgroup values when clearing stale memory protection at startup", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			e2epod.NewPodClient(f).CreateSync(ctx, memqosMakePod("memqos-clobber-check", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			))

			var burstableCgroupPath string
			if cgroupDriver == "systemd" {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice")
			} else {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods", "burstable")
			}

			ginkgo.By("Verifying memory.low is non-zero while MemoryQoS is enabled")
			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(
				gomega.BeNumerically(">", int64(0)),
				"memory.low should be non-zero when MemoryQoS is enabled with burstable pods")

			ginkgo.By("Recording baseline cgroup values before rollback")
			cgroupFiles := []string{"cpu.max", "memory.max"}
			baselines := make(map[string]string)
			for _, cgFile := range cgroupFiles {
				filePath := filepath.Join(burstableCgroupPath, cgFile)
				if _, statErr := os.Stat(filePath); os.IsNotExist(statErr) {
					continue
				}
				val, err := memqosReadCgroupFile(burstableCgroupPath, cgFile)
				framework.ExpectNoError(err, "reading baseline %s", cgFile)
				baselines[cgFile] = val
				framework.Logf("baseline %s: %s", cgFile, val)
			}

			ginkgo.By("Disabling MemoryQoS and restarting kubelet")
			newCfg := oldCfg.DeepCopy()
			if newCfg.FeatureGates == nil {
				newCfg.FeatureGates = make(map[string]bool)
			}
			newCfg.FeatureGates["MemoryQoS"] = false
			newCfg.MemoryReservationPolicy = kubeletconfig.NoneMemoryReservationPolicy
			updateKubeletConfig(ctx, f, newCfg, true)

			ginkgo.By("Verifying memory.low was cleared to 0 at startup")
			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(gomega.Equal(int64(0)),
				"memory.low should be cleared at startup when MemoryQoS is disabled")

			ginkgo.By("Verifying other cgroup values were not clobbered by the rollback")
			for cgFile, baseline := range baselines {
				post, err := memqosReadCgroupFile(burstableCgroupPath, cgFile)
				framework.ExpectNoError(err, "reading post-rollback %s", cgFile)
				framework.Logf("post-rollback %s: got=%s, baseline=%s", cgFile, post, baseline)
				gomega.Expect(post).To(gomega.Equal(baseline),
					fmt.Sprintf("%s changed after rollback", cgFile))
			}
		})
	})

	f.Describe("memory.high formula verification across request/limit ratios", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should correctly compute memory.high for various request/limit combinations", func(ctx context.Context) {
			throttlingFactor := 0.9
			configureMemoryQoS(ctx, throttlingFactor)

			testCases := []struct {
				name     string
				requests string
				limits   string
			}{
				{"low-request", "64Mi", "512Mi"},
				{"half-request", "256Mi", "512Mi"},
				{"high-request", "480Mi", "512Mi"},
				{"equal-req-limit", "512Mi", "512Mi"}, // memory req==limit (still Burstable due to CPU mismatch)
			}

			for _, tc := range testCases {
				ginkgo.By(fmt.Sprintf("Testing %s: requests=%s limits=%s", tc.name, tc.requests, tc.limits))

				reqMem := resource.MustParse(tc.requests)
				limMem := resource.MustParse(tc.limits)
				reqCPU := resource.MustParse("50m")
				limCPU := resource.MustParse("100m")

				pod := memqosMakePod(fmt.Sprintf("memqos-formula-%s", tc.name), f.Namespace.Name,
					v1.ResourceList{v1.ResourceMemory: reqMem, v1.ResourceCPU: reqCPU},
					v1.ResourceList{v1.ResourceMemory: limMem, v1.ResourceCPU: limCPU},
				)
				pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

				podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

				for _, cs := range pod.Status.ContainerStatuses {
					containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

					containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
					framework.ExpectNoError(err)

					containerMemMin, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
					framework.ExpectNoError(err)

					if reqMem.Value() == limMem.Value() {
						// Guaranteed pod - memory.high should be max
						framework.Logf("[%s] Guaranteed: memory.high=%d (expect max), memory.low=%d",
							tc.name, containerMemHigh, containerMemMin)
						gomega.Expect(containerMemHigh).To(gomega.Equal(int64(-1)))
					} else {
						expected := memqosExpectedMemoryHigh(reqMem.Value(), limMem.Value(), throttlingFactor)
						framework.Logf("[%s] Burstable: memory.high=%d (expected=%d), memory.low=%d",
							tc.name, containerMemHigh, expected, containerMemMin)
						gomega.Expect(containerMemHigh).To(gomega.Equal(expected))
					}
				}

				// Cleanup
				e2epod.NewPodClient(f).DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, 2*time.Minute)
			}
		})
	})

	f.Describe("memoryReservationPolicy", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should NOT set memory protection when policy is None", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.NoneMemoryReservationPolicy)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-policy-none", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			framework.Logf("Policy=None: pod memory.low=%d", podMemMin)
			gomega.Expect(podMemMin).To(gomega.Equal(int64(0)))

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)
				containerMemMin, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
				framework.ExpectNoError(err)
				framework.Logf("Policy=None: container %s memory.low=%d", cs.Name, containerMemMin)
				gomega.Expect(containerMemMin).To(gomega.Equal(int64(0)))
			}
		})

		ginkgo.It("should set memory.high independent of memoryReservationPolicy", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.NoneMemoryReservationPolicy)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-policy-none-high", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)
				containerMemHigh, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryHigh)
				framework.ExpectNoError(err)

				expected := memqosExpectedMemoryHigh(requestsMem.Value(), limitsMem.Value(), 0.9)
				framework.Logf("Policy=None: container %s memory.high=%d, expected=%d",
					cs.Name, containerMemHigh, expected)
				gomega.Expect(containerMemHigh).To(gomega.Equal(expected))
			}
		})

		ginkgo.It("should set memory protection when policy is TieredReservation", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			requestsMem := resource.MustParse("256Mi")
			limitsMem := resource.MustParse("512Mi")

			pod := memqosMakePod("memqos-policy-hard", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: limitsMem},
			)
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			podMemMin, err := memqosReadCgroupInt64(podCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			framework.Logf("Policy=TieredReservation: pod memory.low=%d, expected=%d", podMemMin, requestsMem.Value())
			gomega.Expect(podMemMin).To(gomega.Equal(requestsMem.Value()))
		})

		ginkgo.It("should clear memory protection at QoS level when switching from TieredReservation to None", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			pod := memqosMakePod("memqos-policy-rollback", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("128Mi")},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")},
			)
			e2epod.NewPodClient(f).CreateSync(ctx, pod)

			var burstableCgroupPath string
			if cgroupDriver == "systemd" {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice")
			} else {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods", "burstable")
			}

			burstableMemMin, err := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			framework.Logf("Before rollback: burstable QoS memory.low=%d", burstableMemMin)
			gomega.Expect(burstableMemMin).To(gomega.BeNumerically(">", 0))

			ginkgo.By("Switching memoryReservationPolicy to None")
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.NoneMemoryReservationPolicy)

			// Wait for periodic QoS cgroup update (periodicQOSCgroupUpdateInterval = 1 minute)
			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(gomega.Equal(int64(0)),
				"burstable QoS memory.low should be cleared after switching to None")
		})
	})

	f.Describe("memory.high throttling behavior", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should increment memory.events high counter when usage exceeds memory.high", func(ctx context.Context) {
			configureMemoryQoS(ctx, 0.9)

			requestsMem := resource.MustParse("64Mi")
			limitsMem := resource.MustParse("256Mi")

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "memqos-throttle-test",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "mem-eater",
							Image: "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							// memory.high = 64 + 0.9*(256-64) = 236.8Mi, memory.max = 256Mi
							// dd allocates 240Mi which exceeds memory.high but leaves ~16Mi headroom to memory.max.
							Command: []string{"sh", "-c", "while true; do dd if=/dev/zero of=/dev/null bs=240M 2>/dev/null; done"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: requestsMem},
								Limits:   v1.ResourceList{v1.ResourceMemory: limitsMem},
							},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			pod = e2epod.NewPodClient(f).Create(ctx, pod)

			gomega.Eventually(ctx, func(ctx context.Context) v1.PodPhase {
				p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return v1.PodPending
				}
				return p.Status.Phase
			}).WithTimeout(60 * time.Second).WithPolling(2 * time.Second).Should(gomega.Equal(v1.PodRunning))

			pod, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)

				gomega.Eventually(ctx, func() int64 {
					events, err := memqosReadMemoryEvents(containerCgroupPath)
					if err != nil {
						return 0
					}
					return events["high"]
				}).WithTimeout(30*time.Second).WithPolling(2*time.Second).Should(gomega.BeNumerically(">", 0),
					"memory.events high counter should increment when usage exceeds memory.high")

				events, err := memqosReadMemoryEvents(containerCgroupPath)
				framework.ExpectNoError(err)
				framework.Logf("Container %s memory.events: high=%d, max=%d, oom=%d, oom_kill=%d",
					cs.Name, events["high"], events["max"], events["oom"], events["oom_kill"])
				gomega.Expect(events["oom_kill"]).To(gomega.Equal(int64(0)))
			}
		})

		ginkgo.It("should OOM kill container when memory usage exceeds memory.max", func(ctx context.Context) {
			configureMemoryQoS(ctx, 0.9)

			requestsMem := resource.MustParse("15Mi")
			limitsMem := resource.MustParse("15Mi")

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "memqos-oom-kill",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:  "oom-trigger",
							Image: "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							Command: []string{"sh", "-c",
								"sleep 5 && dd if=/dev/zero of=/dev/null bs=20M"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{v1.ResourceMemory: requestsMem},
								Limits:   v1.ResourceList{v1.ResourceMemory: limitsMem},
							},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			pod = e2epod.NewPodClient(f).Create(ctx, pod)

			gomega.Eventually(ctx, func(ctx context.Context) bool {
				p, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return false
				}
				for _, cs := range p.Status.ContainerStatuses {
					if cs.State.Terminated != nil && cs.State.Terminated.Reason == "OOMKilled" {
						return true
					}
				}
				return false
			}).WithTimeout(60 * time.Second).WithPolling(2 * time.Second).Should(gomega.BeTrueBecause("container should be OOM killed when exceeding memory.max"))
		})
	})

	f.Describe("tiered protection edge cases", func() {
		ginkgo.AfterEach(func(ctx context.Context) { restoreConfig(ctx) })

		ginkgo.It("should use memory.low for Burstable pod where memory req == limit but CPU differs", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			memSize := resource.MustParse("128Mi")
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "memqos-burstable-equal-mem",
					Namespace: f.Namespace.Name,
				},
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name:    "test",
							Image:   "registry.k8s.io/e2e-test-images/busybox:1.36.1-1",
							Command: []string{"sleep", "infinity"},
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{
									v1.ResourceMemory: memSize,
									v1.ResourceCPU:    resource.MustParse("100m"),
								},
								Limits: v1.ResourceList{
									v1.ResourceMemory: memSize,
									v1.ResourceCPU:    resource.MustParse("200m"),
								},
							},
						},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
			}
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			gomega.Expect(podCgroupPath).NotTo(gomega.BeEmpty())

			for _, cs := range pod.Status.ContainerStatuses {
				containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath, cs.ContainerID, cgroupDriver)
				gomega.Expect(containerCgroupPath).NotTo(gomega.BeEmpty())

				memLow, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
				framework.ExpectNoError(err)
				framework.Logf("Container %s memory.low=%d, expected=%d", cs.Name, memLow, memSize.Value())
				gomega.Expect(memLow).To(gomega.Equal(memSize.Value()),
					"Burstable pod with CPU mismatch should get memory.low")

				memMin, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryMin)
				framework.ExpectNoError(err)
				gomega.Expect(memMin).To(gomega.Equal(int64(0)),
					"Burstable pod should have memory.min=0")
			}
		})

		ginkgo.It("should include burstable requests in kubepods root memory.min", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			requestsMem := resource.MustParse("200Mi")
			pod := memqosMakePod("memqos-hierarchy-check", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("400Mi")})
			e2epod.NewPodClient(f).CreateSync(ctx, pod)

			var kubepodsCgroupPath string
			if cgroupDriver == "systemd" {
				kubepodsCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice")
			} else {
				kubepodsCgroupPath = filepath.Join(cgroupRoot, "kubepods")
			}

			kubepodsMemMin, err := memqosReadCgroupInt64(kubepodsCgroupPath, cgroupMemoryMin)
			framework.ExpectNoError(err)
			framework.Logf("kubepods memory.min=%d, pod requests=%d", kubepodsMemMin, requestsMem.Value())
			gomega.Expect(kubepodsMemMin).To(gomega.BeNumerically(">=", requestsMem.Value()),
				"kubepods memory.min must include burstable pod requests")
		})

		ginkgo.It("should remove burstable memory.low contribution when pod is deleted", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			var burstableCgroupPath string
			if cgroupDriver == "systemd" {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-burstable.slice")
			} else {
				burstableCgroupPath = filepath.Join(cgroupRoot, "kubepods", "burstable")
			}

			beforeLow, err := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			framework.Logf("burstable QoS memory.low before: %d", beforeLow)

			requestsMem := resource.MustParse("200Mi")
			pod := memqosMakePod("memqos-deletion-check", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("400Mi")})
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)

			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(
				gomega.BeNumerically(">=", beforeLow+requestsMem.Value()),
				"burstable QoS memory.low should increase after pod creation")

			e2epod.NewPodClient(f).DeleteSync(ctx, pod.Name, metav1.DeleteOptions{}, 60*time.Second)

			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(burstableCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(
				gomega.BeNumerically("<=", beforeLow),
				"burstable QoS memory.low should decrease after pod deletion")
		})

		ginkgo.It("should persist container-level memory.low after rollback [known limitation]", func(ctx context.Context) {
			configureMemoryQoSWithPolicy(ctx, 0.9, kubeletconfig.TieredReservationMemoryReservationPolicy)

			requestsMem := resource.MustParse("128Mi")
			pod := memqosMakePod("memqos-container-rollback", f.Namespace.Name,
				v1.ResourceList{v1.ResourceMemory: requestsMem},
				v1.ResourceList{v1.ResourceMemory: resource.MustParse("256Mi")})
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			podCgroupPath := memqosGetPodCgroupPath(pod, cgroupDriver)
			gomega.Expect(podCgroupPath).NotTo(gomega.BeEmpty())

			containerCgroupPath := memqosGetContainerCgroupPath(podCgroupPath,
				pod.Status.ContainerStatuses[0].ContainerID, cgroupDriver)
			gomega.Expect(containerCgroupPath).NotTo(gomega.BeEmpty())

			memLowBefore, err := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
			framework.ExpectNoError(err)
			gomega.Expect(memLowBefore).To(gomega.Equal(requestsMem.Value()))

			ginkgo.By("Disabling MemoryQoS")
			newCfg := oldCfg.DeepCopy()
			if newCfg.FeatureGates == nil {
				newCfg.FeatureGates = make(map[string]bool)
			}
			newCfg.FeatureGates["MemoryQoS"] = false
			newCfg.MemoryReservationPolicy = kubeletconfig.NoneMemoryReservationPolicy
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
			restartKubelet(ctx, true)
			waitForKubeletToStart(ctx, f)

			gomega.Eventually(ctx, func() int64 {
				val, _ := memqosReadCgroupInt64(containerCgroupPath, cgroupMemoryLow)
				return val
			}).WithTimeout(2*time.Minute).WithPolling(5*time.Second).Should(
				gomega.Equal(memLowBefore),
				"container-level memory.low persists after rollback (CRI limitation)")
		})
	})
})

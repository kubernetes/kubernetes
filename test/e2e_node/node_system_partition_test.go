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
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	systemPartitionMemoryLimit = "1Gi"
	// Moving a pod between hierarchies restarts its containers, so give the sync
	// loop room to finish.
	podMoveTimeout      = 2 * time.Minute
	podMovePollInterval = 5 * time.Second
)

// systemPartitionCgroupPath returns the cgroup path of the system partition root.
func systemPartitionCgroupPath(cgroupDriver string) string {
	if cgroupDriver == "systemd" {
		return filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-system.slice")
	}
	return filepath.Join(cgroupRoot, "kubepods", "system")
}

// systemPartitionPodCgroupPath returns the cgroup path a pod would have inside the
// system partition.
func systemPartitionPodCgroupPath(pod *v1.Pod, cgroupDriver string) string {
	uid := string(pod.UID)
	if cgroupDriver == "systemd" {
		uid = strings.ReplaceAll(uid, "-", "_")
		root := filepath.Join(cgroupRoot, "kubepods.slice", "kubepods-system.slice")
		switch pod.Status.QOSClass {
		case v1.PodQOSGuaranteed:
			return filepath.Join(root, fmt.Sprintf("kubepods-system-pod%s.slice", uid))
		case v1.PodQOSBurstable:
			return filepath.Join(root, "kubepods-system-burstable.slice",
				fmt.Sprintf("kubepods-system-burstable-pod%s.slice", uid))
		case v1.PodQOSBestEffort:
			return filepath.Join(root, "kubepods-system-besteffort.slice",
				fmt.Sprintf("kubepods-system-besteffort-pod%s.slice", uid))
		}
		return ""
	}

	root := filepath.Join(cgroupRoot, "kubepods", "system")
	switch pod.Status.QOSClass {
	case v1.PodQOSGuaranteed:
		return filepath.Join(root, fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBurstable:
		return filepath.Join(root, "burstable", fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBestEffort:
		return filepath.Join(root, "besteffort", fmt.Sprintf("pod%s", uid))
	}
	return ""
}

// systemPartitionPods returns the pod count the kubelet reports for the system
// partition.
func systemPartitionPods(ctx context.Context) (float64, error) {
	metrics, err := getKubeletMetrics(ctx)
	if err != nil {
		return 0, err
	}
	return getCounterMetricValue(metrics, "kubelet_partition_pods", map[string]string{"partition": "system"})
}

// newSystemPartitionPod returns a burstable pod that stays running.
func newSystemPartitionPod(name, namespace string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyAlways,
			Containers: []v1.Container{
				{
					Name:    "busybox",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c", "sleep 3600"},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("32Mi")},
					},
				},
			},
		},
	}
}

var _ = SIGDescribe("NodeSystemPartition", framework.WithSerial(), feature.NodeSystemPartition, framework.WithFeatureGate(features.NodeSystemPartition), func() {
	f := framework.NewDefaultFramework("node-system-partition")
	addAfterEachForCleaningUpPods(f)
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		oldCfg       *kubeletconfig.KubeletConfiguration
		cgroupDriver string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("the node system partition requires cgroups v2")
		}
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)
		cgroupDriver = oldCfg.CgroupDriver
	})

	// configureSystemPartition restarts kubelet with the system partition holding
	// the given namespaces.
	configureSystemPartition := func(ctx context.Context, namespaces ...string) {
		newCfg := oldCfg.DeepCopy()
		if newCfg.FeatureGates == nil {
			newCfg.FeatureGates = make(map[string]bool)
		}
		newCfg.FeatureGates["NodeSystemPartition"] = true
		newCfg.CgroupsPerQOS = true
		newCfg.EnforceNodeAllocatable = []string{"pods"}
		newCfg.SystemPartition = &kubeletconfig.SystemPartitionConfiguration{
			MemoryLimit: systemPartitionMemoryLimit,
			Namespaces:  namespaces,
		}
		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
	}

	// disableSystemPartition restarts kubelet with the feature turned off.
	disableSystemPartition := func(ctx context.Context) {
		newCfg := oldCfg.DeepCopy()
		if newCfg.FeatureGates == nil {
			newCfg.FeatureGates = make(map[string]bool)
		}
		newCfg.FeatureGates["NodeSystemPartition"] = false
		newCfg.SystemPartition = nil
		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
	}

	ginkgo.AfterEach(func(ctx context.Context) {
		if oldCfg != nil {
			framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(oldCfg))
			restartKubelet(ctx, true)
			waitForKubeletToStart(ctx, f)
		}
	})

	ginkgo.It("creates the partition cgroup and applies its memory limit", func(ctx context.Context) {
		configureSystemPartition(ctx, f.Namespace.Name)

		partitionPath := systemPartitionCgroupPath(cgroupDriver)
		gomega.Expect(partitionPath).To(gomega.BeADirectory(),
			"the system partition cgroup should exist once the feature is configured")

		limit, err := memqosReadCgroupInt64(partitionPath, cgroupMemoryMax)
		framework.ExpectNoError(err)
		want := resource.MustParse(systemPartitionMemoryLimit)
		gomega.Expect(limit).To(gomega.Equal(want.Value()),
			"the partition root should carry the configured memoryLimit")

		// The namespace is fresh, so the partition is configured but empty. The
		// series has to be there anyway, it is what tells the partition exists.
		gomega.Eventually(ctx, systemPartitionPods, time.Minute, podMovePollInterval).Should(gomega.BeEquivalentTo(0),
			"an empty system partition should report zero pods")
	})

	ginkgo.It("places pods of the configured namespaces in the partition and leaves the others alone", func(ctx context.Context) {
		otherNs, err := f.CreateNamespace(ctx, "node-system-partition-pod-placement", nil)
		framework.ExpectNoError(err)

		configureSystemPartition(ctx, f.Namespace.Name)

		inPartition := e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("in-partition", f.Namespace.Name))
		outOfPartition := e2epod.PodClientNS(f, otherNs.Name).CreateSync(ctx, newSystemPartitionPod("out-of-partition", otherNs.Name))

		gomega.Expect(systemPartitionPodCgroupPath(inPartition, cgroupDriver)).To(gomega.BeADirectory(),
			"should have a cgroup in the configured system partition")
		gomega.Expect(memqosGetPodCgroupPath(inPartition, cgroupDriver)).NotTo(gomega.BeADirectory(),
			"should not have a cgroup in the default hierarchy")

		gomega.Expect(memqosGetPodCgroupPath(outOfPartition, cgroupDriver)).To(gomega.BeADirectory(),
			"should have a cgroup in the default hierarchy")
		gomega.Expect(systemPartitionPodCgroupPath(outOfPartition, cgroupDriver)).NotTo(gomega.BeADirectory(),
			"should not have a cgroup in the configured system partition")

		gomega.Eventually(ctx, systemPartitionPods, time.Minute, podMovePollInterval).Should(gomega.BeEquivalentTo(1),
			"only the pod of the configured namespace should be counted in the system partition")
	})

	ginkgo.It("moves a pod into the partition when the feature is turned on, leaving nothing behind for the default hierarchy", func(ctx context.Context) {
		disableSystemPartition(ctx)

		pod := e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("moved", f.Namespace.Name))
		defaultPath := memqosGetPodCgroupPath(pod, cgroupDriver)
		gomega.Expect(defaultPath).To(gomega.BeADirectory(),
			"the pod starts in the default hierarchy while the feature is off")

		configureSystemPartition(ctx, f.Namespace.Name)

		// The pod is recreated under the system partition on the next sync, which the
		// kubelet does by restarting its containers.
		ginkgo.By("waiting for the pod to be moved into the system partition")
		gomega.Eventually(ctx, func() (bool, error) {
			current, err := e2epod.NewPodClient(f).Get(ctx, pod.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			_, err = os.Stat(systemPartitionPodCgroupPath(current, cgroupDriver))
			return err == nil, nil
		}, podMoveTimeout, podMovePollInterval).Should(
			gomega.BeTrueBecause("the pod should end up in the partition"))

		ginkgo.By("waiting for the cgroup in the default hierarchy to be removed")
		gomega.Eventually(ctx, func() bool {
			_, err := os.Stat(defaultPath)
			return os.IsNotExist(err)
		}, podMoveTimeout, podMovePollInterval).Should(
			gomega.BeTrueBecause("the cgroup left in the default hierarchy should be removed"))
	})

	ginkgo.Context("with the static CPU Manager policy and a partition cpuset", func() {
		var partitionCPUs cpuset.CPUSet

		ginkgo.BeforeEach(func(ctx context.Context) {
			onlineCPUs, err := getOnlineCPUs()
			framework.ExpectNoError(err)
			// Two CPUs for the partition, and enough left for a pinned user pod
			// next to the shared pool.
			if onlineCPUs.Size() < 4 {
				e2eskipper.Skipf("needs at least 4 online CPUs, have %d", onlineCPUs.Size())
			}
			partitionCPUs = cpuset.New(onlineCPUs.List()[:2]...)

			// The partition cpuset must be a subset of reservedSystemCPUs under the
			// static policy, so reserve exactly the partition's CPUs.
			newCfg := configureCPUManagerInKubelet(oldCfg, &cpuManagerKubeletArguments{
				policyName:         string(cpumanager.PolicyStatic),
				reservedSystemCPUs: partitionCPUs,
			})
			newCfg.FeatureGates["NodeSystemPartition"] = true
			newCfg.CgroupsPerQOS = true
			newCfg.EnforceNodeAllocatable = []string{"pods"}
			newCfg.SystemPartition = &kubeletconfig.SystemPartitionConfiguration{
				CPUSet:     partitionCPUs.String(),
				Namespaces: []string{f.Namespace.Name},
			}
			// The CPU Manager checkpoint records the policy, so switching it needs
			// the state file gone.
			updateKubeletConfig(ctx, f, newCfg, true)
		})

		ginkgo.AfterEach(func(ctx context.Context) {
			// The outer AfterEach restores the config too, but it keeps the state
			// file, which would stop kubelet from going back to the none policy.
			updateKubeletConfig(ctx, f, oldCfg, true)
		})

		podResourcesClient := func(ctx context.Context) kubeletpodresourcesv1.PodResourcesListerClient {
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err, "LocalEndpoint() failed")
			cli, conn, err := podresources.GetV1Client(ctx, endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err, "GetV1Client() failed")
			ginkgo.DeferCleanup(conn.Close)
			return cli
		}

		// containerCPUIDs returns the exclusive CPUs podresources reports for the
		// pod's only container.
		containerCPUIDs := func(ctx context.Context, cli kubeletpodresourcesv1.PodResourcesListerClient, pod *v1.Pod) cpuset.CPUSet {
			resp := podresourcesGetWithRetry(ctx, cli, pod.Namespace, pod.Name)
			containers := resp.GetPodResources().GetContainers()
			gomega.Expect(containers).To(gomega.HaveLen(1), "podresources should report the pod's only container")
			var cpus []int
			for _, id := range containers[0].GetCpuIds() {
				cpus = append(cpus, int(id))
			}
			return cpuset.New(cpus...)
		}

		// allowedCPUs returns the CPUs the pod's only container could run on when
		// it started, as reported by the kernel from inside the container. Unlike
		// the requested cpuset, this is what cgroup v2 actually granted.
		allowedCPUs := func(ctx context.Context, pod *v1.Pod) cpuset.CPUSet {
			ctnName := pod.Spec.Containers[0].Name
			logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod.Namespace, pod.Name, ctnName)
			framework.ExpectNoError(err, "failed to read the logs of %s/%s", pod.Namespace, pod.Name)
			return getContainerAllowedCPUsFromLogs(pod.Name, ctnName, logs)
		}

		ginkgo.It("does not pin a guaranteed system pod, which runs on the partition cpuset", func(ctx context.Context) {
			cli := podResourcesClient(ctx)

			pod := makeCPUManagerPod("gu-system", []ctnAttribute{{ctnName: "gu-container", cpuRequest: "1000m", cpuLimit: "1000m"}})
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			gomega.Expect(pod.Status.QOSClass).To(gomega.Equal(v1.PodQOSGuaranteed))
			gomega.Expect(systemPartitionPodCgroupPath(pod, cgroupDriver)).To(gomega.BeADirectory(),
				"the pod should be placed in the system partition")

			gomega.Expect(allowedCPUs(ctx, pod).Equals(partitionCPUs)).To(gomega.BeTrueBecause(
				"the pod should run on the whole partition cpuset %s, not on a single pinned CPU", partitionCPUs))
			gomega.Expect(containerCPUIDs(ctx, cli, pod).IsEmpty()).To(gomega.BeTrueBecause(
				"podresources should report no exclusive CPUs for a pod the CPU Manager did not pin"))

			ginkgo.By("checking that no CPU outside the partition was taken out of the shared pool")
			resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err, "GetAllocatableResources() failed")
			allocatable, _ := demuxCPUsAndDevicesFromGetAllocatableResources(resp)
			gomega.Expect(allocatable.Intersection(partitionCPUs).IsEmpty()).To(gomega.BeTrueBecause(
				"the partition CPUs %s are reserved and must not be allocatable, got %s", partitionCPUs, allocatable))
		})

		ginkgo.It("still pins a guaranteed user pod, away from the partition cpuset", func(ctx context.Context) {
			cli := podResourcesClient(ctx)
			otherNs, err := f.CreateNamespace(ctx, "node-system-partition-cpu-user", nil)
			framework.ExpectNoError(err)

			pod := makeCPUManagerPod("gu-user", []ctnAttribute{{ctnName: "gu-container", cpuRequest: "1000m", cpuLimit: "1000m"}})
			pod = e2epod.PodClientNS(f, otherNs.Name).CreateSync(ctx, pod)
			gomega.Expect(memqosGetPodCgroupPath(pod, cgroupDriver)).To(gomega.BeADirectory(),
				"the pod should stay in the default hierarchy")

			exclusive := containerCPUIDs(ctx, cli, pod)
			gomega.Expect(exclusive.Size()).To(gomega.Equal(1),
				"podresources should report the single exclusive CPU of the user pod")
			gomega.Expect(exclusive.Intersection(partitionCPUs).IsEmpty()).To(gomega.BeTrueBecause(
				"the user pod must not be pinned onto the partition CPUs %s, got %s", partitionCPUs, exclusive))
			gomega.Expect(allowedCPUs(ctx, pod).Equals(exclusive)).To(gomega.BeTrueBecause(
				"the user pod should run on exactly the CPU podresources reports"))
		})
	})

	ginkgo.It("creates no partition cgroup when the feature is disabled", func(ctx context.Context) {
		disableSystemPartition(ctx)

		gomega.Expect(systemPartitionCgroupPath(cgroupDriver)).NotTo(gomega.BeADirectory(),
			"no system partition cgroup should exist while the feature is off")

		pod := e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("default-only", f.Namespace.Name))
		gomega.Expect(memqosGetPodCgroupPath(pod, cgroupDriver)).To(gomega.BeADirectory(),
			"pods run in the default hierarchy after the feature is disabled")

		_, err := systemPartitionPods(ctx)
		gomega.Expect(errors.Is(err, ErrMetricNotFound)).To(gomega.BeTrueBecause(
			"a node without a system partition should report no partition pods series, got err %v", err))
	})
})

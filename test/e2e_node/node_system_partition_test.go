//go:build linux

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
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	stats "k8s.io/kubelet/pkg/apis/stats/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/eviction"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	e2enodekubelet "k8s.io/kubernetes/test/e2e_node/kubeletconfig"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
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

// systemPartitionBurstableCgroupPath returns the cgroup path of the system
// partition's burstable QoS cgroup.
func systemPartitionBurstableCgroupPath(cgroupDriver string) string {
	if cgroupDriver == "systemd" {
		return filepath.Join(systemPartitionCgroupPath(cgroupDriver), "kubepods-system-burstable.slice")
	}
	return filepath.Join(systemPartitionCgroupPath(cgroupDriver), "burstable")
}

// podsCgroupPath returns the cgroup path of kubepods, which holds all pods.
func podsCgroupPath(cgroupDriver string) string {
	if cgroupDriver == "systemd" {
		return filepath.Join(cgroupRoot, "kubepods.slice")
	}
	return filepath.Join(cgroupRoot, "kubepods")
}

// defaultBurstableCgroupPath returns the cgroup path of the default
// partition's burstable QoS cgroup.
func defaultBurstableCgroupPath(cgroupDriver string) string {
	if cgroupDriver == "systemd" {
		return filepath.Join(podsCgroupPath(cgroupDriver), "kubepods-burstable.slice")
	}
	return filepath.Join(podsCgroupPath(cgroupDriver), "burstable")
}

// systemPartitionPodCgroupPath returns the cgroup path a pod would have inside the
// system partition.
func systemPartitionPodCgroupPath(pod *v1.Pod, cgroupDriver string) string {
	uid := string(pod.UID)
	root := systemPartitionCgroupPath(cgroupDriver)
	burstable := systemPartitionBurstableCgroupPath(cgroupDriver)
	if cgroupDriver == "systemd" {
		uid = strings.ReplaceAll(uid, "-", "_")
		switch pod.Status.QOSClass {
		case v1.PodQOSGuaranteed:
			return filepath.Join(root, fmt.Sprintf("kubepods-system-pod%s.slice", uid))
		case v1.PodQOSBurstable:
			return filepath.Join(burstable, fmt.Sprintf("kubepods-system-burstable-pod%s.slice", uid))
		case v1.PodQOSBestEffort:
			return filepath.Join(root, "kubepods-system-besteffort.slice",
				fmt.Sprintf("kubepods-system-besteffort-pod%s.slice", uid))
		}
		return ""
	}

	switch pod.Status.QOSClass {
	case v1.PodQOSGuaranteed:
		return filepath.Join(root, fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBurstable:
		return filepath.Join(burstable, fmt.Sprintf("pod%s", uid))
	case v1.PodQOSBestEffort:
		return filepath.Join(root, "besteffort", fmt.Sprintf("pod%s", uid))
	}
	return ""
}

// waitForPodCgroup waits for the pod's cgroup to show up at the path that
// cgroupPath gives for its current state.
func waitForPodCgroup(ctx context.Context, f *framework.Framework, podName string, cgroupPath func(*v1.Pod) string, reason string) {
	ginkgo.GinkgoHelper()
	gomega.Eventually(ctx, func() (bool, error) {
		current, err := e2epod.NewPodClient(f).Get(ctx, podName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		_, err = os.Stat(cgroupPath(current))
		return err == nil, nil
	}, podMoveTimeout, podMovePollInterval).Should(gomega.BeTrueBecause("%s", reason))
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

// newPartitionMemhogPod returns a pod that allocates 50Mi every 5s up to 900Mi,
// fast enough to cross an eviction threshold within a minute or so.
func newPartitionMemhogPod(name, namespace string) *v1.Pod {
	var gracePeriod int64 = 1
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.PodSpec{
			RestartPolicy:                 v1.RestartPolicyNever,
			TerminationGracePeriodSeconds: &gracePeriod,
			Containers: []v1.Container{
				{
					Name:  "memhog",
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"stress", "--mem-alloc-size", "50Mi", "--mem-alloc-sleep", "5s", "--mem-total", strconv.Itoa(900 << 20)},
					Resources: v1.ResourceRequirements{
						Requests: v1.ResourceList{v1.ResourceMemory: resource.MustParse("32Mi")},
					},
				},
			},
		},
	}
}

// evictionEvents returns the eviction events recorded for the pod.
func evictionEvents(ctx context.Context, f *framework.Framework, pod *v1.Pod) []v1.Event {
	ginkgo.GinkgoHelper()
	selector := fields.Set{
		"involvedObject.kind":      "Pod",
		"involvedObject.name":      pod.Name,
		"involvedObject.namespace": pod.Namespace,
		"reason":                   eviction.Reason,
	}.AsSelector().String()
	events, err := f.ClientSet.CoreV1().Events(pod.Namespace).List(ctx, metav1.ListOptions{FieldSelector: selector})
	framework.ExpectNoError(err, "listing the eviction events of %s/%s", pod.Namespace, pod.Name)
	return events.Items
}

// newBurstablePod returns a burstable pod that stays running and requests cpu.
func newBurstablePod(name, namespace, cpu string) *v1.Pod {
	pod := newSystemPartitionPod(name, namespace)
	pod.Spec.Containers[0].Resources.Requests[v1.ResourceCPU] = resource.MustParse(cpu)
	return pod
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
		runtimeName  string
	)

	ginkgo.BeforeEach(func(ctx context.Context) {
		if !IsCgroup2UnifiedMode() {
			ginkgo.Skip("the node system partition requires cgroups v2")
		}
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)
		cgroupDriver = oldCfg.CgroupDriver
		runtime, _, err := getCRIClient(ctx)
		framework.ExpectNoError(err, "failed to get the CRI client")
		version, err := runtime.Version(ctx, "")
		framework.ExpectNoError(err, "failed to get the runtime version")
		runtimeName = version.GetRuntimeName()
	})

	// allowedCPUs returns the CPUs that cgroup v2 grants the pod's only
	// container, read from the container's cgroup on the host. podCgroupPath
	// is where the pod's own cgroup is.
	allowedCPUs := func(pod *v1.Pod, podCgroupPath func(*v1.Pod, string) string) cpuset.CPUSet {
		ginkgo.GinkgoHelper()
		fullID, found := findContainerIDByName(pod, pod.Spec.Containers[0].Name, false)
		gomega.Expect(found).To(gomega.BeTrueBecause("the status of %s/%s should report its container", pod.Namespace, pod.Name))
		id, err := parseContainerID(fullID)
		framework.ExpectNoError(err)
		containerCgroup := id
		if cgroupDriver == "systemd" {
			containerCgroup = containerCgroupPathPrefixFromDriver(runtimeName) + "-" + id + ".scope"
		}
		data, err := os.ReadFile(filepath.Join(podCgroupPath(pod, cgroupDriver), containerCgroup, "cpuset.cpus.effective"))
		framework.ExpectNoError(err, "failed to read the cpuset of %s/%s", pod.Namespace, pod.Name)
		cpus, err := cpuset.Parse(strings.TrimSpace(string(data)))
		framework.ExpectNoError(err)
		return cpus
	}

	// createPodInNamespace creates the pod in one of the test's extra namespaces,
	// and deletes it once the test ends. The framework only waits for the pods
	// of its own namespace to be gone, so a pod left in another one could still
	// run, and be counted in the QoS cgroups, while the next test starts.
	createPodInNamespace := func(ctx context.Context, namespace string, pod *v1.Pod) *v1.Pod {
		ginkgo.GinkgoHelper()
		client := e2epod.PodClientNS(f, namespace)
		created := client.CreateSync(ctx, pod)
		ginkgo.DeferCleanup(func(ctx context.Context) {
			client.DeleteSync(ctx, created.Name, metav1.DeleteOptions{GracePeriodSeconds: new(int64(0))}, f.Timeouts.PodDelete)
			waitForAllContainerRemoval(ctx, created.Name, created.Namespace)
		})
		return created
	}

	// configureSystemPartitionWith restarts kubelet with the given system partition.
	configureSystemPartitionWith := func(ctx context.Context, sp *kubeletconfig.SystemPartitionConfiguration) {
		newCfg := oldCfg.DeepCopy()
		if newCfg.FeatureGates == nil {
			newCfg.FeatureGates = make(map[string]bool)
		}
		newCfg.FeatureGates["NodeSystemPartition"] = true
		newCfg.CgroupsPerQOS = true
		newCfg.EnforceNodeAllocatable = []string{"pods"}
		newCfg.SystemPartition = sp
		framework.ExpectNoError(e2enodekubelet.WriteKubeletConfigFile(newCfg))
		restartKubelet(ctx, true)
		waitForKubeletToStart(ctx, f)
	}

	// configureSystemPartition restarts kubelet with the system partition holding
	// the given namespaces.
	configureSystemPartition := func(ctx context.Context, namespaces ...string) {
		configureSystemPartitionWith(ctx, &kubeletconfig.SystemPartitionConfiguration{
			MemoryLimit: systemPartitionMemoryLimit,
			Namespaces:  namespaces,
		})
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
		outOfPartition := createPodInNamespace(ctx, otherNs.Name, newSystemPartitionPod("out-of-partition", otherNs.Name))

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

	ginkgo.It("evicts only a system pod when the partition runs low on memory", func(ctx context.Context) {
		otherNs, err := f.CreateNamespace(ctx, "node-system-partition-eviction-user", nil)
		framework.ExpectNoError(err)

		// Eviction starts once less than 400Mi of the 1Gi partition is left, well
		// before the memhog below reaches its 900Mi, so it is evicted rather than
		// OOM killed.
		configureSystemPartitionWith(ctx, &kubeletconfig.SystemPartitionConfiguration{
			MemoryLimit:  systemPartitionMemoryLimit,
			EvictionHard: map[string]string{"memory.available": "400Mi"},
			Namespaces:   []string{f.Namespace.Name},
		})

		idle := e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("idle", f.Namespace.Name))
		user := createPodInNamespace(ctx, otherNs.Name, newSystemPartitionPod("user", otherNs.Name))
		memhog := e2epod.NewPodClient(f).Create(ctx, newPartitionMemhogPod("memhog", f.Namespace.Name))

		ginkgo.By("waiting for the memhog to be evicted")
		gomega.Eventually(ctx, func() (*v1.Pod, error) {
			return e2epod.NewPodClient(f).Get(ctx, memhog.Name, metav1.GetOptions{})
		}, 5*time.Minute, podMovePollInterval).Should(gomega.And(
			gomega.HaveField("Status.Phase", v1.PodFailed),
			gomega.HaveField("Status.Reason", eviction.Reason),
		), "the memhog in the partition should be evicted")

		ginkgo.By("checking that no other pod is evicted once the memhog is gone")
		gomega.Consistently(ctx, func(ctx context.Context) error {
			for _, pod := range []*v1.Pod{idle, user} {
				current, err := f.ClientSet.CoreV1().Pods(pod.Namespace).Get(ctx, pod.Name, metav1.GetOptions{})
				if err != nil {
					return err
				}
				if current.Status.Phase != v1.PodRunning || current.Status.Reason == eviction.Reason {
					return fmt.Errorf("%s/%s should keep running, only the memhog uses much memory, got phase %s and reason %q",
						pod.Namespace, pod.Name, current.Status.Phase, current.Status.Reason)
				}
			}
			if hasNodeCondition(ctx, f, v1.NodeMemoryPressure) {
				return fmt.Errorf("pressure inside the partition should not be reported as node MemoryPressure")
			}
			return nil
		}, time.Minute, 10*time.Second).Should(gomega.Succeed())

		ginkgo.By("checking the eviction events")
		gomega.Expect(evictionEvents(ctx, f, memhog)).To(gomega.HaveExactElements(
			gomega.HaveField("Annotations", gomega.HaveKeyWithValue(eviction.StarvedResourceKey, string(v1.ResourceMemory))),
		), "the memhog should have exactly one eviction event, for memory")
		for _, pod := range []*v1.Pod{idle, user} {
			gomega.Expect(evictionEvents(ctx, f, pod)).To(gomega.BeEmpty(), "%s/%s should have no eviction event", pod.Namespace, pod.Name)
		}
	})

	ginkgo.It("weights the partition by the CPU requests of its pods, and only them", func(ctx context.Context) {
		otherNs, err := f.CreateNamespace(ctx, "node-system-partition-weight-user", nil)
		framework.ExpectNoError(err)
		configureSystemPartition(ctx, f.Namespace.Name)

		partitionWeight := filepath.Join(systemPartitionCgroupPath(cgroupDriver), "cpu.weight")
		partitionBurstableWeight := filepath.Join(systemPartitionBurstableCgroupPath(cgroupDriver), "cpu.weight")
		defaultBurstable := defaultBurstableCgroupPath(cgroupDriver)
		expectWeight := func(path string, milliCPU int64, reason string) {
			ginkgo.GinkgoHelper()
			gomega.Eventually(ctx, func() error {
				return expectFileValToEqual(path, convertSharesToWeight(int64(cm.MilliCPUToShares(milliCPU))), 1)
			}, time.Minute, podMovePollInterval).Should(gomega.Succeed(), reason)
		}

		// Without pods, the partition gets the minimum weight, as an idle pod
		// cgroup would.
		gomega.Eventually(ctx, func() error {
			return expectFileValToEqual(partitionWeight, convertSharesToWeight(int64(cm.MinShares)), 0)
		}, time.Minute, podMovePollInterval).Should(gomega.Succeed())

		// Other pods on the node weight the default partition's burstable
		// cgroup too, so it is only checked for what this test changes.
		defaultBurstableWeight, err := memqosReadCgroupInt64(defaultBurstable, "cpu.weight")
		framework.ExpectNoError(err)

		ginkgo.By("creating pods in the partition")
		guaranteed := newSystemPartitionPod("guaranteed", f.Namespace.Name)
		guaranteed.Spec.Containers[0].Resources = v1.ResourceRequirements{
			Requests: v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("64Mi")},
			Limits:   v1.ResourceList{v1.ResourceCPU: resource.MustParse("500m"), v1.ResourceMemory: resource.MustParse("64Mi")},
		}
		e2epod.NewPodClient(f).CreateSync(ctx, guaranteed)
		e2epod.NewPodClient(f).CreateSync(ctx, newBurstablePod("burstable", f.Namespace.Name, "300m"))

		// The partition competes for CPU with the default partition's pods
		// under kubepods, so it has to weigh as much as all its pods request.
		expectWeight(partitionWeight, 800, "the partition should weigh what all its pods request")
		expectWeight(partitionBurstableWeight, 300, "the partition's burstable cgroup should weigh its burstable pod")
		// Periodic QoS updates must not fold partition pods into the default
		// burstable cgroup.
		gomega.Consistently(ctx, func() (int64, error) {
			return memqosReadCgroupInt64(defaultBurstable, "cpu.weight")
		}, 15*time.Second, podMovePollInterval).Should(gomega.Equal(defaultBurstableWeight),
			"the default partition's burstable cgroup should not count the partition's pods")

		ginkgo.By("creating a pod outside the partition")
		createPodInNamespace(ctx, otherNs.Name, newBurstablePod("user", otherNs.Name, "700m"))
		// Wait for the default hierarchy to observe the new pod before rechecking
		// the partition weights.
		gomega.Eventually(ctx, func() (int64, error) {
			return memqosReadCgroupInt64(defaultBurstable, "cpu.weight")
		}, time.Minute, podMovePollInterval).Should(gomega.BeNumerically(">", defaultBurstableWeight),
			"the default partition's burstable cgroup should count the new pod")
		expectWeight(partitionWeight, 800, "a pod outside the partition should not weigh the partition")
		expectWeight(partitionBurstableWeight, 300, "a pod outside the partition should not weigh the partition's burstable cgroup")
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

		ginkgo.It("does not pin a guaranteed system pod, which runs on the partition cpuset", func(ctx context.Context) {
			cli := podResourcesClient(ctx)

			pod := makeCPUManagerPod("gu-system", []ctnAttribute{{ctnName: "gu-container", cpuRequest: "1000m", cpuLimit: "1000m"}})
			pod = e2epod.NewPodClient(f).CreateSync(ctx, pod)
			gomega.Expect(pod.Status.QOSClass).To(gomega.Equal(v1.PodQOSGuaranteed))
			gomega.Expect(systemPartitionPodCgroupPath(pod, cgroupDriver)).To(gomega.BeADirectory(),
				"the pod should be placed in the system partition")

			gomega.Expect(allowedCPUs(pod, systemPartitionPodCgroupPath).Equals(partitionCPUs)).To(gomega.BeTrueBecause(
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
			pod = createPodInNamespace(ctx, otherNs.Name, pod)
			gomega.Expect(memqosGetPodCgroupPath(pod, cgroupDriver)).To(gomega.BeADirectory(),
				"the pod should stay in the default hierarchy")

			exclusive := containerCPUIDs(ctx, cli, pod)
			gomega.Expect(exclusive.Size()).To(gomega.Equal(1),
				"podresources should report the single exclusive CPU of the user pod")
			gomega.Expect(exclusive.Intersection(partitionCPUs).IsEmpty()).To(gomega.BeTrueBecause(
				"the user pod must not be pinned onto the partition CPUs %s, got %s", partitionCPUs, exclusive))
			gomega.Expect(allowedCPUs(pod, memqosGetPodCgroupPath).Equals(exclusive)).To(gomega.BeTrueBecause(
				"the user pod should run on exactly the CPU podresources reports"))
		})
	})

	ginkgo.It("confines the pods of the partition to its cpuset, and only them", func(ctx context.Context) {
		onlineCPUs, err := getOnlineCPUs()
		framework.ExpectNoError(err)
		if onlineCPUs.Size() < 2 {
			e2eskipper.Skipf("needs at least 2 online CPUs, have %d", onlineCPUs.Size())
		}
		// Under the none CPU Manager policy every container runs on all the CPUs
		// its cgroup allows, which is what this test relies on. The static
		// policy is covered separately below.
		if policy := oldCfg.CPUManagerPolicy; policy != "" && policy != string(cpumanager.PolicyNone) {
			e2eskipper.Skipf("needs the none CPU Manager policy, the node runs %q", policy)
		}
		partitionCPUs := cpuset.New(onlineCPUs.List()[0])
		otherNs, err := f.CreateNamespace(ctx, "node-system-partition-cpuset-user", nil)
		framework.ExpectNoError(err)

		configureSystemPartitionWith(ctx, &kubeletconfig.SystemPartitionConfiguration{
			MemoryLimit: systemPartitionMemoryLimit,
			CPUSet:      partitionCPUs.String(),
			Namespaces:  []string{f.Namespace.Name},
		})

		data, err := os.ReadFile(filepath.Join(systemPartitionCgroupPath(cgroupDriver), "cpuset.cpus"))
		framework.ExpectNoError(err)
		rootCPUs, err := cpuset.Parse(strings.TrimSpace(string(data)))
		framework.ExpectNoError(err)
		gomega.Expect(rootCPUs.Equals(partitionCPUs)).To(gomega.BeTrueBecause(
			"the partition root should carry the configured cpuset %s, got %s", partitionCPUs, rootCPUs))

		burstable := []ctnAttribute{{ctnName: "burstable", cpuRequest: "100m", cpuLimit: "200m"}}
		system := e2epod.NewPodClient(f).CreateSync(ctx, makeCPUManagerPod("system", burstable))
		user := createPodInNamespace(ctx, otherNs.Name, makeCPUManagerPod("user", burstable))

		gomega.Expect(allowedCPUs(system, systemPartitionPodCgroupPath).Equals(partitionCPUs)).To(gomega.BeTrueBecause(
			"a pod of the partition should run on its cpuset %s only", partitionCPUs))
		// The node may confine kubepods itself, so compare against what it grants
		// rather than against all online CPUs.
		data, err = os.ReadFile(filepath.Join(podsCgroupPath(cgroupDriver), "cpuset.cpus.effective"))
		framework.ExpectNoError(err)
		podsCPUs, err := cpuset.Parse(strings.TrimSpace(string(data)))
		framework.ExpectNoError(err)
		gomega.Expect(allowedCPUs(user, memqosGetPodCgroupPath).Equals(podsCPUs)).To(gomega.BeTrueBecause(
			"a pod outside the partition should keep all the CPUs of kubepods %s", podsCPUs))
	})

	ginkgo.It("reports the partition's usage in the summary API", func(ctx context.Context) {
		configureSystemPartition(ctx, f.Namespace.Name)
		e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("in-partition", f.Namespace.Name))

		// Eviction uses the partition entry, so its memory values should fit
		// within the partition limit rather than the node limit.
		memoryLimit := resource.MustParse(systemPartitionMemoryLimit)
		partitionMemory := gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"WorkingSetBytes": bounded(1*e2evolume.Kb, memoryLimit.Value()),
			"AvailableBytes":  bounded(1*e2evolume.Kb, memoryLimit.Value()),
		}))
		podsMemory := gstruct.PointTo(gstruct.MatchFields(gstruct.IgnoreExtras, gstruct.Fields{
			"WorkingSetBytes": gomega.Not(gomega.BeNil()),
		}))

		var containers map[string]*stats.MemoryStats
		gomega.Eventually(ctx, func(ctx context.Context) (map[string]*stats.MemoryStats, error) {
			summary, err := getNodeSummary(ctx)
			if err != nil {
				return nil, err
			}
			containers = map[string]*stats.MemoryStats{}
			for _, container := range summary.Node.SystemContainers {
				containers[container.Name] = container.Memory
			}
			return containers, nil
		}, time.Minute, podMovePollInterval).Should(gstruct.MatchKeys(gstruct.IgnoreExtras, gstruct.Keys{
			stats.SystemContainerSystemPods: partitionMemory,
			stats.SystemContainerPods:       podsMemory,
		}))

		// The partition entry nests inside the entry of all pods rather than
		// splitting the node with it.
		gomega.Expect(*containers[stats.SystemContainerPods].WorkingSetBytes).To(
			gomega.BeNumerically(">=", *containers[stats.SystemContainerSystemPods].WorkingSetBytes),
			"the %q entry should include the %q entry", stats.SystemContainerPods, stats.SystemContainerSystemPods)
	})

	ginkgo.It("moves a pod into the partition when it is turned on, and back out when it is turned off", func(ctx context.Context) {
		ginkgo.By("starting with the feature off")
		disableSystemPartition(ctx)
		gomega.Expect(systemPartitionCgroupPath(cgroupDriver)).NotTo(gomega.BeADirectory(),
			"no system partition cgroup should exist while the feature is off")
		pod := e2epod.NewPodClient(f).CreateSync(ctx, newSystemPartitionPod("moved", f.Namespace.Name))
		defaultPath := memqosGetPodCgroupPath(pod, cgroupDriver)
		gomega.Expect(defaultPath).To(gomega.BeADirectory(),
			"the pod starts in the default hierarchy while the feature is off")

		// Moving a pod between hierarchies is done by restarting its
		// containers on the next sync.
		ginkgo.By("turning the feature on")
		configureSystemPartition(ctx, f.Namespace.Name)
		waitForPodCgroup(ctx, f, pod.Name, func(current *v1.Pod) string {
			return systemPartitionPodCgroupPath(current, cgroupDriver)
		}, "the pod should be moved into the partition")
		gomega.Eventually(ctx, func() bool {
			_, err := os.Stat(defaultPath)
			return os.IsNotExist(err)
		}, podMoveTimeout, podMovePollInterval).Should(
			gomega.BeTrueBecause("the cgroup left in the default hierarchy should be removed"))

		ginkgo.By("turning the feature off again")
		disableSystemPartition(ctx)
		waitForPodCgroup(ctx, f, pod.Name, func(current *v1.Pod) string {
			return memqosGetPodCgroupPath(current, cgroupDriver)
		}, "the pod should be moved back into the default hierarchy")
		// The emptied partition is removed by a periodic task, which runs every
		// few minutes.
		gomega.Eventually(ctx, func() bool {
			_, err := os.Stat(systemPartitionCgroupPath(cgroupDriver))
			return os.IsNotExist(err)
		}, 7*time.Minute, podMovePollInterval).Should(
			gomega.BeTrueBecause("the partition left behind once the feature is off should be removed"))
	})
})

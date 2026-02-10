/*
Copyright 2017 The Kubernetes Authors.

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
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"
	"k8s.io/utils/ptr"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	evictionHardMemory = "memory.available"
	resourceMemory     = "memory"
	staticPolicy       = "Static"
	nonePolicy         = "None"
	hugepages2MiCount  = 8
)

// Helper for makeMemoryManagerPod().
type memoryManagerCtnAttributes struct {
	ctnName      string
	cpus         string
	memory       string
	hugepages2Mi string
}

// makeMemoryManagerContainers returns slice of containers with provided attributes and indicator of hugepages mount needed for those.
func makeMemoryManagerContainers(ctnCmd string, ctnAttributes []memoryManagerCtnAttributes) ([]v1.Container, bool) {
	hugepagesMount := false
	var containers []v1.Container
	for _, ctnAttr := range ctnAttributes {
		res := v1.ResourceRequirements{
			Limits:   v1.ResourceList{},
			Requests: v1.ResourceList{},
		}
		if ctnAttr.cpus != "" {
			res.Limits[v1.ResourceCPU] = resource.MustParse(ctnAttr.cpus)
			res.Requests[v1.ResourceCPU] = resource.MustParse(ctnAttr.cpus)
		}
		if ctnAttr.memory != "" {
			res.Limits[v1.ResourceMemory] = resource.MustParse(ctnAttr.memory)
			res.Requests[v1.ResourceMemory] = resource.MustParse(ctnAttr.memory)
		}

		ctn := v1.Container{
			Name:      ctnAttr.ctnName,
			Image:     busyboxImage,
			Resources: res,
			Command:   []string{"sh", "-c", ctnCmd},
		}
		if ctnAttr.hugepages2Mi != "" {
			hugepagesMount = true

			ctn.Resources.Limits[hugepagesResourceName2Mi] = resource.MustParse(ctnAttr.hugepages2Mi)
			ctn.VolumeMounts = []v1.VolumeMount{
				{
					Name:      "hugepages-2mi",
					MountPath: "/hugepages-2Mi",
				},
			}
		}

		containers = append(containers, ctn)
	}

	return containers, hugepagesMount
}

// makeMemoryMangerPod returns a pod with the provided ctnAttributes.
func makeMemoryManagerPod(podName string, initCtnAttributes, ctnAttributes []memoryManagerCtnAttributes) *v1.Pod {
	hugepagesMount := false
	memsetCmd := "grep Mems_allowed_list /proc/self/status | cut -f2"
	memsetSleepCmd := memsetCmd + "&& sleep 1d"
	var containers, initContainers []v1.Container
	if len(initCtnAttributes) > 0 {
		initContainers, _ = makeMemoryManagerContainers(memsetCmd, initCtnAttributes)
	}
	containers, hugepagesMount = makeMemoryManagerContainers(memsetSleepCmd, ctnAttributes)

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy:  v1.RestartPolicyNever,
			Containers:     containers,
			InitContainers: initContainers,
		},
	}

	if hugepagesMount {
		pod.Spec.Volumes = []v1.Volume{
			{
				Name: "hugepages-2mi",
				VolumeSource: v1.VolumeSource{
					EmptyDir: &v1.EmptyDirVolumeSource{
						Medium: mediumHugepages2Mi,
					},
				},
			},
		}
	}

	return pod
}

func getMemoryManagerState() (*state.MemoryManagerCheckpoint, error) {
	if _, err := os.Stat(memoryManagerStateFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("the memory manager state file %s does not exist", memoryManagerStateFile)
	}

	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat %s", memoryManagerStateFile)).Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run command 'cat %s': out: %s, err: %w", memoryManagerStateFile, out, err)
	}

	memoryManagerCheckpoint := &state.MemoryManagerCheckpoint{}
	if err := json.Unmarshal(out, memoryManagerCheckpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal memory manager state file: %w", err)
	}
	return memoryManagerCheckpoint, nil
}

func getAllocatableMemoryFromStateFile(s *state.MemoryManagerCheckpoint) []state.Block {
	var allocatableMemory []state.Block
	for numaNodeID, numaNodeState := range s.MachineState {
		for resourceName, memoryTable := range numaNodeState.MemoryMap {
			if memoryTable.Allocatable == 0 {
				continue
			}

			block := state.Block{
				NUMAAffinity: []int{numaNodeID},
				Type:         resourceName,
				Size:         memoryTable.Allocatable,
			}
			allocatableMemory = append(allocatableMemory, block)
		}
	}
	return allocatableMemory
}

type memoryManagerKubeletParams struct {
	policy               string
	systemReservedMemory []kubeletconfig.MemoryReservation
	systemReserved       map[string]string
	kubeReserved         map[string]string
	evictionHard         map[string]string
}

func updateKubeletConfigWithMemoryManagerParams(initialCfg *kubeletconfig.KubeletConfiguration, params *memoryManagerKubeletParams) {
	if initialCfg.FeatureGates == nil {
		initialCfg.FeatureGates = map[string]bool{}
	}

	initialCfg.MemoryManagerPolicy = params.policy

	// update system-reserved
	if initialCfg.SystemReserved == nil {
		initialCfg.SystemReserved = map[string]string{}
	}
	for resourceName, value := range params.systemReserved {
		initialCfg.SystemReserved[resourceName] = value
	}

	// update kube-reserved
	if initialCfg.KubeReserved == nil {
		initialCfg.KubeReserved = map[string]string{}
	}
	for resourceName, value := range params.kubeReserved {
		initialCfg.KubeReserved[resourceName] = value
	}

	// update hard eviction threshold
	if initialCfg.EvictionHard == nil {
		initialCfg.EvictionHard = map[string]string{}
	}
	for resourceName, value := range params.evictionHard {
		initialCfg.EvictionHard[resourceName] = value
	}

	// update reserved memory
	if initialCfg.ReservedMemory == nil {
		initialCfg.ReservedMemory = []kubeletconfig.MemoryReservation{}
	}
	initialCfg.ReservedMemory = append(initialCfg.ReservedMemory, params.systemReservedMemory...)
}

func getAllNUMANodes() []int {
	outData, err := exec.Command("/bin/sh", "-c", "lscpu").Output()
	framework.ExpectNoError(err)

	numaNodeRegex, err := regexp.Compile(`NUMA node(\d+) CPU\(s\):`)
	framework.ExpectNoError(err)

	matches := numaNodeRegex.FindAllSubmatch(outData, -1)

	var numaNodes []int
	for _, m := range matches {
		n, err := strconv.Atoi(string(m[1]))
		framework.ExpectNoError(err)

		numaNodes = append(numaNodes, n)
	}

	sort.Ints(numaNodes)
	return numaNodes
}

func verifyMemoryPinning(f *framework.Framework, ctx context.Context, pod *v1.Pod, numaNodeIDs []int) {
	ginkgo.GinkgoHelper()
	ginkgo.By("Verifying the NUMA pinning")

	output, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
	framework.ExpectNoError(err)

	currentNUMANodeIDs, err := cpuset.Parse(strings.Trim(output, "\n"))
	framework.ExpectNoError(err)

	gomega.Expect(numaNodeIDs).To(gomega.Equal(currentNUMANodeIDs.List()))
}

func verifyMemoryManagerAllocations(ctx context.Context, pod *v1.Pod, expectedGuaranteedContainers []string, expectedSharedMemorySize int64) {
	ginkgo.GinkgoHelper()
	ginkgo.By("Verifying memory manager allocations via pod resource API")
	endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
	framework.ExpectNoError(err)

	cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
	framework.ExpectNoError(err)
	defer conn.Close() //nolint:errcheck

	gomega.Eventually(ctx, func(ctx context.Context) error {
		_, err := cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
		return err
	}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
	resp, err := cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
	framework.ExpectNoError(err)

	expectedGuaranteedSet := sets.NewString(expectedGuaranteedContainers...)

	for _, podResource := range resp.PodResources {
		if podResource.Name != pod.Name {
			continue
		}

		for _, containerResource := range podResource.Containers {
			// find the container in the pod spec
			var containerSpec *v1.Container
			for i := range pod.Spec.Containers {
				if pod.Spec.Containers[i].Name == containerResource.Name {
					containerSpec = &pod.Spec.Containers[i]
					break
				}
			}
			if containerSpec == nil {
				// could be an init container
				for i := range pod.Spec.InitContainers {
					if pod.Spec.InitContainers[i].Name == containerResource.Name {
						containerSpec = &pod.Spec.InitContainers[i]
						break
					}
				}
			}
			gomega.Expect(containerSpec).ToNot(gomega.BeNil(), "container spec for %s not found", containerResource.Name)

			if expectedGuaranteedSet.Has(containerResource.Name) {
				gomega.Expect(containerResource.Memory).ToNot(gomega.BeEmpty(), "expected memory resources for container %s", containerResource.Name)
				for _, mem := range containerResource.Memory {
					q := containerSpec.Resources.Limits[v1.ResourceName(mem.MemoryType)]
					val, ok := q.AsInt64()
					gomega.Expect(ok).To(gomega.BeTrueBecause("cannot convert value to integer"))
					gomega.Expect(val).To(gomega.BeEquivalentTo(mem.Size))
				}
			} else {
				if expectedSharedMemorySize > 0 {
					gomega.Expect(containerResource.Memory).ToNot(gomega.BeEmpty(), "expected memory resources for non-guaranteed container %s", containerResource.Name)
					var totalAllocated uint64
					for _, mem := range containerResource.Memory {
						totalAllocated += mem.Size
					}
					gomega.Expect(totalAllocated).To(gomega.BeEquivalentTo(expectedSharedMemorySize), "container %s memory should match shared pool size", containerResource.Name)
				} else {
					gomega.Expect(containerResource.Memory).To(gomega.BeEmpty(), "expected no memory resources for container %s", containerResource.Name)
				}
			}
		}
	}
}

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Memory Manager", "[LinuxOnly]", framework.WithDisruptive(), framework.WithSerial(), feature.MemoryManager, func() {
	// TODO: add more complex tests that will include interaction between CPUManager, MemoryManager and TopologyManager
	var (
		allNUMANodes             []int
		ctnParams, initCtnParams []memoryManagerCtnAttributes
		is2MiHugepagesSupported  *bool
		isMultiNUMASupported     *bool
		testPod                  *v1.Pod
	)

	f := framework.NewDefaultFramework("memory-manager-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	memoryQuantity := resource.MustParse("1100Mi")
	defaultKubeParams := &memoryManagerKubeletParams{
		systemReservedMemory: []kubeletconfig.MemoryReservation{
			{
				NumaNode: 0,
				Limits: v1.ResourceList{
					resourceMemory: memoryQuantity,
				},
			},
		},
		systemReserved: map[string]string{resourceMemory: "500Mi"},
		kubeReserved:   map[string]string{resourceMemory: "500Mi"},
		evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
	}

	waitingForHugepages := func(ctx context.Context, hugepagesCount int) {
		gomega.Eventually(ctx, func(ctx context.Context) error {
			node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
			if err != nil {
				return err
			}

			capacity, ok := node.Status.Capacity[v1.ResourceName(hugepagesResourceName2Mi)]
			if !ok {
				return fmt.Errorf("the node does not have the resource %s", hugepagesResourceName2Mi)
			}

			size, succeed := capacity.AsInt64()
			if !succeed {
				return fmt.Errorf("failed to convert quantity to int64")
			}

			// 512 Mb, the expected size in bytes
			expectedSize := int64(hugepagesCount * hugepagesSize2M * 1024)
			if size != expectedSize {
				return fmt.Errorf("the actual size %d is different from the expected one %d", size, expectedSize)
			}
			return nil
		}).WithTimeout(time.Minute).WithPolling(framework.Poll).Should(gomega.BeNil())
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		if isMultiNUMASupported == nil {
			isMultiNUMASupported = ptr.To(isMultiNUMA())
		}

		if is2MiHugepagesSupported == nil {
			is2MiHugepagesSupported = ptr.To(isHugePageAvailable(hugepagesSize2M))
		}

		if len(allNUMANodes) == 0 {
			allNUMANodes = getAllNUMANodes()
		}

		// allocate hugepages
		if *is2MiHugepagesSupported {
			ginkgo.By("Configuring hugepages")
			gomega.Eventually(ctx, func() error {
				return configureHugePages(hugepagesSize2M, hugepages2MiCount, ptr.To[int](0))
			}).WithTimeout(30 * time.Second).WithPolling(framework.Poll).Should(gomega.BeNil())
		}
	})

	// dynamically update the kubelet configuration
	ginkgo.JustBeforeEach(func(ctx context.Context) {
		// allocate hugepages
		if *is2MiHugepagesSupported {
			ginkgo.By("Waiting for hugepages resource to become available on the local node")
			waitingForHugepages(ctx, hugepages2MiCount)

			for i := 0; i < len(ctnParams); i++ {
				ctnParams[i].hugepages2Mi = "8Mi"
			}
		}

		if len(ctnParams) > 0 {
			testPod = makeMemoryManagerPod(ctnParams[0].ctnName, initCtnParams, ctnParams)
		}
	})

	ginkgo.JustAfterEach(func(ctx context.Context) {
		// delete the test pod
		if testPod != nil && testPod.Name != "" {
			e2epod.NewPodClient(f).DeleteSync(ctx, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		}

		// release hugepages
		if *is2MiHugepagesSupported {
			ginkgo.By("Releasing allocated hugepages")
			gomega.Eventually(ctx, func() error {
				// configure hugepages on the NUMA node 0 to avoid hugepages split across NUMA nodes
				return configureHugePages(hugepagesSize2M, 0, ptr.To[int](0))
			}).WithTimeout(90*time.Second).WithPolling(15*time.Second).ShouldNot(gomega.HaveOccurred(), "failed to release hugepages")
		}
	})

	ginkgo.Context("with static policy", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			kubeParams := *defaultKubeParams
			kubeParams.policy = staticPolicy
			updateKubeletConfigWithMemoryManagerParams(initialConfig, &kubeParams)
		})

		ginkgo.JustAfterEach(func() {
			// reset containers attributes
			ctnParams = []memoryManagerCtnAttributes{}
			initCtnParams = []memoryManagerCtnAttributes{}
		})

		// TODO: move the test to pod resource API test suite, see - https://github.com/kubernetes/kubernetes/issues/101945
		ginkgo.It("should report memory data during request to pod resources GetAllocatableResources", func(ctx context.Context) {
			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
			framework.ExpectNoError(err)
			defer conn.Close() //nolint:errcheck

			resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
			framework.ExpectNoError(err)
			gomega.Expect(resp.Memory).ToNot(gomega.BeEmpty())

			stateData, err := getMemoryManagerState()
			framework.ExpectNoError(err)

			stateAllocatableMemory := getAllocatableMemoryFromStateFile(stateData)
			gomega.Expect(resp.Memory).To(gomega.HaveLen(len(stateAllocatableMemory)))

			for _, containerMemory := range resp.Memory {
				gomega.Expect(containerMemory.Topology).NotTo(gomega.BeNil())
				gomega.Expect(containerMemory.Topology.Nodes).To(gomega.HaveLen(1))
				gomega.Expect(containerMemory.Topology.Nodes[0]).NotTo(gomega.BeNil())

				numaNodeID := int(containerMemory.Topology.Nodes[0].ID)
				for _, numaStateMemory := range stateAllocatableMemory {
					gomega.Expect(numaStateMemory.NUMAAffinity).To(gomega.HaveLen(1))
					if numaNodeID != numaStateMemory.NUMAAffinity[0] {
						continue
					}

					if containerMemory.MemoryType != string(numaStateMemory.Type) {
						continue
					}

					gomega.Expect(containerMemory.Size).To(gomega.BeEquivalentTo(numaStateMemory.Size))
				}
			}

			gomega.Expect(resp.Memory).ToNot(gomega.BeEmpty())
		})

		ginkgo.When("guaranteed pod has init and app containers", func() {
			ginkgo.BeforeEach(func() {
				// override containers parameters
				ctnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "memory-manager-static",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
				// override init container parameters
				initCtnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "init-memory-manager-static",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
			})

			ginkgo.It("should succeed to start the pod", func(ctx context.Context) {
				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})
		})

		ginkgo.When("guaranteed pod has only app containers", func() {
			ginkgo.BeforeEach(func() {
				// override containers parameters
				ctnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "memory-manager-static",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
			})

			ginkgo.It("should succeed to start the pod", func(ctx context.Context) {
				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})
		})

		ginkgo.When("multiple guaranteed pods started", func() {
			var testPod2 *v1.Pod

			ginkgo.BeforeEach(func() {
				// override containers parameters
				ctnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "memory-manager-static",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
			})

			ginkgo.JustBeforeEach(func() {
				testPod2 = makeMemoryManagerPod("memory-manager-static", initCtnParams, ctnParams)
			})

			ginkgo.It("should succeed to start all pods", func(ctx context.Context) {
				ginkgo.By("Running the test pod and the test pod 2")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				ginkgo.By("Running the test pod 2")
				testPod2 = e2epod.NewPodClient(f).CreateSync(ctx, testPod2)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
				verifyMemoryPinning(f, ctx, testPod2, []int{0})
			})

			// TODO: move the test to pod resource API test suite, see - https://github.com/kubernetes/kubernetes/issues/101945
			ginkgo.It("should report memory data for each guaranteed pod and container during request to pod resources List", func(ctx context.Context) {
				ginkgo.By("Running the test pod and the test pod 2")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				ginkgo.By("Running the test pod 2")
				testPod2 = e2epod.NewPodClient(f).CreateSync(ctx, testPod2)

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)
				defer conn.Close() //nolint:errcheck

				resp, err := cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
				framework.ExpectNoError(err)

				for _, pod := range []*v1.Pod{testPod, testPod2} {
					for _, podResource := range resp.PodResources {
						if podResource.Name != pod.Name {
							continue
						}

						for _, c := range pod.Spec.Containers {
							for _, containerResource := range podResource.Containers {
								if containerResource.Name != c.Name {
									continue
								}

								for _, containerMemory := range containerResource.Memory {
									q := c.Resources.Limits[v1.ResourceName(containerMemory.MemoryType)]
									value, ok := q.AsInt64()
									gomega.Expect(ok).To(gomega.BeTrueBecause("cannot convert value to integer"))
									gomega.Expect(value).To(gomega.BeEquivalentTo(containerMemory.Size))
								}
							}
						}
					}
				}
			})

			ginkgo.JustAfterEach(func(ctx context.Context) {
				// delete the test pod 2
				if testPod2.Name != "" {
					e2epod.NewPodClient(f).DeleteSync(ctx, testPod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				}
			})
		})

		// the test requires at least two NUMA nodes
		// test on each NUMA node will start the pod that will consume almost all memory of the NUMA node except 256Mi
		// after it will start an additional pod with the memory request that can not be satisfied by the single NUMA node
		// free memory
		ginkgo.When("guaranteed pod memory request is bigger than free memory on each NUMA node", func() {
			var workloadPods []*v1.Pod

			ginkgo.BeforeEach(func() {
				if !*isMultiNUMASupported {
					ginkgo.Skip("The machines has less than two NUMA nodes")
				}

				ctnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "memory-manager-static",
						cpus:    "100m",
						memory:  "384Mi",
					},
				}
			})

			ginkgo.JustBeforeEach(func(ctx context.Context) {
				stateData, err := getMemoryManagerState()
				framework.ExpectNoError(err)

				for _, memoryState := range stateData.MachineState {
					// consume all memory except of 256Mi on each NUMA node via workload pods
					workloadPodMemory := memoryState.MemoryMap[v1.ResourceMemory].Free - 256*1024*1024
					memoryQuantity := resource.NewQuantity(int64(workloadPodMemory), resource.BinarySI)
					workloadCtnAttrs := []memoryManagerCtnAttributes{
						{
							ctnName: "workload-pod",
							cpus:    "100m",
							memory:  memoryQuantity.String(),
						},
					}
					workloadPod := makeMemoryManagerPod(workloadCtnAttrs[0].ctnName, initCtnParams, workloadCtnAttrs)

					workloadPod = e2epod.NewPodClient(f).CreateSync(ctx, workloadPod)
					workloadPods = append(workloadPods, workloadPod)
				}
			})

			ginkgo.It("should be rejected", func(ctx context.Context) {
				ginkgo.By("Creating the pod")
				testPod = e2epod.NewPodClient(f).Create(ctx, testPod)

				ginkgo.By("Checking that pod failed to start because of admission error")
				gomega.Eventually(ctx, func() bool {
					tmpPod, err := e2epod.NewPodClient(f).Get(ctx, testPod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)

					if tmpPod.Status.Phase != v1.PodFailed {
						return false
					}

					if tmpPod.Status.Reason != "UnexpectedAdmissionError" {
						return false
					}

					if !strings.Contains(tmpPod.Status.Message, "Allocate failed due to [memorymanager]") {
						return false
					}

					return true
				}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(
					gomega.BeTrueBecause(
						"the pod succeeded to start, when it should fail with the admission error",
					))
			})

			ginkgo.JustAfterEach(func(ctx context.Context) {
				for _, workloadPod := range workloadPods {
					if workloadPod.Name != "" {
						e2epod.NewPodClient(f).DeleteSync(ctx, workloadPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
					}
				}
			})
		})
	})

	ginkgo.Context("with none policy", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			kubeParams := *defaultKubeParams
			kubeParams.policy = nonePolicy
			updateKubeletConfigWithMemoryManagerParams(initialConfig, &kubeParams)
		})

		// empty context to configure same container parameters for all tests
		ginkgo.Context("", func() {
			ginkgo.BeforeEach(func() {
				// override pod parameters
				ctnParams = []memoryManagerCtnAttributes{
					{
						ctnName: "memory-manager-none",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
			})

			// TODO: move the test to pod resource API test suite, see - https://github.com/kubernetes/kubernetes/issues/101945
			ginkgo.It("should not report any memory data during request to pod resources GetAllocatableResources", func(ctx context.Context) {
				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)
				defer conn.Close() //nolint:errcheck

				resp, err := cli.GetAllocatableResources(ctx, &kubeletpodresourcesv1.AllocatableResourcesRequest{})
				framework.ExpectNoError(err)

				gomega.Expect(resp.Memory).To(gomega.BeEmpty())
			})

			// TODO: move the test to pod resource API test suite, see - https://github.com/kubernetes/kubernetes/issues/101945
			ginkgo.It("should not report any memory data during request to pod resources List", func(ctx context.Context) {
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
				framework.ExpectNoError(err)

				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				framework.ExpectNoError(err)
				defer conn.Close() //nolint:errcheck

				resp, err := cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})
				framework.ExpectNoError(err)

				for _, podResource := range resp.PodResources {
					if podResource.Name != testPod.Name {
						continue
					}

					for _, containerResource := range podResource.Containers {
						gomega.Expect(containerResource.Memory).To(gomega.BeEmpty())
					}
				}
			})

			ginkgo.It("should succeed to start the pod", func(ctx context.Context) {
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})
		})
	})
})

type memoryManagerKubeletArguments struct {
	policyName                     string
	topologyManagerPolicy          string
	topologyManagerScope           string
	enablePodLevelResources        bool
	enablePodLevelResourceManagers bool
}

func configureMemoryManagerInKubelet(oldCfg *kubeletconfig.KubeletConfiguration, kubeletArguments *memoryManagerKubeletArguments) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()
	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = make(map[string]bool)
	}

	newCfg.FeatureGates["PodLevelResources"] = kubeletArguments.enablePodLevelResources
	newCfg.FeatureGates["PodLevelResourceManagers"] = kubeletArguments.enablePodLevelResourceManagers

	newCfg.MemoryManagerPolicy = kubeletArguments.policyName
	newCfg.TopologyManagerPolicy = kubeletArguments.topologyManagerPolicy
	newCfg.TopologyManagerScope = kubeletArguments.topologyManagerScope

	return newCfg
}

var _ = SIGDescribe("Memory Manager Pod Level Resources", ginkgo.Ordered, ginkgo.ContinueOnFailure, framework.WithDisruptive(), framework.WithSerial(), feature.MemoryManager, feature.PodLevelResources, feature.PodLevelResourceManagers, framework.WithFeatureGate(features.PodLevelResources), framework.WithFeatureGate(features.PodLevelResourceManagers), func() {
	f := framework.NewDefaultFramework("memory-manager-pod-level-resources")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	var (
		allNUMANodes         []int
		isMultiNUMASupported *bool
		oldCfg               *kubeletconfig.KubeletConfiguration
	)

	memoryQuantity := resource.MustParse("1100Mi")
	defaultKubeParams := &memoryManagerKubeletParams{
		systemReservedMemory: []kubeletconfig.MemoryReservation{
			{
				NumaNode: 0,
				Limits: v1.ResourceList{
					resourceMemory: memoryQuantity,
				},
			},
		},
		systemReserved: map[string]string{resourceMemory: "500Mi"},
		kubeReserved:   map[string]string{resourceMemory: "500Mi"},
		evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
	}

	ginkgo.BeforeAll(func(ctx context.Context) {
		var err error
		oldCfg, err = getCurrentKubeletConfig(ctx)
		framework.ExpectNoError(err)

		if isMultiNUMASupported == nil {
			isMultiNUMASupported = ptr.To(isMultiNUMA())
		}

		if len(allNUMANodes) == 0 {
			allNUMANodes = getAllNUMANodes()
		}
	})

	ginkgo.AfterAll(func(ctx context.Context) {
		updateKubeletConfig(ctx, f, oldCfg, true)
	})

	ginkgo.Context("with static policy", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			kubeParams := *defaultKubeParams
			kubeParams.policy = staticPolicy
			updateKubeletConfigWithMemoryManagerParams(initialConfig, &kubeParams)
			if initialConfig.FeatureGates == nil {
				initialConfig.FeatureGates = make(map[string]bool)
			}
		})

		ginkgo.Context("when the topology manager scope is 'pod'", func() {
			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and guaranteed container, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-gu-ctn", nil, ctnParams)
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)

				verifyMemoryManagerAllocations(ctx, testPod, []string{
					"gu-container",
				}, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				testPod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "gu-pod-level-ngu-ctn-",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Name:    "ngu-container",
								Image:   busyboxImage,
								Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
								},
							},
						},
					},
				}
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, nil, 128*1024*1024)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and mix of guaranteed and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-mix-ctn", nil, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, []string{"gu-container"}, 64*1024*1024)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and mix of guaranteed and non-guaranteed standard and init containers, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				initCtnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-init-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-mix-init-ctn", initCtnParams, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, []string{"gu-init-container", "gu-container"}, 64*1024*1024)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should not allocate exclusive memory to a non-guaranteed pod with pod-level resources and guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("ngu-pod-level-gu-ctn", nil, ctnParams)
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("64Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, nil, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})

			ginkgo.It("should not allocate exclusive memory to a non-guaranteed pod with pod-level resources and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				testPod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "ngu-pod-level-ngu-ctn-",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Name:    "ngu-container",
								Image:   busyboxImage,
								Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
								},
							},
						},
					},
				}
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("64Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, nil, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})

			ginkgo.It("should reject a pod that would result in an empty pod shared pool, topologymanager.PolicyBestEffort, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyBestEffort,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container-1",
						cpus:    "100m",
						memory:  "64Mi",
					},
					{
						ctnName: "gu-container-2",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-empty-shared", nil, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Creating the pod")
				testPod = e2epod.NewPodClient(f).Create(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("Checking that pod failed to start because of admission error")
				gomega.Eventually(ctx, func(g gomega.Gomega) {
					tmpPod, err := e2epod.NewPodClient(f).Get(ctx, testPod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)

					g.Expect(tmpPod.Status.Phase).To(gomega.Equal(v1.PodFailed))
					g.Expect(tmpPod.Status.Message).To(gomega.ContainSubstring("sum of exclusive container memory requests equals pod budget"))
				}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
			})

			ginkgo.It("should reject a pod that would result in an empty pod shared pool, topologymanager.PolicyRestricted, PodLevelResourceManagers enabled", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container-1",
						cpus:    "100m",
						memory:  "64Mi",
					},
					{
						ctnName: "gu-container-2",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-empty-shared", nil, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Creating the pod")
				testPod = e2epod.NewPodClient(f).Create(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("Checking that pod failed to start because of admission error")
				gomega.Eventually(ctx, func(g gomega.Gomega) {
					tmpPod, err := e2epod.NewPodClient(f).Get(ctx, testPod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)

					g.Expect(tmpPod.Status.Phase).To(gomega.Equal(v1.PodFailed))
					g.Expect(tmpPod.Status.Message).To(gomega.ContainSubstring("pod with pod-level resources failed admission under pod-scope topology manager"))
				}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
			})

			// This test demonstrates that memory resources are correctly released and re-allocated.
			// It runs two pods sequentially that each request a significant amount of memory from a single NUMA node.
			// It verifies that the second pod is allocated the same NUMA node(s) as the first,
			// proving that the memory manager freed the resources after the first pod completed.
			ginkgo.It("should release and re-allocate memory correctly for sequential guaranteed pods with guaranteed containers and empty shared pool", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				stateData, err := getMemoryManagerState()
				framework.ExpectNoError(err)

				// Find the first NUMA node with enough allocatable memory to run the test.
				var memRequest string
				foundNuma := false
				for _, numaState := range stateData.MachineState {
					memToRequest := numaState.MemoryMap[v1.ResourceMemory].Allocatable - (1 * 1024 * 1024) // leave 1Mi buffer
					memRequest = fmt.Sprintf("%d", memToRequest)
					foundNuma = true
					break
				}
				if !foundNuma {
					ginkgo.Skip("Skipping test: no NUMA node found with sufficient allocatable memory")
				}
				framework.Logf("Node has sufficient memory. Requesting %s for pods.", memRequest)

				podLevelResources := &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse(memRequest),
					},
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse(memRequest),
					},
				}

				// A command that prints the allowed NUMA nodes and then exits.
				command := []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 2"}

				makeTestPod := func(name string) *v1.Pod {
					return &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: name,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Name:    "gu-container",
									Image:   busyboxImage,
									Command: command,
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse(memRequest),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse(memRequest),
										},
									},
								},
							},
							Resources: podLevelResources,
						},
					}
				}

				pod1 := makeTestPod("gu-pod-sequential-1")
				podClient := e2epod.NewPodClient(f)

				ginkgo.By("creating the first test pod")
				pod1 = podClient.CreateSync(ctx, pod1)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("getting NUMA allocation for the first pod")
				logs1, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod1.Namespace, pod1.Name, "gu-container")
				framework.ExpectNoError(err, "failed to get logs for the first pod")
				numaNodes1, err := cpuset.Parse(strings.TrimSpace(logs1))
				framework.ExpectNoError(err, "failed to parse NUMA nodes from logs for the first pod")
				framework.Logf("Pod 1 allocated NUMA nodes: %s", numaNodes1.String())

				ginkgo.By("waiting for the first pod to succeed")
				err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod1.Name, f.Namespace.Name)
				framework.ExpectNoError(err, "first pod failed to complete")

				pod2 := makeTestPod("gu-pod-sequential-2")

				ginkgo.By("creating the second test pod")
				pod2 = podClient.CreateSync(ctx, pod2)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("getting NUMA allocation for the second pod")
				logs2, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod2.Namespace, pod2.Name, "gu-container")
				framework.ExpectNoError(err, "failed to get logs for the second pod")
				numaNodes2, err := cpuset.Parse(strings.TrimSpace(logs2))
				framework.ExpectNoError(err, "failed to parse NUMA nodes from logs for the second pod")
				framework.Logf("Pod 2 allocated NUMA nodes: %s", numaNodes2.String())

				ginkgo.By("verifying the second pod reused the NUMA nodes from the first pod")
				gomega.Expect(numaNodes2).To(gomega.Equal(numaNodes1), "The NUMA set for the second pod should be the same as the first pod's.")
			})

			// This test demonstrates that memory is correctly released from the pod shared
			// pool, even when no containers are linked to it. This test demonstrates that
			// the remove container logic cleans resources properly using the state.
			ginkgo.It("should release resources from the pod shared pool when no containers used it", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				stateData, err := getMemoryManagerState()
				framework.ExpectNoError(err)

				// Find the first NUMA node with enough allocatable memory to run the test.
				var memRequest string
				foundNuma := false
				for _, numaState := range stateData.MachineState {
					memToRequest := numaState.MemoryMap[v1.ResourceMemory].Allocatable - (1 * 1024 * 1024) // leave 1Mi buffer
					memRequest = fmt.Sprintf("%d", memToRequest)
					foundNuma = true
					break
				}
				if !foundNuma {
					ginkgo.Skip("Skipping test: no NUMA node found with sufficient allocatable memory")
				}
				framework.Logf("Node has sufficient memory. Requesting %s for pods.", memRequest)

				podLevelResources := &v1.ResourceRequirements{
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse(memRequest),
					},
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse(memRequest),
					},
				}

				// A command that prints the allowed NUMA nodes and then exits.
				command := []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 2"}

				makeTestPod := func(name string) *v1.Pod {
					return &v1.Pod{
						ObjectMeta: metav1.ObjectMeta{
							Name: name,
						},
						Spec: v1.PodSpec{
							RestartPolicy: v1.RestartPolicyNever,
							Containers: []v1.Container{
								{
									Name:    "gu-container",
									Image:   busyboxImage,
									Command: command,
									Resources: v1.ResourceRequirements{
										Requests: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
										Limits: v1.ResourceList{
											v1.ResourceCPU:    resource.MustParse("100m"),
											v1.ResourceMemory: resource.MustParse("128Mi"),
										},
									},
								},
							},
							Resources: podLevelResources,
						},
					}
				}
				pod1 := makeTestPod("gu-pod-sequential-1")
				podClient := e2epod.NewPodClient(f)

				ginkgo.By("creating the first test pod")
				pod1 = podClient.CreateSync(ctx, pod1)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("getting NUMA allocation for the first pod")
				logs1, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod1.Namespace, pod1.Name, "gu-container")
				framework.ExpectNoError(err, "failed to get logs for the first pod")
				numaNodes1, err := cpuset.Parse(strings.TrimSpace(logs1))
				framework.ExpectNoError(err, "failed to parse NUMA nodes from logs for the first pod")
				framework.Logf("Pod 1 allocated NUMA nodes: %s", numaNodes1.String())

				ginkgo.By("waiting for the first pod to succeed")
				err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod1.Name, f.Namespace.Name)
				framework.ExpectNoError(err, "first pod failed to complete")

				pod2 := makeTestPod("gu-pod-sequential-2")

				ginkgo.By("creating the second test pod")
				pod2 = podClient.CreateSync(ctx, pod2)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("getting NUMA allocation for the second pod")
				logs2, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod2.Namespace, pod2.Name, "gu-container")
				framework.ExpectNoError(err, "failed to get logs for the second pod")
				numaNodes2, err := cpuset.Parse(strings.TrimSpace(logs2))
				framework.ExpectNoError(err, "failed to parse NUMA nodes from logs for the second pod")
				framework.Logf("Pod 2 allocated NUMA nodes: %s", numaNodes2.String())

				ginkgo.By("waiting for the second pod to succeed")
				err = e2epod.WaitForPodSuccessInNamespace(ctx, f.ClientSet, pod2.Name, f.Namespace.Name)
				framework.ExpectNoError(err, "second pod failed to complete")
			})

			// This test verifies that the pod shared pool memory is not removed until all containers
			// in the pod have finished. It creates a pod with two containers sharing the pool,
			// allows one to terminate, and then ensures that the memory remains reserved in the state.
			ginkgo.It("should not release pod shared pool memory while at least one container is using it", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				stateData, err := getMemoryManagerState()
				framework.ExpectNoError(err)

				// Find a NUMA node with sufficient memory.
				var memRequestString string
				var memRequestValue int64
				foundNuma := false
				for _, numaState := range stateData.MachineState {
					allocatable := numaState.MemoryMap[v1.ResourceMemory].Allocatable
					if allocatable > 256*1024*1024 {
						// Request half the allocatable memory to be safe, but substantial enough to verify reservation.
						memRequestValue = int64(allocatable / 2)
						memRequestString = fmt.Sprintf("%d", memRequestValue)
						foundNuma = true
						break
					}
				}
				if !foundNuma {
					ginkgo.Skip("Skipping test: no NUMA node found with sufficient allocatable memory")
				}

				podA := makeMemoryManagerPod("pod-a-shared-pool", nil, []memoryManagerCtnAttributes{
					{
						ctnName: "short-container",
						cpus:    "50m",
					},
					{
						ctnName: "long-container",
						cpus:    "50m",
					},
				})
				podA.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse(memRequestString),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse(memRequestString),
					},
				}
				podA.Spec.Containers[0].Command = []string{"sh", "-c", "sleep 5"}
				podA.Spec.RestartPolicy = v1.RestartPolicyNever

				ginkgo.By("creating Pod A with shared containers")
				podA = e2epod.NewPodClient(f).CreateSync(ctx, podA)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, podA.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("waiting for long-container to be running")
				framework.ExpectNoError(e2epod.WaitForContainerRunning(ctx, f.ClientSet, podA.Namespace, podA.Name, "long-container", 2*time.Minute))

				ginkgo.By("waiting for short-container to terminate")
				framework.ExpectNoError(e2epod.WaitForContainerTerminated(ctx, f.ClientSet, podA.Namespace, podA.Name, "short-container", 2*time.Minute))

				// Identify the NUMA node used by Pod A
				logs, err := e2epod.GetPodLogs(ctx, f.ClientSet, podA.Namespace, podA.Name, "long-container")
				framework.ExpectNoError(err, "failed to get logs for long-container")
				numaNodes, err := cpuset.Parse(strings.TrimSpace(logs))
				framework.ExpectNoError(err, "failed to parse NUMA nodes from logs")
				framework.Logf("Pod A allocated NUMA nodes: %s", numaNodes.String())

				ginkgo.By("verifying memory is still reserved in Memory Manager state")
				stateAfterExit, err := getMemoryManagerState()
				framework.ExpectNoError(err)

				for _, nodeID := range numaNodes.List() {
					nodeState := stateAfterExit.MachineState[nodeID]
					reserved := nodeState.MemoryMap[v1.ResourceMemory].Reserved
					// We verify that Reserved memory is at least what we requested, the pod shared pool should not be affected.
					gomega.Expect(int64(reserved)).To(gomega.BeNumerically(">=", memRequestValue), "Reserved memory on node %d should be at least %d", nodeID, memRequestValue)
				}
			})

			// This test verifies that resource accounting is performed at the pod level for memory.
			// It creates two pods, each requesting half of the node's allocatable memory at the pod level
			// but containing three non-guaranteed containers. If accounting were incorrect
			// (e.g., summing container resources), the pods might be rejected.
			// The test ensures both pods are admitted and run concurrently.
			ginkgo.It("should admit multiple pods with pod-level resources and many non-guaranteed containers", ginkgo.Label("pod-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.PodTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				node, err := f.ClientSet.CoreV1().Nodes().Get(ctx, framework.TestContext.NodeName, metav1.GetOptions{})
				framework.ExpectNoError(err)
				allocatableMem := node.Status.Allocatable.Memory().Value()
				memRequestPerPodVal := allocatableMem / 2
				framework.Logf("Node has %d allocatable memory. Requesting %d for each of the 2 pods.", allocatableMem, memRequestPerPodVal)

				memRequestPerPod := fmt.Sprintf("%d", memRequestPerPodVal)

				makeTestPod := func(name string) *v1.Pod {
					pod := makeMemoryManagerPod(name, nil, []memoryManagerCtnAttributes{
						{ctnName: "ngu-container-1", cpus: "50m"},
						{ctnName: "ngu-container-2", cpus: "50m"},
						{ctnName: "ngu-container-3", cpus: "50m"},
					})
					pod.Spec.Resources = &v1.ResourceRequirements{
						Requests: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("200m"),
							v1.ResourceMemory: resource.MustParse(memRequestPerPod),
						},
						Limits: v1.ResourceList{
							v1.ResourceCPU:    resource.MustParse("200m"),
							v1.ResourceMemory: resource.MustParse(memRequestPerPod),
						},
					}
					return pod
				}

				pod1 := makeTestPod("concurrent-pod-1")
				pod2 := makeTestPod("concurrent-pod-2")
				podClient := e2epod.NewPodClient(f)

				ginkgo.By("creating two concurrent pods")
				pod1 = podClient.CreateSync(ctx, pod1)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod1.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
				pod2 = podClient.CreateSync(ctx, pod2)
				ginkgo.DeferCleanup(podClient.DeleteSync, pod2.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				ginkgo.By("verifying NUMA allocation for both pods")
				logs1, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod1.Namespace, pod1.Name, "ngu-container-1")
				framework.ExpectNoError(err, "failed to get logs for pod1")
				numaNodes1, err := cpuset.Parse(strings.TrimSpace(logs1))
				framework.ExpectNoError(err, "failed to parse NUMA nodes for pod1")

				logs2, err := e2epod.GetPodLogs(ctx, f.ClientSet, pod2.Namespace, pod2.Name, "ngu-container-1")
				framework.ExpectNoError(err, "failed to get logs for pod2")
				numaNodes2, err := cpuset.Parse(strings.TrimSpace(logs2))
				framework.ExpectNoError(err, "failed to parse NUMA nodes for pod2")

				ginkgo.By("verifying NUMA sets do not overlap if on different nodes")
				// This is a soft check. If the machine has enough NUMA nodes, they should be different.
				// If not, they might be the same, which is also valid. The main point is that both were admitted.
				if numaNodes1.String() != numaNodes2.String() {
					framework.Logf("Pods are on different NUMA nodes, verifying no overlap.")
					gomega.Expect(numaNodes1.Intersection(numaNodes2).IsEmpty()).To(gomega.BeTrueBecause("NUMA sets of the two pods should not overlap"))
				} else {
					framework.Logf("Pods are on the same NUMA node (%s). This is acceptable.", numaNodes1.String())
				}
			})
		})

		ginkgo.Context("when the topology manager scope is 'container'", func() {
			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and guaranteed container, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "128Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-gu-ctn-ctn-scope", nil, ctnParams)
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, []string{"gu-container"}, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should not allocate exclusive memory to a guaranteed pod with pod-level resources and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				testPod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "gu-pod-level-ngu-ctn-ctn-scope-",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Name:    "ngu-container",
								Image:   busyboxImage,
								Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
								},
							},
						},
					},
				}
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, nil, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})

			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and mix of guaranteed and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-mix-ctn-ctn-scope", nil, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("200m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, []string{"gu-container"}, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should allocate exclusive memory to a guaranteed pod with pod-level resources and mix of guaranteed and non-guaranteed standard and init containers, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				initCtnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-init-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-mix-init-ctn-ctn-scope", initCtnParams, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, []string{"gu-init-container", "gu-container"}, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})

			ginkgo.It("should not allocate exclusive memory to a non-guaranteed pod with pod-level resources and guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("ngu-pod-level-gu-ctn-ctn-scope", nil, ctnParams)
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("64Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})

			ginkgo.It("should not allocate exclusive memory to a non-guaranteed pod with pod-level resources and non-guaranteed containers, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				testPod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						GenerateName: "ngu-pod-level-ngu-ctn-ctn-scope-",
					},
					Spec: v1.PodSpec{
						RestartPolicy: v1.RestartPolicyNever,
						Containers: []v1.Container{
							{
								Name:    "ngu-container",
								Image:   busyboxImage,
								Command: []string{"sh", "-c", "grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d"},
								Resources: v1.ResourceRequirements{
									Limits: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
									Requests: v1.ResourceList{
										v1.ResourceCPU: resource.MustParse("100m"),
									},
								},
							},
						},
					},
				}
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("100m"),
						v1.ResourceMemory: resource.MustParse("64Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				verifyMemoryManagerAllocations(ctx, testPod, nil, 0)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
			})

			ginkgo.It("should not reject a pod that would result in an empty pod shared pool, no pod shared pool in container scope, PodLevelResourceManagers enabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
				currentCfg, err := getCurrentKubeletConfig(ctx)
				framework.ExpectNoError(err)
				updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
					policyName:                     string(staticPolicy),
					topologyManagerPolicy:          topologymanager.PolicyRestricted,
					topologyManagerScope:           topologymanager.ContainerTopologyScope,
					enablePodLevelResources:        true,
					enablePodLevelResourceManagers: true,
				}))

				ctnParams := []memoryManagerCtnAttributes{
					{
						ctnName: "gu-container-1",
						cpus:    "100m",
						memory:  "64Mi",
					},
					{
						ctnName: "gu-container-2",
						cpus:    "100m",
						memory:  "64Mi",
					},
				}
				testPod := makeMemoryManagerPod("gu-pod-level-empty-shared-ctn-scope", nil, ctnParams)
				testPod.Spec.Containers = append(testPod.Spec.Containers, v1.Container{
					Name:    "ngu-container",
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "sleep 1d"},
					Resources: v1.ResourceRequirements{
						Limits: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
						Requests: v1.ResourceList{
							v1.ResourceCPU: resource.MustParse("100m"),
						},
					},
				})
				testPod.Spec.Resources = &v1.ResourceRequirements{
					Limits: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
					Requests: v1.ResourceList{
						v1.ResourceCPU:    resource.MustParse("300m"),
						v1.ResourceMemory: resource.MustParse("128Mi"),
					},
				}

				ginkgo.By("Running the test pod")
				testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
				ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

				if !*isMultiNUMASupported {
					framework.Logf("Skipping memory pinning verification on single-NUMA machine")
					return
				}
				verifyMemoryPinning(f, ctx, testPod, []int{0})
			})
		})

		ginkgo.It("should not report any memory data during request and run on the shared memory pool for a pod with pod-level resources when PodLevelResourceManagers is disabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
			currentCfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
			updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
				policyName:                     string(staticPolicy),
				topologyManagerPolicy:          topologymanager.PolicyRestricted,
				topologyManagerScope:           topologymanager.ContainerTopologyScope,
				enablePodLevelResources:        true,
				enablePodLevelResourceManagers: false,
			}))

			ctnParams := []memoryManagerCtnAttributes{
				{
					ctnName: "memory-manager-none",
					cpus:    "100m",
					memory:  "128Mi",
				},
			}
			testPod := makeMemoryManagerPod(ctnParams[0].ctnName, nil, ctnParams)
			testPod.Spec.Resources = &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("128Mi"),
				},
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("128Mi"),
				},
			}
			testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			endpoint, err := util.LocalEndpoint(defaultPodResourcesPath, podresources.Socket)
			framework.ExpectNoError(err)

			var resp *kubeletpodresourcesv1.ListPodResourcesResponse
			gomega.Eventually(ctx, func(ctx context.Context) error {
				cli, conn, err := podresources.GetV1Client(endpoint, defaultPodResourcesTimeout, defaultPodResourcesMaxSize)
				if err != nil {
					return err
				}
				defer conn.Close() //nolint:errcheck
				resp, err = cli.List(ctx, &kubeletpodresourcesv1.ListPodResourcesRequest{})

				return err
			}).WithTimeout(time.Minute).WithPolling(5 * time.Second).Should(gomega.Succeed())
			for _, podResource := range resp.PodResources {
				if podResource.Name != testPod.Name {
					continue
				}

				for _, containerResource := range podResource.Containers {
					gomega.Expect(containerResource.Memory).To(gomega.BeEmpty())
				}
			}
		})

		ginkgo.It("should run on the shared memory pool for a pod with pod-level resources when PodLevelResourceManagers is disabled", ginkgo.Label("container-scope"), func(ctx context.Context) {
			currentCfg, err := getCurrentKubeletConfig(ctx)
			framework.ExpectNoError(err)
			updateKubeletConfigIfNeeded(ctx, f, configureMemoryManagerInKubelet(currentCfg, &memoryManagerKubeletArguments{
				policyName:                     string(staticPolicy),
				topologyManagerPolicy:          topologymanager.PolicyRestricted,
				topologyManagerScope:           topologymanager.ContainerTopologyScope,
				enablePodLevelResources:        true,
				enablePodLevelResourceManagers: false,
			}))

			ctnParams := []memoryManagerCtnAttributes{
				{
					ctnName: "memory-manager-none",
					cpus:    "100m",
					memory:  "128Mi",
				},
			}
			testPod := makeMemoryManagerPod(ctnParams[0].ctnName, nil, ctnParams)
			testPod.Spec.Resources = &v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("128Mi"),
				},
				Requests: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse("100m"),
					v1.ResourceMemory: resource.MustParse("128Mi"),
				},
			}
			testPod = e2epod.NewPodClient(f).CreateSync(ctx, testPod)
			ginkgo.DeferCleanup(e2epod.NewPodClient(f).DeleteSync, testPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)

			if !*isMultiNUMASupported {
				framework.Logf("Skipping memory pinning verification on single-NUMA machine")
				return
			}
			verifyMemoryPinning(f, ctx, testPod, allNUMANodes)
		})
	})
})

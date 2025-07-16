//go:build linux
// +build linux

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
	kubeletpodresourcesv1 "k8s.io/kubelet/pkg/apis/podresources/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/apis/podresources"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/cpuset"
	"k8s.io/utils/pointer"

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
		ctn := v1.Container{
			Name:  ctnAttr.ctnName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpus),
					v1.ResourceMemory: resource.MustParse(ctnAttr.memory),
				},
			},
			Command: []string{"sh", "-c", ctnCmd},
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

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("Memory Manager", framework.WithDisruptive(), framework.WithSerial(), feature.MemoryManager, func() {
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

	verifyMemoryPinning := func(ctx context.Context, pod *v1.Pod, numaNodeIDs []int) {
		ginkgo.By("Verifying the NUMA pinning")

		output, err := e2epod.GetPodLogs(ctx, f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		currentNUMANodeIDs, err := cpuset.Parse(strings.Trim(output, "\n"))
		framework.ExpectNoError(err)

		gomega.Expect(numaNodeIDs).To(gomega.Equal(currentNUMANodeIDs.List()))
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
		}, time.Minute, framework.Poll).Should(gomega.BeNil())
	}

	ginkgo.BeforeEach(func(ctx context.Context) {
		if isMultiNUMASupported == nil {
			isMultiNUMASupported = pointer.BoolPtr(isMultiNUMA())
		}

		if is2MiHugepagesSupported == nil {
			is2MiHugepagesSupported = pointer.BoolPtr(isHugePageAvailable(hugepagesSize2M))
		}

		if len(allNUMANodes) == 0 {
			allNUMANodes = getAllNUMANodes()
		}

		// allocate hugepages
		if *is2MiHugepagesSupported {
			ginkgo.By("Configuring hugepages")
			gomega.Eventually(ctx, func() error {
				return configureHugePages(hugepagesSize2M, hugepages2MiCount, pointer.IntPtr(0))
			}, 30*time.Second, framework.Poll).Should(gomega.BeNil())
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
				return configureHugePages(hugepagesSize2M, 0, pointer.IntPtr(0))
			}, 90*time.Second, 15*time.Second).ShouldNot(gomega.HaveOccurred(), "failed to release hugepages")
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
			defer conn.Close()

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

					gomega.Expect(containerMemory.Size_).To(gomega.BeEquivalentTo(numaStateMemory.Size))
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
					return
				}

				verifyMemoryPinning(ctx, testPod, []int{0})
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
					return
				}

				verifyMemoryPinning(ctx, testPod, []int{0})
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
					return
				}

				verifyMemoryPinning(ctx, testPod, []int{0})
				verifyMemoryPinning(ctx, testPod2, []int{0})
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
				defer conn.Close()

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
									gomega.Expect(value).To(gomega.BeEquivalentTo(containerMemory.Size_))
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
				}, time.Minute, 5*time.Second).Should(
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
				defer conn.Close()

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
				defer conn.Close()

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
					return
				}

				verifyMemoryPinning(ctx, testPod, allNUMANodes)
			})
		})
	})
})

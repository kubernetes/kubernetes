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
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager/state"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	"k8s.io/utils/pointer"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
)

const (
	evictionHardMemory     = "memory.available"
	memoryManagerStateFile = "/var/lib/kubelet/memory_manager_state"
	resourceMemory         = "memory"
	staticPolicy           = "Static"
	nonePolicy             = "None"
)

// Helper for makeMemoryManagerPod().
type memoryManagerCtnAttributes struct {
	ctnName      string
	cpus         string
	memory       string
	hugepages2Mi string
}

//  makeMemoryManagerContainers returns slice of containers with provided attributes and indicator of hugepages mount needed for those.
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

func deleteMemoryManagerStateFile() {
	err := exec.Command("/bin/sh", "-c", fmt.Sprintf("rm -f %s", memoryManagerStateFile)).Run()
	framework.ExpectNoError(err, "failed to delete the state file")
}

func getMemoryManagerState() (*state.MemoryManagerCheckpoint, error) {
	if _, err := os.Stat(memoryManagerStateFile); os.IsNotExist(err) {
		return nil, fmt.Errorf("the memory manager state file %s does not exist", memoryManagerStateFile)
	}

	out, err := exec.Command("/bin/sh", "-c", fmt.Sprintf("cat %s", memoryManagerStateFile)).Output()
	if err != nil {
		return nil, fmt.Errorf("failed to run command 'cat %s': out: %s, err: %v", memoryManagerStateFile, out, err)
	}

	memoryManagerCheckpoint := &state.MemoryManagerCheckpoint{}
	if err := json.Unmarshal(out, memoryManagerCheckpoint); err != nil {
		return nil, fmt.Errorf("failed to unmarshal memory manager state file: %v", err)
	}
	return memoryManagerCheckpoint, nil
}

type kubeletParams struct {
	memoryManagerFeatureGate bool
	memoryManagerPolicy      string
	systemReservedMemory     []kubeletconfig.MemoryReservation
	systemReserved           map[string]string
	kubeReserved             map[string]string
	evictionHard             map[string]string
}

func getUpdatedKubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration, params *kubeletParams) *kubeletconfig.KubeletConfiguration {
	newCfg := oldCfg.DeepCopy()

	if newCfg.FeatureGates == nil {
		newCfg.FeatureGates = map[string]bool{}
	}
	newCfg.FeatureGates["MemoryManager"] = params.memoryManagerFeatureGate
	newCfg.MemoryManagerPolicy = params.memoryManagerPolicy

	// update system-reserved
	if newCfg.SystemReserved == nil {
		newCfg.SystemReserved = map[string]string{}
	}
	for resourceName, value := range params.systemReserved {
		newCfg.SystemReserved[resourceName] = value
	}

	// update kube-reserved
	if newCfg.KubeReserved == nil {
		newCfg.KubeReserved = map[string]string{}
	}
	for resourceName, value := range params.kubeReserved {
		newCfg.KubeReserved[resourceName] = value
	}

	// update hard eviction threshold
	if newCfg.EvictionHard == nil {
		newCfg.EvictionHard = map[string]string{}
	}
	for resourceName, value := range params.evictionHard {
		newCfg.EvictionHard[resourceName] = value
	}

	// update reserved memory
	if newCfg.ReservedMemory == nil {
		newCfg.ReservedMemory = []kubeletconfig.MemoryReservation{}
	}
	for _, memoryReservation := range params.systemReservedMemory {
		newCfg.ReservedMemory = append(newCfg.ReservedMemory, memoryReservation)
	}

	return newCfg
}

func updateKubeletConfig(f *framework.Framework, cfg *kubeletconfig.KubeletConfiguration) {
	// remove the state file
	deleteMemoryManagerStateFile()

	// Update the Kubelet configuration
	framework.ExpectNoError(setKubeletConfiguration(f, cfg))

	// Wait for the Kubelet to be ready.
	gomega.Eventually(func() bool {
		nodes, err := e2enode.TotalReady(f.ClientSet)
		framework.ExpectNoError(err)
		return nodes == 1
	}, time.Minute, time.Second).Should(gomega.BeTrue())
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
var _ = SIGDescribe("Memory Manager [Serial] [Feature:MemoryManager][NodeAlphaFeature:MemoryManager]", func() {
	// TODO: add more complex tests that will include interaction between CPUManager, MemoryManager and TopologyManager
	var (
		allNUMANodes             []int
		ctnParams, initCtnParams []memoryManagerCtnAttributes
		is2MiHugepagesSupported  *bool
		isMultiNUMASupported     *bool
		kubeParams               *kubeletParams
		oldCfg                   *kubeletconfig.KubeletConfiguration
		testPod                  *v1.Pod
	)

	f := framework.NewDefaultFramework("memory-manager-test")

	memoryQuantatity := resource.MustParse("1100Mi")
	defaultKubeParams := &kubeletParams{
		memoryManagerFeatureGate: true,
		systemReservedMemory: []kubeletconfig.MemoryReservation{
			{
				NumaNode: 0,
				Limits: v1.ResourceList{
					resourceMemory: memoryQuantatity,
				},
			},
		},
		systemReserved: map[string]string{resourceMemory: "500Mi"},
		kubeReserved:   map[string]string{resourceMemory: "500Mi"},
		evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
	}

	verifyMemoryPinning := func(pod *v1.Pod, numaNodeIDs []int) {
		ginkgo.By("Verifying the NUMA pinning")

		output, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, pod.Name, pod.Spec.Containers[0].Name)
		framework.ExpectNoError(err)

		currentNUMANodeIDs, err := cpuset.Parse(strings.Trim(output, "\n"))
		framework.ExpectNoError(err)

		framework.ExpectEqual(numaNodeIDs, currentNUMANodeIDs.ToSlice())
	}

	ginkgo.BeforeEach(func() {
		if isMultiNUMASupported == nil {
			isMultiNUMASupported = pointer.BoolPtr(isMultiNUMA())
		}

		if is2MiHugepagesSupported == nil {
			is2MiHugepagesSupported = pointer.BoolPtr(isHugePageAvailable(hugepagesSize2M))
		}

		if len(allNUMANodes) == 0 {
			allNUMANodes = getAllNUMANodes()
		}
	})

	// dynamically update the kubelet configuration
	ginkgo.JustBeforeEach(func() {
		var err error

		// allocate hugepages
		if *is2MiHugepagesSupported {
			err := configureHugePages(hugepagesSize2M, 256)
			framework.ExpectNoError(err)
		}

		// get the old kubelet config
		oldCfg, err = getCurrentKubeletConfig()
		framework.ExpectNoError(err)

		// update the kubelet config with new parameters
		newCfg := getUpdatedKubeletConfig(oldCfg, kubeParams)
		updateKubeletConfig(f, newCfg)

		// request hugepages resources under the container
		if *is2MiHugepagesSupported {
			for i := 0; i < len(ctnParams); i++ {
				ctnParams[i].hugepages2Mi = "128Mi"
			}
		}

		testPod = makeMemoryManagerPod(ctnParams[0].ctnName, initCtnParams, ctnParams)
	})

	ginkgo.JustAfterEach(func() {
		// delete the test pod
		if testPod.Name != "" {
			f.PodClient().DeleteSync(testPod.Name, metav1.DeleteOptions{}, 2*time.Minute)
		}

		// release hugepages
		gomega.Eventually(func() error {
			return configureHugePages(hugepagesSize2M, 0)
		}, 90*time.Second, 15*time.Second).ShouldNot(gomega.HaveOccurred(), "failed to release hugepages")

		// update the kubelet config with old values
		updateKubeletConfig(f, oldCfg)
	})

	ginkgo.Context("with static policy", func() {
		ginkgo.BeforeEach(func() {
			// override kubelet configuration parameters
			tmpParams := *defaultKubeParams
			tmpParams.memoryManagerPolicy = staticPolicy
			kubeParams = &tmpParams
		})

		ginkgo.JustAfterEach(func() {
			// reset containers attributes
			ctnParams = []memoryManagerCtnAttributes{}
			initCtnParams = []memoryManagerCtnAttributes{}
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

			ginkgo.It("should succeed to start the pod", func() {
				ginkgo.By("Running the test pod")
				testPod = f.PodClient().CreateSync(testPod)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					return
				}

				verifyMemoryPinning(testPod, []int{0})
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

			ginkgo.It("should succeed to start the pod", func() {
				ginkgo.By("Running the test pod")
				testPod = f.PodClient().CreateSync(testPod)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					return
				}

				verifyMemoryPinning(testPod, []int{0})
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

			ginkgo.It("should succeed to start all pods", func() {
				ginkgo.By("Running the test pod and the test pod 2")
				testPod = f.PodClient().CreateSync(testPod)

				ginkgo.By("Running the test pod 2")
				testPod2 = f.PodClient().CreateSync(testPod2)

				// it no taste to verify NUMA pinning when the node has only one NUMA node
				if !*isMultiNUMASupported {
					return
				}

				verifyMemoryPinning(testPod, []int{0})
				verifyMemoryPinning(testPod2, []int{0})
			})

			ginkgo.JustAfterEach(func() {
				// delete the test pod 2
				if testPod2.Name != "" {
					f.PodClient().DeleteSync(testPod2.Name, metav1.DeleteOptions{}, 2*time.Minute)
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

			ginkgo.JustBeforeEach(func() {
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

					workloadPod = f.PodClient().CreateSync(workloadPod)
					workloadPods = append(workloadPods, workloadPod)
				}
			})

			ginkgo.It("should be rejected", func() {
				ginkgo.By("Creating the pod")
				testPod = f.PodClient().Create(testPod)

				ginkgo.By("Checking that pod failed to start because of admission error")
				gomega.Eventually(func() bool {
					tmpPod, err := f.PodClient().Get(context.TODO(), testPod.Name, metav1.GetOptions{})
					framework.ExpectNoError(err)

					if tmpPod.Status.Phase != v1.PodFailed {
						return false
					}

					if tmpPod.Status.Reason != "UnexpectedAdmissionError" {
						return false
					}

					if !strings.Contains(tmpPod.Status.Message, "Pod Allocate failed due to [memorymanager]") {
						return false
					}

					return true
				}, time.Minute, 5*time.Second).Should(
					gomega.Equal(true),
					"the pod succeeded to start, when it should fail with the admission error",
				)
			})

			ginkgo.JustAfterEach(func() {
				for _, workloadPod := range workloadPods {
					if workloadPod.Name != "" {
						f.PodClient().DeleteSync(workloadPod.Name, metav1.DeleteOptions{}, 2*time.Minute)
					}
				}
			})
		})
	})

	ginkgo.Context("with none policy", func() {
		ginkgo.BeforeEach(func() {
			tmpParams := *defaultKubeParams
			tmpParams.memoryManagerPolicy = nonePolicy
			kubeParams = &tmpParams

			// override pod parameters
			ctnParams = []memoryManagerCtnAttributes{
				{
					ctnName: "memory-manager-none",
					cpus:    "100m",
					memory:  "128Mi",
				},
			}
		})

		ginkgo.It("should succeed to start the pod", func() {
			testPod = f.PodClient().CreateSync(testPod)

			// it no taste to verify NUMA pinning when the node has only one NUMA node
			if !*isMultiNUMASupported {
				return
			}

			verifyMemoryPinning(testPod, allNUMANodes)
		})
	})
})

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
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"k8s.io/klog/v2"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpuset"
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
	reservedLimit          = "limit"
	reservedNUMANode       = "numa-node"
	reservedType           = "type"
	resourceMemory         = "memory"
	staticPolicy           = "static"
	nonePolicy             = "none"
)

// Helper for makeMemoryManagerPod().
type memoryManagerCtnAttributes struct {
	ctnName      string
	cpus         string
	memory       string
	hugepages2Mi string
}

// makeCPUMangerPod returns a pod with the provided ctnAttributes.
func makeMemoryManagerPod(podName string, ctnAttributes []memoryManagerCtnAttributes) *v1.Pod {
	hugepagesMount := false
	var containers []v1.Container
	for _, ctnAttr := range ctnAttributes {
		memsetCmd := fmt.Sprintf("grep Mems_allowed_list /proc/self/status | cut -f2 && sleep 1d")
		ctn := v1.Container{
			Name:  ctnAttr.ctnName,
			Image: busyboxImage,
			Resources: v1.ResourceRequirements{
				Limits: v1.ResourceList{
					v1.ResourceCPU:    resource.MustParse(ctnAttr.cpus),
					v1.ResourceMemory: resource.MustParse(ctnAttr.memory),
				},
			},
			Command: []string{"sh", "-c", memsetCmd},
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

	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers:    containers,
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

type kubeletParams struct {
	memoryManagerFeatureGate bool
	memoryManagerPolicy      string
	systemReservedMemory     []map[string]string
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
		newCfg.ReservedMemory = []map[string]string{}
	}
	for _, p := range params.systemReservedMemory {
		newCfg.ReservedMemory = append(newCfg.ReservedMemory, p)
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
		allNUMANodes            []int
		ctnParams               []memoryManagerCtnAttributes
		is2MiHugepagesSupported *bool
		isMultiNUMASupported    *bool
		kubeParams              *kubeletParams
		oldCfg                  *kubeletconfig.KubeletConfiguration
		testPod                 *v1.Pod
	)

	f := framework.NewDefaultFramework("memory-manager-test")
	defaultKubeParams := &kubeletParams{
		memoryManagerFeatureGate: true,
		systemReservedMemory: []map[string]string{
			{reservedNUMANode: "0", reservedType: resourceMemory, reservedLimit: "1100Mi"},
		},
		systemReserved: map[string]string{resourceMemory: "500Mi"},
		kubeReserved:   map[string]string{resourceMemory: "500Mi"},
		evictionHard:   map[string]string{evictionHardMemory: "100Mi"},
	}

	verifyMemoryPinning := func(numaNodeIDs []int) {
		ginkgo.By("Verifying the NUMA pinning")

		output, err := e2epod.GetPodLogs(f.ClientSet, f.Namespace.Name, testPod.Name, testPod.Spec.Containers[0].Name)
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

		testPod = makeMemoryManagerPod(ctnParams[0].ctnName, ctnParams)
	})

	ginkgo.JustAfterEach(func() {
		// delete the test pod
		f.PodClient().DeleteSync(testPod.Name, metav1.DeleteOptions{}, time.Minute)

		// release hugepages
		if err := configureHugePages(hugepagesSize2M, 0); err != nil {
			klog.Errorf("failed to release hugepages: %v", err)
		}

		// update the kubelet config with old values
		updateKubeletConfig(f, oldCfg)
	})

	ginkgo.Context("with static policy", func() {
		ginkgo.BeforeEach(func() {
			// override kubelet configuration parameters
			tmpParams := *defaultKubeParams
			tmpParams.memoryManagerPolicy = staticPolicy
			kubeParams = &tmpParams

			// override pod parameters
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

			verifyMemoryPinning([]int{0})
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

			verifyMemoryPinning(allNUMANodes)
		})
	})
})

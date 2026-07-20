//go:build windows

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

package e2enodewindows

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/onsi/gomega/gstruct"
	"github.com/onsi/gomega/types"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/cluster/ports"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm/topologymanager"
	"k8s.io/kubernetes/pkg/kubelet/winstats"
	e2emetrics "k8s.io/kubernetes/test/e2e/framework/metrics"
)

// nodeID is the identity extractor used by gstruct.MatchAllElements for
// node-scoped kubelet metrics: these samples carry no distinguishing label, so
// every sample maps to the same (empty) key. It mirrors the Linux e2e_node
// helper of the same name.
func nodeID(element interface{}) string {
	return ""
}

// timelessSample matches a single Prometheus sample by value while ignoring the
// metric labels, timestamp, and histogram. It mirrors the Linux e2e_node helper
// so the Windows topology manager metrics assertions read identically.
func timelessSample(value interface{}) types.GomegaMatcher {
	return gstruct.PointTo(gstruct.MatchAllFields(gstruct.Fields{
		// We already check Metric when matching the Id.
		"Metric":    gstruct.Ignore(),
		"Value":     gomega.BeNumerically("==", value),
		"Timestamp": gstruct.Ignore(),
		"Histogram": gstruct.Ignore(),
	}))
}

// getWindowsKubeletMetrics scrapes the kubelet's Prometheus metrics from the
// read-only port on the local node. It is the Windows counterpart of the Linux
// getKubeletMetrics helper and is used as a gomega.Eventually poll function.
func getWindowsKubeletMetrics(ctx context.Context) (e2emetrics.KubeletMetrics, error) {
	ginkgo.By("Getting Kubelet metrics from the metrics API")
	return e2emetrics.GrabKubeletMetricsWithoutProxy(ctx, fmt.Sprintf("%s:%d", nodeNameOrIP(), ports.KubeletReadOnlyPort), "/metrics")
}

// windowsDetectCPUsPerNUMANode returns a map of NUMA node number to the count of
// logical CPUs on that node, as reported by the Windows kernel. It walks NUMA
// node numbers from 0 upward and stops at the first node that reports no CPUs
// (or that the OS rejects), which marks the end of the populated nodes.
func windowsDetectCPUsPerNUMANode() map[int]int {
	perNode := map[int]int{}
	for node := 0; node <= 0xffff; node++ {
		affinities, err := winstats.GetCPUsforNUMANode(uint16(node))
		if err != nil {
			break
		}
		count := 0
		for _, aff := range affinities {
			count += len(aff.Processors())
		}
		if count == 0 {
			break
		}
		perNode[node] = count
	}
	return perNode
}

// buildWindowsMemoryManagerKubeletConfig returns a KubeletConfiguration with the
// static CPU manager (feature gate on) and the memory manager's BestEffort
// policy enabled together. The memory manager validates at startup that the sum
// of ReservedMemory across NUMA nodes equals the node's allocatable memory
// reservation (SystemReserved + KubeReserved + hard-eviction); it does not care
// how that total is distributed, so — like the Linux memory_manager_test — the
// whole reservation is placed on NUMA node 0.
func buildWindowsMemoryManagerKubeletConfig(oldCfg *kubeletconfig.KubeletConfiguration) *kubeletconfig.KubeletConfiguration {
	const (
		systemReservedMem = "500Mi"
		kubeReservedMem   = "500Mi"
		evictionHardMem   = "100Mi"
		// Total reservation = systemReserved + kubeReserved + hard-eviction.
		totalReservedMem = "1100Mi"
	)

	newCfg := buildWindowsCPUManagerKubeletConfig(oldCfg, true)
	newCfg.MemoryManagerPolicy = "BestEffort"
	// Keep the topology manager out of the admission decision so the memory
	// manager's Allocate always runs. With single-numa-node/restricted the
	// topology manager would reject an unsatisfiable request at the hint-merge
	// stage before Allocate is called, and the pinning counters would never
	// increment. "none" (like the Linux memory manager metrics test) admits the
	// pod to the allocate phase so the failure path is exercised.
	newCfg.TopologyManagerPolicy = topologymanager.PolicyNone

	if newCfg.SystemReserved == nil {
		newCfg.SystemReserved = map[string]string{}
	}
	newCfg.SystemReserved[string(v1.ResourceMemory)] = systemReservedMem

	if newCfg.KubeReserved == nil {
		newCfg.KubeReserved = map[string]string{}
	}
	newCfg.KubeReserved[string(v1.ResourceMemory)] = kubeReservedMem

	if newCfg.EvictionHard == nil {
		newCfg.EvictionHard = map[string]string{}
	}
	newCfg.EvictionHard["memory.available"] = evictionHardMem

	newCfg.ReservedMemory = []kubeletconfig.MemoryReservation{
		{
			NumaNode: 0,
			Limits: v1.ResourceList{
				v1.ResourceMemory: resource.MustParse(totalReservedMem),
			},
		},
	}
	return newCfg
}

// makeGuaranteedMemoryManagerPodWindows builds a Guaranteed-QoS Windows pod with
// a single container named "gu-ctr" whose CPU and memory requests equal their
// limits. It is used to exercise the memory manager's allocation paths (e.g. an
// oversized memory request that no NUMA node can satisfy).
func makeGuaranteedMemoryManagerPodWindows(name, cpu, memory string) *v1.Pod {
	rl := v1.ResourceList{
		v1.ResourceCPU:    resource.MustParse(cpu),
		v1.ResourceMemory: resource.MustParse(memory),
	}
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.PodSpec{
			RestartPolicy: v1.RestartPolicyNever,
			Containers: []v1.Container{
				{
					Name:  "gu-ctr",
					Image: busyboxImage,
					Resources: v1.ResourceRequirements{
						Requests: rl,
						Limits:   rl,
					},
					// Long-running sleep; powershell.exe is available in the Windows BusyBox image.
					Command: []string{"powershell.exe", "-Command", "Start-Sleep -Seconds 86400"},
				},
			},
			NodeSelector: map[string]string{"kubernetes.io/os": "windows"},
		},
	}
}

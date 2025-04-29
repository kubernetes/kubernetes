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

package node

import (
	"context"
	"fmt"
	"strconv"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/common/node/framework/podresize"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	fakeExtendedResource = "dummy.com/dummy"

	originalCPU       = "20m"
	originalCPULimit  = "30m"
	reducedCPU        = "15m"
	reducedCPULimit   = "25m"
	increasedCPU      = "25m"
	increasedCPULimit = "35m"

	originalMem       = "20Mi"
	originalMemLimit  = "30Mi"
	reducedMem        = "15Mi"
	reducedMemLimit   = "25Mi"
	increasedMem      = "25Mi"
	increasedMemLimit = "35Mi"
)

func offsetCPU(index int, value string) string {
	val := resource.MustParse(value)
	ptr := &val
	ptr.Add(resource.MustParse(fmt.Sprintf("%dm", 2*index)))
	return ptr.String()
}

func offsetMemory(index int64, value string) string {
	val := resource.MustParse(value)
	ptr := &val
	ptr.Add(resource.MustParse(fmt.Sprintf("%dMi", 2*index)))
	return ptr.String()
}

func doPodResizeTests(f *framework.Framework) {
	type testCase struct {
		name                string
		containers          []podresize.ResizableContainerInfo
		patchString         string
		expected            []podresize.ResizableContainerInfo
		addExtendedResource bool
		// TODO(123940): test rollback for all test cases once resize is more responsive.
		testRollback bool
	}

	noRestart := v1.NotRequired
	doRestart := v1.RestartContainer
	tests := []testCase{
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
					{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
				]}}`, increasedCPU, increasedMem, increasedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: increasedMem, MemLim: increasedMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedCPU, reducedCPU),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			testRollback: true,
		},
		{
			name: "Guaranteed QoS pod, three containers (c1, c2, c3) - increase: CPU (c1,c3), memory (c2, c3) ; decrease: CPU (c2)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, originalCPU), CPULim: offsetCPU(1, originalCPU), MemReq: offsetMemory(1, originalMem), MemLim: offsetMemory(1, originalMem)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, originalCPU), CPULim: offsetCPU(2, originalCPU), MemReq: offsetMemory(2, originalMem), MemLim: offsetMemory(2, originalMem)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}},
							{"name":"c2", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}},
							{"name":"c3", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`,
				increasedCPU, increasedCPU,
				offsetCPU(1, reducedCPU), offsetMemory(1, increasedMem), offsetCPU(1, reducedCPU), offsetMemory(1, increasedMem),
				offsetCPU(2, increasedCPU), offsetMemory(2, increasedMem), offsetCPU(2, increasedCPU), offsetMemory(2, increasedMem)),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, reducedCPU), CPULim: offsetCPU(1, reducedCPU), MemReq: offsetMemory(1, increasedMem), MemLim: offsetMemory(1, increasedMem)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, increasedCPU), CPULim: offsetCPU(2, increasedCPU), MemReq: offsetMemory(2, increasedMem), MemLim: offsetMemory(2, increasedMem)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"}}}
						]}}`, reducedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: reducedMem, MemLim: originalMemLimit},
				},
			},
			testRollback: true,
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"}}}
						]}}`, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: increasedMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory limits only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"limits":{"memory":"%s"}}}
						]}}`, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: increasedMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"}}}
						]}}`, reducedCPU),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			testRollback: true,
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU limits only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"limits":{"cpu":"%s"}}}
						]}}`, reducedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: reducedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			testRollback: true,
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"}}}
						]}}`, increasedCPU),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU limits only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"limits":{"cpu":"%s"}}}
						]}}`, increasedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedCPU, reducedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, increasedCPU, increasedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase CPU limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedCPU, increasedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease CPU limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, increasedCPU, reducedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: reducedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"},"limits":{"memory":"%s"}}}
						]}}`, increasedMem, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: increasedMem, MemLim: increasedMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase memory limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"},"limits":{"memory":"%s"}}}
						]}}`, reducedMem, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: reducedMem, MemLim: increasedMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase memory limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"},"limits":{"memory":"%s"}}}
						]}}`, reducedCPU, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: increasedMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase CPU limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedMem, increasedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: increasedCPULimit, MemReq: reducedMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease CPU limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, increasedMem, reducedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: reducedCPULimit, MemReq: increasedMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - decrease memory request",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"memory":"%s"}}}
						]}}`, reducedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: reducedMem},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - increase cpu request",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s"}}}
						]}}`, increasedCPU),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, MemReq: originalMem},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu requests and limits - resize with equivalents",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: "2m", CPULim: "10m"},
				},
			},
			patchString: `{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"1m"},"limits":{"cpu":"5m"}}}
						]}}`,
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: "1m", CPULim: "5m"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU (NotRequired) & memory (RestartContainer)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, increasedCPU, increasedMem, increasedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: increasedMem, MemLim: increasedMem},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
			testRollback: true,
		},
		{
			name: "Burstable QoS pod, one container - decrease CPU (NotRequired) & memory (RestartContainer)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, reducedCPU, reducedMem, reducedCPULimit, reducedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPULimit, MemReq: reducedMem, MemLim: reducedMemLimit},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
			testRollback: true,
		},
		{
			name: "Burstable QoS pod, one container - decrease memory request (RestartContainer memory resize policy)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
					]}}`, originalCPU, reducedMem, originalCPULimit, originalMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: reducedMem, MemLim: originalMemLimit},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container - increase memory request (NoRestart memory resize policy)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, originalCPU, increasedMem, originalCPULimit, originalMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: increasedMem, MemLim: originalMemLimit},
					CPUPolicy:    &noRestart,
					MemPolicy:    &noRestart,
					RestartCount: 0,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - increase c1 resources, no change for c2, decrease c3 resources (no net change for pod)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, originalCPU), CPULim: offsetCPU(1, originalCPULimit), MemReq: offsetMemory(1, originalMem), MemLim: offsetMemory(1, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, originalCPU), CPULim: offsetCPU(2, originalCPULimit), MemReq: offsetMemory(2, originalMem), MemLim: offsetMemory(2, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}},
							{"name":"c3", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s"}}}
						]}}`,
				increasedCPU, increasedMem, increasedCPULimit, increasedMemLimit,
				offsetCPU(2, reducedCPU), offsetMemory(2, reducedMem), offsetCPU(2, reducedCPULimit)),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPULimit, MemReq: increasedMem, MemLim: increasedMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, originalCPU), CPULim: offsetCPU(1, originalCPULimit), MemReq: offsetMemory(1, originalMem), MemLim: offsetMemory(1, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, reducedCPU), CPULim: offsetCPU(2, reducedCPULimit), MemReq: offsetMemory(2, reducedMem), MemLim: offsetMemory(2, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - decrease c1 resources, increase c2 resources, no change for c3 (net increase for pod)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, originalCPU), CPULim: offsetCPU(1, originalCPULimit), MemReq: offsetMemory(1, originalMem), MemLim: offsetMemory(1, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, originalCPU), CPULim: offsetCPU(2, originalCPULimit), MemReq: offsetMemory(2, originalMem), MemLim: offsetMemory(2, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s"}}},
							{"name":"c2", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`,
				reducedCPU, reducedMem, reducedCPULimit,
				offsetCPU(2, increasedCPU), offsetMemory(2, increasedMem), offsetCPU(2, increasedCPULimit), offsetMemory(2, increasedMemLimit)),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPULimit, MemReq: reducedMem, MemLim: originalMemLimit},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:         "c2",
					Resources:    &podresize.ContainerResources{CPUReq: offsetCPU(2, increasedCPU), CPULim: offsetCPU(2, increasedCPULimit), MemReq: offsetMemory(2, increasedMem), MemLim: offsetMemory(2, increasedMemLimit)},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, originalCPU), CPULim: offsetCPU(2, originalCPULimit), MemReq: offsetMemory(2, originalMem), MemLim: offsetMemory(2, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - no change for c1, increase c2 resources, decrease c3 (net decrease for pod)",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(1, originalCPU), CPULim: offsetCPU(1, originalCPULimit), MemReq: offsetMemory(1, originalMem), MemLim: offsetMemory(1, originalMemLimit)},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &podresize.ContainerResources{CPUReq: offsetCPU(2, originalCPU), CPULim: offsetCPU(2, originalCPULimit), MemReq: offsetMemory(2, originalMem), MemLim: offsetMemory(2, originalMemLimit)},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c2", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}},
							{"name":"c3", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`,
				offsetCPU(1, increasedCPU), offsetMemory(1, increasedMem), offsetCPU(1, increasedCPULimit), offsetMemory(1, increasedMemLimit),
				reducedCPU, reducedMem, reducedCPULimit, reducedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:         "c2",
					Resources:    &podresize.ContainerResources{CPUReq: offsetCPU(1, increasedCPU), CPULim: offsetCPU(1, increasedCPULimit), MemReq: offsetMemory(1, increasedMem), MemLim: offsetMemory(1, increasedMemLimit)},
					CPUPolicy:    &doRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
				{
					Name:         "c3",
					Resources:    &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPULimit, MemReq: reducedMem, MemLim: reducedMemLimit},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - scale up cpu and memory",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, increasedCPU, increasedMem, increasedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: increasedMem, MemLim: increasedMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{},
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - add requests",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c2", "resources":{"requests":{"cpu":"%s","memory":"%s"}}}
						]}}`, originalCPU, originalMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - add limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
							{"name":"c2", "resources":{"limits":{"cpu":"%s"}}}
						]}}`, originalCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory with an extended resource",
			containers: []podresize.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem,
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, increasedCPU, increasedMem, increasedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: increasedMem, MemLim: increasedMem,
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			addExtendedResource: true,
		},
		{
			name: "BestEffort QoS pod - empty resize",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{},
				},
			},
			patchString: `{}`,
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one restartable init container - increase CPU & memory",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy:     &noRestart,
					MemPolicy:     &noRestart,
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, increasedCPU, increasedMem, increasedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPU, MemReq: increasedMem, MemLim: increasedMem},
					CPUPolicy:     &noRestart,
					MemPolicy:     &noRestart,
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one restartable init container - decrease CPU & increase memory",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, reducedCPU, increasedMem, reducedCPU, increasedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPU, MemReq: increasedMem, MemLim: increasedMem},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container, one restartable init container - decrease init container CPU",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedCPU, reducedCPU),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPU, MemReq: originalMem, MemLim: originalMem},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container, one restartable init container - increase init container CPU & memory",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
						]}}`, increasedCPU, increasedMem, increasedCPULimit, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPULimit, MemReq: increasedMem, MemLim: increasedMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container, one restartable init container - decrease init container CPU only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, reducedCPU, reducedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: reducedCPU, CPULim: reducedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container, one restartable init container - increase init container CPU only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"cpu":"%s"},"limits":{"cpu":"%s"}}}
						]}}`, increasedCPU, increasedCPULimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: increasedCPU, CPULim: increasedCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container, one restartable init container - decrease init container memory requests only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"memory":"%s"}}}
						]}}`, reducedMem),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: reducedMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
		{
			name: "Burstable QoS pod, one container, one restartable init container - increase init container memory only",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"initContainers":[
							{"name":"c1-init", "resources":{"requests":{"memory":"%s"},"limits":{"memory":"%s"}}}
						]}}`, increasedMem, increasedMemLimit),
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPU, MemReq: originalMem, MemLim: originalMem},
				},
				{
					Name:          "c1-init",
					Resources:     &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: increasedMem, MemLim: increasedMemLimit},
					InitCtr:       true,
					RestartPolicy: v1.ContainerRestartPolicyAlways,
				},
			},
		},
	}

	for idx := range tests {
		tc := tests[idx]

		ginkgo.It(tc.name, func(ctx context.Context) {
			podClient := e2epod.NewPodClient(f)
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod = podresize.MakePodWithResizableContainers(f.Namespace.Name, "", tStamp, tc.containers)
			testPod.GenerateName = "resize-test-"
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			if tc.addExtendedResource {
				nodes, err := e2enode.GetReadySchedulableNodes(context.Background(), f.ClientSet)
				framework.ExpectNoError(err)

				for _, node := range nodes.Items {
					e2enode.AddExtendedResource(ctx, f.ClientSet, node.Name, fakeExtendedResource, resource.MustParse("123"))
				}
				defer func() {
					for _, node := range nodes.Items {
						e2enode.RemoveExtendedResource(ctx, f.ClientSet, node.Name, fakeExtendedResource)
					}
				}()
			}

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources are as expected")
			podresize.VerifyPodResources(newPod, tc.containers)
			ginkgo.By("verifying initial pod resize policy is as expected")
			podresize.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources are as expected")
			framework.ExpectNoError(podresize.VerifyPodStatusResources(newPod, tc.containers))
			ginkgo.By("verifying initial cgroup config are as expected")
			framework.ExpectNoError(podresize.VerifyPodContainersCgroupValues(ctx, f, newPod, tc.containers))

			patchAndVerify := func(patchString string, expectedContainers []podresize.ResizableContainerInfo, opStr string) {
				ginkgo.By(fmt.Sprintf("patching pod for %s", opStr))
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
					types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))
				expected := podresize.UpdateExpectedContainerRestarts(ctx, patchedPod, expectedContainers)

				ginkgo.By(fmt.Sprintf("verifying pod patched for %s", opStr))
				podresize.VerifyPodResources(patchedPod, expected)

				ginkgo.By(fmt.Sprintf("waiting for %s to be actuated", opStr))
				resizedPod := podresize.WaitForPodResizeActuation(ctx, f, podClient, newPod, expected)
				podresize.ExpectPodResized(ctx, f, resizedPod, expected)
			}

			patchAndVerify(tc.patchString, tc.expected, "resize")

			if tc.testRollback {
				// Resize has been actuated, test rollback
				rollbackContainers := make([]podresize.ResizableContainerInfo, len(tc.containers))
				copy(rollbackContainers, tc.containers)
				for i, c := range rollbackContainers {
					gomega.Expect(c.Name).To(gomega.Equal(tc.expected[i].Name),
						"test case containers & expectations should be in the same order")
					// Resizes that trigger a restart should trigger a second restart when rolling back.
					rollbackContainers[i].RestartCount = tc.expected[i].RestartCount
				}

				rbPatchStr, err := podresize.ResizeContainerPatch(tc.containers)
				framework.ExpectNoError(err)
				patchAndVerify(rbPatchStr, rollbackContainers, "rollback")
			}

			ginkgo.By("deleting pod")
			podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		})
	}
}

func doPodResizeErrorTests(f *framework.Framework) {

	type testCase struct {
		name        string
		containers  []podresize.ResizableContainerInfo
		patchString string
		patchError  string
		expected    []podresize.ResizableContainerInfo
	}

	tests := []testCase{
		{
			name: "BestEffort pod - try requesting memory, expect error",
			containers: []podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"40Mi"}}}
					]}}`,
			patchError: "Pod QOS Class may not change as a result of resizing",
			expected: []podresize.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - remove memory limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory": null}}}
					]}}`,
			patchError: "resource limits cannot be removed",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - remove CPU limits",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu": null}}}
					]}}`,
			patchError: "resource limits cannot be removed",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with memory requests + limits, cpu requests - remove CPU requests",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu": null}}}
					]}}`,
			patchError: "resource requests cannot be removed",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with CPU requests + limits, cpu requests - remove memory requests",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory": null}}}
					]}}`,
			patchError: "resource requests cannot be removed",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem},
				},
			},
		},
		{
			name: "Burstable QoS pod, two containers with cpu & memory requests + limits - reorder containers",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
				{"name":"c2", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}},
				{"name":"c1", "resources":{"requests":{"cpu":"%s","memory":"%s"},"limits":{"cpu":"%s","memory":"%s"}}}
			]}}`, originalCPU, originalMem, originalCPULimit, originalMemLimit, originalCPU, originalMem, originalCPULimit, originalMemLimit),
			patchError: "spec.containers[0].name: Forbidden: containers may not be renamed or reordered on resize, spec.containers[1].name: Forbidden: containers may not be renamed or reordered on resize",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
				{
					Name:      "c2",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
		{
			name: "Burstable QoS pod with memory requests + limits - decrease memory limit",
			containers: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
			patchString: fmt.Sprintf(`{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"%s"}}}
					]}}`, reducedMemLimit),
			patchError: "memory limits cannot be decreased",
			expected: []podresize.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &podresize.ContainerResources{CPUReq: originalCPU, CPULim: originalCPULimit, MemReq: originalMem, MemLim: originalMemLimit},
				},
			},
		},
	}

	for idx := range tests {
		tc := tests[idx]

		ginkgo.It(tc.name, func(ctx context.Context) {
			podClient := e2epod.NewPodClient(f)
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			testPod = podresize.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, and policy are as expected")
			podresize.VerifyPodResources(newPod, tc.containers)
			podresize.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			framework.ExpectNoError(podresize.VerifyPodStatusResources(newPod, tc.containers))

			ginkgo.By("patching pod for resize")
			patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
				types.StrategicMergePatchType, []byte(tc.patchString), metav1.PatchOptions{}, "resize")
			if tc.patchError == "" {
				framework.ExpectNoError(pErr, "failed to patch pod for resize")
			} else {
				gomega.Expect(pErr).To(gomega.HaveOccurred())
				gomega.Expect(pErr.Error()).To(gomega.ContainSubstring(tc.patchError))
				patchedPod = newPod
			}

			ginkgo.By("verifying pod resources after patch")
			podresize.VerifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod status resources after patch")
			framework.ExpectNoError(podresize.VerifyPodStatusResources(patchedPod, tc.expected))

			ginkgo.By("deleting pod")
			podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, f.Timeouts.PodDelete)
		})
	}
}

// NOTE: Pod resize scheduler resource quota tests are out of scope in e2e_node tests,
//       because in e2e_node tests
//          a) scheduler and controller manager is not running by the Node e2e
//          b) api-server in services doesn't start with --enable-admission-plugins=ResourceQuota
//             and is not possible to start it from TEST_ARGS
//       Above tests are performed by doSheduletTests() and doPodResizeResourceQuotaTests()
//       in test/e2e/node/pod_resize.go

var _ = SIGDescribe("Pod InPlace Resize Container", framework.WithFeatureGate(features.InPlacePodVerticalScaling), func() {
	f := framework.NewDefaultFramework("pod-resize-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		_, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	doPodResizeTests(f)
	doPodResizeErrorTests(f)
})

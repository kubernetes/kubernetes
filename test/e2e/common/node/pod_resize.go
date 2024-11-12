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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	v1 "k8s.io/api/core/v1"
)

const (
	fakeExtendedResource = "dummy.com/dummy"
)

func doPodResizeTests(f *framework.Framework) {
	type testCase struct {
		name                string
		containers          []e2epod.ResizableContainerInfo
		patchString         string
		expected            []e2epod.ResizableContainerInfo
		addExtendedResource bool
		// TODO(123940): test rollback for all test cases once resize is more responsive.
		testRollback bool
	}

	noRestart := v1.NotRequired
	doRestart := v1.RestartContainer
	tests := []testCase{
		{
			name:         "Guaranteed QoS pod, one container - increase CPU & memory",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "500Mi", MemLim: "500Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m","memory":"250Mi"},"limits":{"cpu":"100m","memory":"250Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & decrease memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"100Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "100Mi", MemLim: "100Mi"},
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - decrease CPU & increase memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"300Mi"},"limits":{"cpu":"50m","memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "50m", CPULim: "50m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name:         "Guaranteed QoS pod, three containers (c1, c2, c3) - increase: CPU (c1,c3), memory (c2) ; decrease: CPU (c2), memory (c1,c3)",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "300m", MemReq: "300Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"140m","memory":"50Mi"},"limits":{"cpu":"140m","memory":"50Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"150m","memory":"240Mi"},"limits":{"cpu":"150m","memory":"240Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"340m","memory":"250Mi"},"limits":{"cpu":"340m","memory":"250Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "140m", CPULim: "140m", MemReq: "50Mi", MemLim: "50Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "150m", MemReq: "240Mi", MemLim: "240Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "340m", CPULim: "340m", MemReq: "250Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name:         "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests only",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name:         "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory limits only",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory":"600Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "600Mi"},
				},
			},
		},
		{
			name:         "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests only",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name:         "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU limits only",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU limits only",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"200m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"400m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"cpu":"500m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "500m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"memory":"300Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "300Mi", MemLim: "300Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease CPU requests and increase memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"100m"},"limits":{"memory":"500Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase CPU requests and decrease memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "400m", MemReq: "200Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m"},"limits":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - decrease memory requests and increase CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"100Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "300m", MemReq: "100Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - increase memory requests and decrease CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "200Mi", MemLim: "400Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"300Mi"},"limits":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "300Mi", MemLim: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - decrease memory request",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "400Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests - increase cpu request",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"300m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", MemReq: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu requests - resize with equivalent request",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "2m"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"1m"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "1m"},
				},
			},
		},
		{
			name:         "Guaranteed QoS pod, one container - increase CPU (NotRequired) & memory (RestartContainer)",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name:         "Burstable QoS pod, one container - decrease CPU (RestartContainer) & memory (NotRequired)",
			testRollback: true,
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "200Mi", MemLim: "400Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"100Mi"},"limits":{"cpu":"100m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:         "c1",
					Resources:    &e2epod.ContainerResources{CPUReq: "50m", CPULim: "100m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - increase c1 resources, no change for c2, decrease c3 resources (no net change for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"150m","memory":"150Mi"},"limits":{"cpu":"250m","memory":"250Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "150m", CPULim: "250m", MemReq: "150Mi", MemLim: "250Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - decrease c1 resources, increase c2 resources, no change for c3 (net increase for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"50m","memory":"50Mi"},"limits":{"cpu":"150m","memory":"150Mi"}}},
						{"name":"c2", "resources":{"requests":{"cpu":"350m","memory":"350Mi"},"limits":{"cpu":"450m","memory":"450Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "50m", CPULim: "150m", MemReq: "50Mi", MemLim: "150Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:         "c2",
					Resources:    &e2epod.ContainerResources{CPUReq: "350m", CPULim: "450m", MemReq: "350Mi", MemLim: "450Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, three containers - no change for c1, increase c2 resources, decrease c3 (net decrease for pod)",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "300m", MemReq: "200Mi", MemLim: "300Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c3",
					Resources: &e2epod.ContainerResources{CPUReq: "300m", CPULim: "400m", MemReq: "300Mi", MemLim: "400Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &doRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c2", "resources":{"requests":{"cpu":"250m","memory":"250Mi"},"limits":{"cpu":"350m","memory":"350Mi"}}},
						{"name":"c3", "resources":{"requests":{"cpu":"100m","memory":"100Mi"},"limits":{"cpu":"200m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &doRestart,
					MemPolicy: &doRestart,
				},
				{
					Name:         "c2",
					Resources:    &e2epod.ContainerResources{CPUReq: "250m", CPULim: "350m", MemReq: "250Mi", MemLim: "350Mi"},
					CPUPolicy:    &noRestart,
					MemPolicy:    &noRestart,
					RestartCount: 1,
				},
				{
					Name:         "c3",
					Resources:    &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy:    &doRestart,
					MemPolicy:    &doRestart,
					RestartCount: 1,
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - scale up cpu and memory",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"200Mi"},"limits":{"cpu":"200m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "200Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{},
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - add requests",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c2", "resources":{"requests":{"cpu":"100m","memory":"100Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", MemReq: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Burstable QoS pod, mixed containers - add limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", MemReq: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c2", "resources":{"limits":{"cpu":"200m","memory":"200Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "100Mi", MemLim: "100Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
				{
					Name:      "c2",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "200m", MemReq: "100Mi", MemLim: "200Mi"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
		},
		{
			name: "Guaranteed QoS pod, one container - increase CPU & memory with an extended resource",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "100m", CPULim: "100m", MemReq: "200Mi", MemLim: "200Mi",
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			patchString: `{"spec":{"containers":[
					{"name":"c1", "resources":{"requests":{"cpu":"200m","memory":"400Mi"},"limits":{"cpu":"200m","memory":"400Mi"}}}
					]}}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "200m", MemReq: "400Mi", MemLim: "400Mi",
						ExtendedResourceReq: "1", ExtendedResourceLim: "1"},
					CPUPolicy: &noRestart,
					MemPolicy: &noRestart,
				},
			},
			addExtendedResource: true,
		},
		{
			name: "BestEffort QoS pod - empty resize",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{},
				},
			},
			patchString: `{}`,
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{},
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
			e2epod.InitDefaultResizePolicy(tc.containers)
			e2epod.InitDefaultResizePolicy(tc.expected)
			testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, "", tStamp, tc.containers)
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
			e2epod.VerifyPodResources(newPod, tc.containers)
			ginkgo.By("verifying initial pod resize policy is as expected")
			e2epod.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources are as expected")
			framework.ExpectNoError(e2epod.VerifyPodStatusResources(newPod, tc.containers))
			ginkgo.By("verifying initial cgroup config are as expected")
			framework.ExpectNoError(e2epod.VerifyPodContainersCgroupValues(ctx, f, newPod, tc.containers))

			patchAndVerify := func(patchString string, expectedContainers []e2epod.ResizableContainerInfo, opStr string) {
				ginkgo.By(fmt.Sprintf("patching pod for %s", opStr))
				patchedPod, pErr = f.ClientSet.CoreV1().Pods(newPod.Namespace).Patch(ctx, newPod.Name,
					types.StrategicMergePatchType, []byte(patchString), metav1.PatchOptions{}, "resize")
				framework.ExpectNoError(pErr, fmt.Sprintf("failed to patch pod for %s", opStr))

				ginkgo.By(fmt.Sprintf("verifying pod patched for %s", opStr))
				e2epod.VerifyPodResources(patchedPod, expectedContainers)

				ginkgo.By(fmt.Sprintf("waiting for %s to be actuated", opStr))
				resizedPod := e2epod.WaitForPodResizeActuation(ctx, f, podClient, newPod)
				e2epod.ExpectPodResized(ctx, f, resizedPod, expectedContainers)
			}

			patchAndVerify(tc.patchString, tc.expected, "resize")

			if tc.testRollback {
				// Resize has been actuated, test rollback
				rollbackContainers := make([]e2epod.ResizableContainerInfo, len(tc.containers))
				copy(rollbackContainers, tc.containers)
				for i, c := range rollbackContainers {
					gomega.Expect(c.Name).To(gomega.Equal(tc.expected[i].Name),
						"test case containers & expectations should be in the same order")
					// Resizes that trigger a restart should trigger a second restart when rolling back.
					rollbackContainers[i].RestartCount = tc.expected[i].RestartCount * 2
				}

				rbPatchStr, err := e2epod.ResizeContainerPatch(tc.containers)
				framework.ExpectNoError(err)
				patchAndVerify(rbPatchStr, rollbackContainers, "rollback")
			}

			ginkgo.By("deleting pod")
			framework.ExpectNoError(podClient.Delete(ctx, newPod.Name, metav1.DeleteOptions{}))
		})
	}
}

func doPodResizeErrorTests(f *framework.Framework) {

	type testCase struct {
		name        string
		containers  []e2epod.ResizableContainerInfo
		patchString string
		patchError  string
		expected    []e2epod.ResizableContainerInfo
	}

	tests := []testCase{
		{
			name: "BestEffort pod - try requesting memory, expect error",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory":"400Mi"}}}
					]}}`,
			patchError: "Pod QOS Class may not change as a result of resizing",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name: "c1",
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - remove memory limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"memory": null}}}
					]}}`,
			patchError: "resource limits cannot be removed",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with cpu & memory requests + limits - remove CPU limits",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"limits":{"cpu": null}}}
					]}}`,
			patchError: "resource limits cannot be removed",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with memory requests + limits, cpu requests - remove CPU requests",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"cpu": null}}}
					]}}`,
			patchError: "resource requests cannot be removed",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", MemReq: "250Mi", MemLim: "500Mi"},
				},
			},
		},
		{
			name: "Burstable QoS pod, one container with CPU requests + limits, cpu requests - remove memory requests",
			containers: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi"},
				},
			},
			patchString: `{"spec":{"containers":[
						{"name":"c1", "resources":{"requests":{"memory": null}}}
					]}}`,
			patchError: "resource requests cannot be removed",
			expected: []e2epod.ResizableContainerInfo{
				{
					Name:      "c1",
					Resources: &e2epod.ContainerResources{CPUReq: "200m", CPULim: "400m", MemReq: "250Mi"},
				},
			},
		},
	}

	timeouts := f.Timeouts

	for idx := range tests {
		tc := tests[idx]
		ginkgo.It(tc.name, func(ctx context.Context) {
			podClient := e2epod.NewPodClient(f)
			var testPod, patchedPod *v1.Pod
			var pErr error

			tStamp := strconv.Itoa(time.Now().Nanosecond())
			e2epod.InitDefaultResizePolicy(tc.containers)
			e2epod.InitDefaultResizePolicy(tc.expected)
			testPod = e2epod.MakePodWithResizableContainers(f.Namespace.Name, "testpod", tStamp, tc.containers)
			testPod = e2epod.MustMixinRestrictedPodSecurity(testPod)

			ginkgo.By("creating pod")
			newPod := podClient.CreateSync(ctx, testPod)

			ginkgo.By("verifying initial pod resources, and policy are as expected")
			e2epod.VerifyPodResources(newPod, tc.containers)
			e2epod.VerifyPodResizePolicy(newPod, tc.containers)

			ginkgo.By("verifying initial pod status resources and cgroup config are as expected")
			framework.ExpectNoError(e2epod.VerifyPodStatusResources(newPod, tc.containers))

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
			e2epod.VerifyPodResources(patchedPod, tc.expected)

			ginkgo.By("verifying pod status resources after patch")
			framework.ExpectNoError(e2epod.VerifyPodStatusResources(patchedPod, tc.expected))

			ginkgo.By("deleting pod")
			podClient.DeleteSync(ctx, newPod.Name, metav1.DeleteOptions{}, timeouts.PodDelete)
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

var _ = SIGDescribe("Pod InPlace Resize Container", framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("pod-resize-tests")

	ginkgo.BeforeEach(func(ctx context.Context) {
		node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
		framework.ExpectNoError(err)
		if framework.NodeOSDistroIs("windows") || e2enode.IsARM64(node) {
			e2eskipper.Skipf("runtime does not support InPlacePodVerticalScaling -- skipping")
		}
	})

	doPodResizeTests(f)
	doPodResizeErrorTests(f)
})

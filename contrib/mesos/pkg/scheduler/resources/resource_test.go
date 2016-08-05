/*
Copyright 2015 The Kubernetes Authors.

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

package resources

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
)

type resources struct {
	cpuR, cpuL, memR, memL float64
}

func TestResources(tst *testing.T) {
	assert := assert.New(tst)

	const (
		defCpu = float64(DefaultDefaultContainerCPULimit)
		defMem = float64(DefaultDefaultContainerMemLimit)
		minCpu = float64(MinimumContainerCPU)
		minMem = float64(MinimumContainerMem)
	)
	undef := math.NaN()
	defined := func(f float64) bool { return !math.IsNaN(f) }

	for _, t := range []struct {
		input, want resources
	}{
		{resources{undef, 3.0, undef, 768.0}, resources{defCpu + 3.0, defCpu + 3.0, defMem + 768.0, defMem + 768.0}},
		{resources{0.0, 3.0, 0.0, 768.0}, resources{defCpu + minCpu, defCpu + 3.0, defMem + minMem, defMem + 768.0}},
		{resources{undef, undef, undef, undef}, resources{2 * defCpu, 2 * defCpu, 2 * defMem, 2 * defMem}},
		{resources{0.0, 0.0, 0.0, 0.0}, resources{minCpu + defCpu, minCpu + defCpu, minMem + defMem, minMem + defMem}},
		{resources{2.0, 3.0, undef, 768.0}, resources{defCpu + 2.0, defCpu + 3.0, defMem + 768.0, defMem + 768.0}},
		{resources{2.0, 3.0, 256.0, 768.0}, resources{defCpu + 2.0, defCpu + 3.0, defMem + 256.0, defMem + 768.0}},
	} {
		pod := &api.Pod{
			Spec: api.PodSpec{
				Containers: []api.Container{{
					Name: "a",
				}, {
					Name: "b",
				}},
			},
		}

		if defined(t.input.cpuR) || defined(t.input.memR) {
			pod.Spec.Containers[0].Resources.Requests = api.ResourceList{}
			if defined(t.input.cpuR) {
				pod.Spec.Containers[0].Resources.Requests[api.ResourceCPU] = *CPUShares(t.input.cpuR).Quantity()
			}
			if defined(t.input.memR) {
				pod.Spec.Containers[0].Resources.Requests[api.ResourceMemory] = *MegaBytes(t.input.memR).Quantity()
			}
		}
		if defined(t.input.cpuL) || defined(t.input.memL) {
			pod.Spec.Containers[0].Resources.Limits = api.ResourceList{}
			if defined(t.input.cpuL) {
				pod.Spec.Containers[0].Resources.Limits[api.ResourceCPU] = *CPUShares(t.input.cpuL).Quantity()
			}
			if defined(t.input.memL) {
				pod.Spec.Containers[0].Resources.Limits[api.ResourceMemory] = *MegaBytes(t.input.memL).Quantity()
			}
		}

		tst.Logf("Testing resource computation for %v => request=%v limit=%v", t, pod.Spec.Containers[0].Resources.Requests, pod.Spec.Containers[0].Resources.Limits)

		beforeCpuR, beforeCpuL, _, err := LimitedCPUForPod(pod, DefaultDefaultContainerCPULimit)
		assert.NoError(err, "CPUForPod should not return an error")

		beforeMemR, beforeMemL, _, err := LimitedMemForPod(pod, DefaultDefaultContainerMemLimit)
		assert.NoError(err, "MemForPod should not return an error")

		cpuR, cpuL, _, err := LimitPodCPU(pod, DefaultDefaultContainerCPULimit)
		assert.NoError(err, "LimitPodCPU should not return an error")

		memR, memL, _, err := LimitPodMem(pod, DefaultDefaultContainerMemLimit)
		assert.NoError(err, "LimitPodMem should not return an error")

		tst.Logf("New resources container 0: request=%v limit=%v", pod.Spec.Containers[0].Resources.Requests, pod.Spec.Containers[0].Resources.Limits)
		tst.Logf("New resources container 1: request=%v limit=%v", pod.Spec.Containers[1].Resources.Requests, pod.Spec.Containers[1].Resources.Limits)

		assert.Equal(t.want.cpuR, float64(beforeCpuR), "cpu request before modifiation is wrong")
		assert.Equal(t.want.cpuL, float64(beforeCpuL), "cpu limit before modifiation is wrong")

		assert.Equal(t.want.memR, float64(beforeMemR), "mem request before modifiation is wrong")
		assert.Equal(t.want.memL, float64(beforeMemL), "mem limit before modifiation is wrong")

		assert.Equal(t.want.cpuR, float64(cpuR), "cpu request is wrong")
		assert.Equal(t.want.cpuL, float64(cpuL), "cpu limit is wrong")

		assert.Equal(t.want.memR, float64(memR), "mem request is wrong")
		assert.Equal(t.want.memL, float64(memL), "mem limit is wrong")
	}
}

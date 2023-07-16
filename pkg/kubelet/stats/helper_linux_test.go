/*
Copyright 2023 The Kubernetes Authors.

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

package stats

import (
	"testing"
	"time"

	"github.com/golang/mock/gomock"
	cadvisorapiv1 "github.com/google/cadvisor/info/v1"
	cadvisorapiv2 "github.com/google/cadvisor/info/v2"
	"github.com/stretchr/testify/assert"
	"k8s.io/apimachinery/pkg/types"
	cadvisortest "k8s.io/kubernetes/pkg/kubelet/cadvisor/testing"
	"k8s.io/kubernetes/pkg/kubelet/cm"
)

/* using sample metrics from a v2 node:
 *
 * /# cat /sys/fs/cgroup/memory.stat
 * anon 696201216
 * file 6395928576
 * /# cat /proc/meminfo
 * MemTotal:       16374584 kB
 * MemFree:         8745124 kB
 *
 * (16374584 - 8745124) * 1024 = 7812567040 bytes
 */

func TestAdjustCgroupv2NodeMemoryUsage(t *testing.T) {
	metrics := &cadvisorapiv1.MemoryStats{
		Usage: 7812567040,
		Cache: 6395928576,
		RSS:   696201216,
		Swap:  0,
	}
	cInfo := cadvisorapiv2.ContainerInfo{
		Stats: []*cadvisorapiv2.ContainerStats{
			{
				Memory: metrics,
			},
			{
				Memory: metrics,
			},
		},
	}

	expectedUsage := metrics.Cache + metrics.RSS

	adjustCgroupv2NodeMemoryUsage(&cInfo)

	assert.Equal(t, cInfo.Stats[0].Memory.Usage, expectedUsage)
	assert.Equal(t, cInfo.Stats[0].Memory.Swap, metrics.Swap)
	assert.Equal(t, cInfo.Stats[1].Memory.Usage, expectedUsage)
	assert.Equal(t, cInfo.Stats[1].Memory.Swap, metrics.Swap)
}

func TestGetCgroupInfoUsage(t *testing.T) {
	mockCtrl := gomock.NewController(t)
	defer mockCtrl.Finish()
	var (
		mockCadvisor = cadvisortest.NewMockInterface(mockCtrl)

		sandbox0       = makeFakePodSandbox("sandbox0-name", "sandbox0-uid", "sandbox0-ns", false)
		sandbox0Cgroup = "/" + cm.GetPodCgroupNameSuffix(types.UID(sandbox0.PodSandboxStatus.Metadata.Uid))
	)

	cases := map[string]struct {
		containerName          string
		expectedUsage          uint64
		infos                  map[string]cadvisorapiv2.ContainerInfo
		isCgroup2UnifiedModeFn func() bool
	}{
		"root-v1": {
			containerName: "/",
			expectedUsage: uint64(seedRoot + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/": getTestContainerInfo(seedRoot, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv1ForTest,
		},
		"kubelet-v1": {
			containerName: "/kubelet",
			expectedUsage: uint64(seedKubelet + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/kubelet": getTestContainerInfo(seedKubelet, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv1ForTest,
		},
		"system-v1": {
			containerName: "/system",
			expectedUsage: uint64(seedMisc + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/system": getTestContainerInfo(seedMisc, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv1ForTest,
		},
		"pod-v1": {
			containerName: sandbox0Cgroup,
			expectedUsage: uint64(seedSandbox0 + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				sandbox0Cgroup: getTestContainerInfo(seedSandbox0, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv1ForTest,
		},
		"root-v2": {
			containerName: "/",
			expectedUsage: uint64(seedRoot + offsetMemRSSBytes + seedRoot + offsetMemCacheBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/": getTestContainerInfo(seedRoot, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv2ForTest,
		},
		"kubelet-v2": {
			containerName: "/kubelet",
			expectedUsage: uint64(seedKubelet + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/kubelet": getTestContainerInfo(seedKubelet, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv2ForTest,
		},
		"system-v2": {
			containerName: "/system",
			expectedUsage: uint64(seedMisc + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				"/system": getTestContainerInfo(seedMisc, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv2ForTest,
		},
		"pod-v2": {
			containerName: sandbox0Cgroup,
			expectedUsage: uint64(seedSandbox0 + offsetMemUsageBytes),
			infos: map[string]cadvisorapiv2.ContainerInfo{
				sandbox0Cgroup: getTestContainerInfo(seedSandbox0, "", "", ""),
			},
			isCgroup2UnifiedModeFn: isCgroupv2ForTest,
		},
	}

	for name := range cases {
		name := name
		tc := cases[name]
		t.Run(name, func(t *testing.T) {
			var maxAge *time.Duration
			age := 0 * time.Second
			maxAge = &age

			options := cadvisorapiv2.RequestOptions{
				IdType:    cadvisorapiv2.TypeName,
				Count:     2,
				Recursive: false,
				MaxAge:    maxAge,
			}

			mockCadvisor.EXPECT().ContainerInfoV2(tc.containerName, options).Return(tc.infos, nil)

			isCgroup2UnifiedMode = tc.isCgroup2UnifiedModeFn

			cInfo, err := getCgroupInfo(mockCadvisor, tc.containerName, true)
			assert.NoError(t, err)
			assert.NotNil(t, cInfo)

			assert.Equal(t, cInfo.Stats[0].Memory.Usage, tc.expectedUsage)
		})
	}
}

func isCgroupv1ForTest() bool {
	return false
}

func isCgroupv2ForTest() bool {
	return true
}

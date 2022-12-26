/*
Copyright 2022 The Kubernetes Authors.

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
	"bufio"
	"context"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strconv"
	"strings"

	"github.com/onsi/ginkgo/v2"
	nodev1 "k8s.io/api/node/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	reservedSwapSize      = "256Mi"
	revervedSwapSizeBytes = 2 << 27
)

// Serial because the test updates kubelet configuration.
var _ = SIGDescribe("System reserved swap [LinuxOnly] [Serial]", func() {
	f := framework.NewDefaultFramework("system-reserved-swap")
	ginkgo.Context("With config updated with swap reserved", func() {
		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.FailSwapOn = false
			initialConfig.FeatureGates[string(kubefeatures.NodeSwap)] = true
			initialConfig.MemorySwap = kubeletconfig.MemorySwapConfiguration{
				SwapBehavior: "LimitedSwap",
			}
			if initialConfig.SystemReserved == nil {
				initialConfig.SystemReserved = map[string]string{}
			}
			initialConfig.SystemReserved[string(nodev1.ResourceSwap)] = reservedSwapSize
		})
		runSystemReservedSwapTests(f)
	})
})

func runSystemReservedSwapTests(f *framework.Framework) {
	ginkgo.It("node should not allocate reserved swap size", func() {
		hostMemInfo, err := getHostMemInfo()
		framework.ExpectNoError(err)
		if hostMemInfo.swapTotal <= revervedSwapSizeBytes {
			ginkgo.Skip("skipping test when swap is not enough on host")
		}

		ginkgo.By("by check node status")
		nodeList, err := f.ClientSet.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{})
		framework.ExpectNoError(err)
		// Assuming that there is only one node, because this is a node e2e test.
		framework.ExpectEqual(len(nodeList.Items), 1)
		allocateble := nodev1.Swap(nodeList.Items[0].Status.Allocatable)
		capacity := nodev1.Swap(nodeList.Items[0].Status.Capacity)
		reserved := resource.MustParse(reservedSwapSize)
		reserved.Add(*allocateble)
		framework.ExpectEqual(reserved.Cmp(*capacity), 0)
		// check cgroup limit
		limitsize, unified, err := getCgroupLimit()
		framework.ExpectNoError(err)
		if unified {
			framework.ExpectEqual(hostMemInfo.swapTotal-revervedSwapSizeBytes, limitsize, "total swap - systemreserved swap = cgourpv2 swap limit")
		} else {
			framework.ExpectEqual(hostMemInfo.swapTotal-revervedSwapSizeBytes, limitsize-hostMemInfo.memTotal, "total swap - systemreserved swap = cgroupv1 swap limit")
		}
	})
}

type hostMemInfo struct {
	swapTotal uint64
	memTotal  uint64
}

func getHostMemInfo() (*hostMemInfo, error) {
	var result = &hostMemInfo{}
	file, err := os.Open("/proc/meminfo")
	if err != nil {
		return nil, err
	}
	defer file.Close()
	reader := bufio.NewReader(file)
	for {
		line, _, err := reader.ReadLine()
		if err == io.EOF {
			break
		}
		fields := strings.Split(string(line), ":")
		if len(fields) != 2 {
			continue
		}
		key := strings.TrimSpace(fields[0])
		value := strings.TrimSpace(fields[1])
		value = strings.Replace(value, " kB", "", -1)
		switch key {
		case "MemTotal":
			t, err := strconv.ParseUint(value, 10, 64)
			if err != nil {
				return nil, err
			}
			result.memTotal = t * 1024
		case "SwapTotal":
			t, err := strconv.ParseUint(value, 10, 64)
			if err != nil {
				return nil, err
			}
			result.swapTotal = t * 1024
		}
	}
	return result, nil
}

func getCgroupLimit() (uint64, bool, error) {
	var cgroupfilename string
	unified := IsCgroup2UnifiedMode()
	if unified {
		cgroupfilename = fmt.Sprintf("/sys/fs/cgroup/memory/%s/memory.swap.max", toCgroupFsName(cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)))
	} else {
		cgroupfilename = fmt.Sprintf("/sys/fs/cgroup/memory/%s/memory.memsw.limit_in_bytes", toCgroupFsName(cm.NewCgroupName(cm.RootCgroupName, defaultNodeAllocatableCgroup)))
	}
	bs, err := ioutil.ReadFile(cgroupfilename)
	if err != nil {
		return 0, unified, err
	}
	size, err := strconv.Atoi(strings.TrimSuffix(string(bs), "\n"))
	if err != nil {
		return 0, unified, err
	}
	return uint64(size), unified, nil
}

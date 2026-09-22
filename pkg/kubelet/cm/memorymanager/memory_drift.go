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

package memorymanager

import (
	"math"
	"os"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

const memoryDriftGraceBytes uint64 = 64 * 1024 * 1024

func memoryDriftFromKernelImage(logger klog.Logger, iomemPath string) uint64 {
	iomem, err := os.ReadFile(iomemPath)
	if err != nil {
		logger.Info("Could not read the kernel image size, memory drift tolerance stays off", "path", iomemPath, "err", err)
		return 0
	}
	size := parseKernelImageSize(string(iomem))
	if size == 0 {
		logger.Info("Could not determine the kernel image size, memory drift tolerance stays off", "path", iomemPath)
		return 0
	}
	maxDrift := size + memoryDriftGraceBytes
	logger.V(2).Info("Derived the memory drift bound from the kernel image size", "kernelImageBytes", size, "maxDriftBytes", maxDrift)
	return maxDrift
}

func parseKernelImageSize(iomem string) uint64 {
	minStart, maxEnd := uint64(math.MaxUint64), uint64(0)
	for line := range strings.Lines(iomem) {
		addresses, name, ok := strings.Cut(strings.TrimSpace(line), " : ")
		if !ok || (name != "Kernel code" && name != "Kernel data" && name != "Kernel bss") {
			continue
		}
		startText, endText, ok := strings.Cut(addresses, "-")
		if !ok {
			continue
		}
		start, err1 := strconv.ParseUint(startText, 16, 64)
		end, err2 := strconv.ParseUint(endText, 16, 64)
		if err1 != nil || err2 != nil {
			continue
		}
		minStart = min(minStart, start)
		maxEnd = max(maxEnd, end)
	}
	if maxEnd == 0 {
		return 0
	}
	return maxEnd - minStart + 1
}

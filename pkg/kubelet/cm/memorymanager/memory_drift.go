/*
Copyright 2026 The Kubernetes Authors.

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
	"bufio"
	"io"
	"os"
	"strconv"
	"strings"

	"k8s.io/klog/v2"
)

const memoryDriftGraceBytes uint64 = 64 * 1024 * 1024

const procIomemPath = "/proc/iomem"

func memoryDriftFromKernelImage(logger klog.Logger, iomemPath string) uint64 {
	f, err := os.Open(iomemPath)
	if err != nil {
		logger.Info("Could not read the kernel image size, using the default memory drift bound", "path", iomemPath, "err", err)
		return defaultMaxMemoryDriftBytes
	}
	defer func() {
		_ = f.Close()
	}()

	size, ok := parseKernelImageSize(f)
	if !ok {
		logger.Info("Could not determine the kernel image size, using the default memory drift bound", "path", iomemPath)
		return defaultMaxMemoryDriftBytes
	}
	drift := size + memoryDriftGraceBytes
	logger.V(2).Info("Derived the memory drift bound from the kernel image size", "kernelImageBytes", size, "driftBytes", drift)
	return drift
}

func parseKernelImageSize(r io.Reader) (uint64, bool) {
	var minStart, maxEnd uint64
	found, nonZero := false, false

	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		idx := strings.LastIndexByte(line, ':')
		if idx < 0 {
			continue
		}
		switch strings.TrimSpace(line[idx+1:]) {
		case "Kernel code", "Kernel data", "Kernel bss":
		default:
			continue
		}
		rng := strings.TrimSpace(line[:idx])
		dash := strings.IndexByte(rng, '-')
		if dash < 0 {
			continue
		}
		start, err1 := strconv.ParseUint(strings.TrimSpace(rng[:dash]), 16, 64)
		end, err2 := strconv.ParseUint(strings.TrimSpace(rng[dash+1:]), 16, 64)
		if err1 != nil || err2 != nil || end < start {
			continue
		}
		if start != 0 || end != 0 {
			nonZero = true
		}
		if !found || start < minStart {
			minStart = start
		}
		if !found || end > maxEnd {
			maxEnd = end
		}
		found = true
	}
	if !found || !nonZero {
		return 0, false
	}
	return maxEnd - minStart + 1, true
}

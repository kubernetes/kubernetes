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
	"strings"
	"testing"

	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestParseKernelImageSize(t *testing.T) {
	testCases := []struct {
		description string
		iomem       string
		wantSize    uint64
		wantOK      bool
	}{
		{
			description: "kernel lines with a gap between the sections",
			iomem: "  00100000-b7ffffff : System RAM\n" +
				"    aff800000-b00dfffff : Kernel code\n" +
				"    b01e00000-b0225523f : Kernel data\n" +
				"    b02752000-b02bfffff : Kernel bss\n",
			wantSize: 0xb02bfffff - 0xaff800000 + 1,
			wantOK:   true,
		},
		{
			description: "addresses hidden without CAP_SYS_ADMIN are rejected",
			iomem: "  00000000-00000000 : Kernel code\n" +
				"  00000000-00000000 : Kernel data\n" +
				"  00000000-00000000 : Kernel bss\n",
			wantOK: false,
		},
		{
			description: "no kernel lines are rejected",
			iomem: "  00100000-07ffffff : System RAM\n" +
				"  08000000-0fffffff : PCI Bus 0000:00\n",
			wantOK: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			got, ok := parseKernelImageSize(strings.NewReader(tc.iomem))
			if ok != tc.wantOK {
				t.Fatalf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && got != tc.wantSize {
				t.Fatalf("size = %d, want %d", got, tc.wantSize)
			}
		})
	}
}

func TestMemoryDriftFromKernelImage(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	if got := memoryDriftFromKernelImage(logger, "/does/not/exist/iomem"); got != defaultMaxMemoryDriftBytes {
		t.Fatalf("fallback bound = %d, want %d", got, defaultMaxMemoryDriftBytes)
	}
}

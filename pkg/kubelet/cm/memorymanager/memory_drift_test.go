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
	"os"
	"path/filepath"
	"testing"

	"k8s.io/ktesting"
)

const testIomem = `00000000-00000fff : Reserved
00001000-0009efff : System RAM
00100000-bffdffff : System RAM
fec00000-fec003ff : PNP0003:00
  fec00000-fec003ff : IOAPIC 0
100000000-83fffffff : System RAM
  4d8a00000-4d9c021d1 : Kernel code
  4d9e00000-4da42afff : Kernel rodata
  4da600000-4da8ac1bf : Kernel data
  4dad2e000-4db1fffff : Kernel bss
`

const testIomemKernelImageBytes = 0x4db1fffff - 0x4d8a00000 + 1

const testIomemHiddenAddresses = `00000000-00000000 : System RAM
  00000000-00000000 : Kernel code
  00000000-00000000 : Kernel rodata
  00000000-00000000 : Kernel data
  00000000-00000000 : Kernel bss
`

func TestParseKernelImageSize(t *testing.T) {
	testCases := []struct {
		description  string
		iomem        string
		expectedSize uint64
	}{
		{
			description:  "the span from the start of the kernel code to the end of the kernel bss",
			iomem:        testIomem,
			expectedSize: testIomemKernelImageBytes,
		},
		{
			description:  "addresses hidden from a reader without CAP_SYS_ADMIN",
			iomem:        testIomemHiddenAddresses,
			expectedSize: 0,
		},
		{
			description: "no kernel image sections",
			iomem: `00100000-07ffffff : System RAM
08000000-0fffffff : PCI Bus 0000:00
`,
			expectedSize: 0,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.description, func(t *testing.T) {
			if size := parseKernelImageSize(tc.iomem); size != tc.expectedSize {
				t.Fatalf("size = %d, want %d", size, tc.expectedSize)
			}
		})
	}
}

func TestMemoryDriftFromKernelImage(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)
	dir := t.TempDir()
	iomemPath := filepath.Join(dir, "iomem")
	if err := os.WriteFile(iomemPath, []byte(testIomem), 0o644); err != nil {
		t.Fatalf("could not write %s: %v", iomemPath, err)
	}
	hiddenIomemPath := filepath.Join(dir, "iomem-hidden")
	if err := os.WriteFile(hiddenIomemPath, []byte(testIomemHiddenAddresses), 0o644); err != nil {
		t.Fatalf("could not write %s: %v", hiddenIomemPath, err)
	}

	if maxDrift := memoryDriftFromKernelImage(logger, iomemPath); maxDrift != testIomemKernelImageBytes+memoryDriftGraceBytes {
		t.Fatalf("bound = %d, want %d", maxDrift, testIomemKernelImageBytes+memoryDriftGraceBytes)
	}
	if maxDrift := memoryDriftFromKernelImage(logger, hiddenIomemPath); maxDrift != 0 {
		t.Fatalf("bound for hidden addresses = %d, want 0", maxDrift)
	}
	if maxDrift := memoryDriftFromKernelImage(logger, filepath.Join(dir, "missing")); maxDrift != 0 {
		t.Fatalf("bound for a missing file = %d, want 0", maxDrift)
	}
}

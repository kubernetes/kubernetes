// Copyright The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//go:build linux

package procfs

import (
	"github.com/prometheus/procfs/internal/parsers"
)

// KernelTaintBit represents a single kernel taint flag.
type KernelTaintBit struct {
	// Index is the zero-based bit position in the tainted bitmask.
	Index int
	// Flag is the letter code identifying this taint (e.g. "L" for soft lockup).
	// See https://www.kernel.org/doc/html/latest/admin-guide/tainted-kernels.html
	Flag string
	// Description is a human-readable explanation of what this taint means.
	Description string
	// Set indicates whether this taint flag is currently active.
	Set bool
}

// KernelTainted holds the kernel taint state read from /proc/sys/kernel/tainted.
type KernelTainted struct {
	// Value is the raw integer from /proc/sys/kernel/tainted.
	Value uint64
	// Bits contains each known taint flag and whether it is currently set.
	// Flags are listed in bit-position order (bit 0 first).
	Bits []KernelTaintBit
}

// kernelTaintBitDefs lists all known kernel taint flags in bit-position order.
// Reference: https://www.kernel.org/doc/html/latest/admin-guide/tainted-kernels.html
var kernelTaintBitDefs = []struct {
	flag        string
	description string
}{
	{"P", "Proprietary module was loaded."},
	{"F", "Module was force loaded."},
	{"S", "SMP kernel oops on an officially SMP incapable processor."},
	{"R", "Module was force unloaded."},
	{"M", "Processor reported a Machine Check Exception (MCE)."},
	{"B", "Bad page referenced or some unexpected page flags."},
	{"U", "Taint requested by userspace application."},
	{"D", "Kernel died recently, i.e. there was an OOPS or BUG."},
	{"A", "An ACPI table was overridden by user."},
	{"W", "Kernel issued warning."},
	{"C", "Staging driver was loaded."},
	{"I", "Workaround for bug in platform firmware applied."},
	{"O", "Externally-built (out-of-tree) module was loaded."},
	{"E", "Unsigned module was loaded."},
	{"L", "Soft lockup occurred."},
	{"K", "Kernel has been live patched."},
	{"X", "Auxiliary taint, defined and used by distros."},
	{"T", "The kernel was built with the struct randomization plugin."},
	{"N", "An in-kernel test such as a KUnit test has been run."},
	{"J", "Userspace used a mutating debug operation in fwctl."},
}

// KernelTainted reads /proc/sys/kernel/tainted and returns the raw taint value
// along with each known flag parsed into a KernelTainted struct.
// See https://www.kernel.org/doc/html/latest/admin-guide/tainted-kernels.html
func (fs FS) KernelTainted() (KernelTainted, error) {
	value, err := parsers.SysReadUintFromFile(fs.proc.Path("sys", "kernel", "tainted"))
	if err != nil {
		return KernelTainted{}, err
	}
	return parseTainted(value), nil
}

// parseTainted builds a KernelTainted from the raw taint bitmask value.
func parseTainted(value uint64) KernelTainted {
	bits := make([]KernelTaintBit, len(kernelTaintBitDefs))
	for i, def := range kernelTaintBitDefs {
		bits[i] = KernelTaintBit{
			Index:       i,
			Flag:        def.flag,
			Description: def.description,
			Set:         value&(1<<uint(i)) != 0,
		}
	}
	return KernelTainted{Value: value, Bits: bits}
}

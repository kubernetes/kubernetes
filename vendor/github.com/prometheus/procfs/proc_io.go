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

package procfs

import (
	"fmt"

	"github.com/prometheus/procfs/internal/util"
)

// ProcIO models the content of /proc/<pid>/io.
type ProcIO struct {
	// Chars read.
	RChar uint64
	// Chars written.
	WChar uint64
	// Read syscalls.
	SyscR uint64
	// Write syscalls.
	SyscW uint64
	// Bytes read.
	ReadBytes uint64
	// Bytes written.
	WriteBytes uint64
	// Bytes written, but taking into account truncation. See
	// Documentation/filesystems/proc.txt in the kernel sources for
	// detailed explanation.
	CancelledWriteBytes int64
}

// IO creates a new ProcIO instance from a given Proc instance.
func (p Proc) IO() (ProcIO, error) {
	pio := ProcIO{}

	data, err := util.ReadFileNoStat(p.path("io"))
	if err != nil {
		return pio, err
	}

	ioFormat := "rchar: %d\nwchar: %d\nsyscr: %d\nsyscw: %d\n" +
		"read_bytes: %d\nwrite_bytes: %d\n" +
		"cancelled_write_bytes: %d\n" //nolint:misspell

	_, err = fmt.Sscanf(string(data), ioFormat, &pio.RChar, &pio.WChar, &pio.SyscR,
		&pio.SyscW, &pio.ReadBytes, &pio.WriteBytes, &pio.CancelledWriteBytes)

	return pio, err
}

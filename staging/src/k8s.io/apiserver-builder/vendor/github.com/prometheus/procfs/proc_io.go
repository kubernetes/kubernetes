package procfs

import (
	"fmt"
	"io/ioutil"
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

// NewIO creates a new ProcIO instance from a given Proc instance.
func (p Proc) NewIO() (ProcIO, error) {
	pio := ProcIO{}

	f, err := p.open("io")
	if err != nil {
		return pio, err
	}
	defer f.Close()

	data, err := ioutil.ReadAll(f)
	if err != nil {
		return pio, err
	}

	ioFormat := "rchar: %d\nwchar: %d\nsyscr: %d\nsyscw: %d\n" +
		"read_bytes: %d\nwrite_bytes: %d\n" +
		"cancelled_write_bytes: %d\n"

	_, err = fmt.Sscanf(string(data), ioFormat, &pio.RChar, &pio.WChar, &pio.SyscR,
		&pio.SyscW, &pio.ReadBytes, &pio.WriteBytes, &pio.CancelledWriteBytes)
	if err != nil {
		return pio, err
	}

	return pio, nil
}

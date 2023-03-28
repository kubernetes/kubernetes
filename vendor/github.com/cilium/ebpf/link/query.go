package link

import (
	"fmt"
	"os"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

// QueryOptions defines additional parameters when querying for programs.
type QueryOptions struct {
	// Path can be a path to a cgroup, netns or LIRC2 device
	Path string
	// Attach specifies the AttachType of the programs queried for
	Attach ebpf.AttachType
	// QueryFlags are flags for BPF_PROG_QUERY, e.g. BPF_F_QUERY_EFFECTIVE
	QueryFlags uint32
}

// QueryPrograms retrieves ProgramIDs associated with the AttachType.
//
// It only returns IDs of programs that were attached using PROG_ATTACH and not bpf_link.
// Returns (nil, nil) if there are no programs attached to the queried kernel resource.
// Calling QueryPrograms on a kernel missing PROG_QUERY will result in ErrNotSupported.
func QueryPrograms(opts QueryOptions) ([]ebpf.ProgramID, error) {
	if haveProgQuery() != nil {
		return nil, fmt.Errorf("can't query program IDs: %w", ErrNotSupported)
	}

	f, err := os.Open(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("can't open file: %s", err)
	}
	defer f.Close()

	// query the number of programs to allocate correct slice size
	attr := sys.ProgQueryAttr{
		TargetFd:   uint32(f.Fd()),
		AttachType: sys.AttachType(opts.Attach),
		QueryFlags: opts.QueryFlags,
	}
	if err := sys.ProgQuery(&attr); err != nil {
		return nil, fmt.Errorf("can't query program count: %w", err)
	}

	// return nil if no progs are attached
	if attr.ProgCount == 0 {
		return nil, nil
	}

	// we have at least one prog, so we query again
	progIds := make([]ebpf.ProgramID, attr.ProgCount)
	attr.ProgIds = sys.NewPointer(unsafe.Pointer(&progIds[0]))
	attr.ProgCount = uint32(len(progIds))
	if err := sys.ProgQuery(&attr); err != nil {
		return nil, fmt.Errorf("can't query program IDs: %w", err)
	}

	return progIds, nil

}

package link

import (
	"fmt"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

// QueryOptions defines additional parameters when querying for programs.
type QueryOptions struct {
	// Target to query. This is usually a file descriptor but may refer to
	// something else based on the attach type.
	Target int
	// Attach specifies the AttachType of the programs queried for
	Attach ebpf.AttachType
	// QueryFlags are flags for BPF_PROG_QUERY, e.g. BPF_F_QUERY_EFFECTIVE
	QueryFlags uint32
}

// QueryResult describes which programs and links are active.
type QueryResult struct {
	// List of attached programs.
	Programs []AttachedProgram

	// Incremented by one every time the set of attached programs changes.
	// May be zero if not supported by the [ebpf.AttachType].
	Revision uint64
}

// HaveLinkInfo returns true if the kernel supports querying link information
// for a particular [ebpf.AttachType].
func (qr *QueryResult) HaveLinkInfo() bool {
	return qr.Revision > 0
}

type AttachedProgram struct {
	ID     ebpf.ProgramID
	linkID ID
}

// LinkID returns the ID associated with the program.
//
// Returns 0, false if the kernel doesn't support retrieving the ID or if the
// program wasn't attached via a link. See [QueryResult.HaveLinkInfo] if you
// need to tell the two apart.
func (ap *AttachedProgram) LinkID() (ID, bool) {
	return ap.linkID, ap.linkID != 0
}

// QueryPrograms retrieves a list of programs for the given AttachType.
//
// Returns a slice of attached programs, which may be empty.
// revision counts how many times the set of attached programs has changed and
// may be zero if not supported by the [ebpf.AttachType].
// Returns ErrNotSupportd on a kernel without BPF_PROG_QUERY
func QueryPrograms(opts QueryOptions) (*QueryResult, error) {
	// query the number of programs to allocate correct slice size
	attr := sys.ProgQueryAttr{
		TargetFdOrIfindex: uint32(opts.Target),
		AttachType:        sys.AttachType(opts.Attach),
		QueryFlags:        opts.QueryFlags,
	}
	err := sys.ProgQuery(&attr)
	if err != nil {
		if haveFeatErr := haveProgQuery(); haveFeatErr != nil {
			return nil, fmt.Errorf("query programs: %w", haveFeatErr)
		}
		return nil, fmt.Errorf("query programs: %w", err)
	}
	if attr.Count == 0 {
		return &QueryResult{Revision: attr.Revision}, nil
	}

	// The minimum bpf_mprog revision is 1, so we can use the field to detect
	// whether the attach type supports link ids.
	haveLinkIDs := attr.Revision != 0

	count := attr.Count
	progIds := make([]ebpf.ProgramID, count)
	attr = sys.ProgQueryAttr{
		TargetFdOrIfindex: uint32(opts.Target),
		AttachType:        sys.AttachType(opts.Attach),
		QueryFlags:        opts.QueryFlags,
		Count:             count,
		ProgIds:           sys.NewPointer(unsafe.Pointer(&progIds[0])),
	}

	var linkIds []ID
	if haveLinkIDs {
		linkIds = make([]ID, count)
		attr.LinkIds = sys.NewPointer(unsafe.Pointer(&linkIds[0]))
	}

	if err := sys.ProgQuery(&attr); err != nil {
		return nil, fmt.Errorf("query programs: %w", err)
	}

	// NB: attr.Count might have changed between the two syscalls.
	var programs []AttachedProgram
	for i, id := range progIds[:attr.Count] {
		ap := AttachedProgram{ID: id}
		if haveLinkIDs {
			ap.linkID = linkIds[i]
		}
		programs = append(programs, ap)
	}

	return &QueryResult{programs, attr.Revision}, nil
}

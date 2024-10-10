package link

import (
	"fmt"
	"runtime"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type RawAttachProgramOptions struct {
	// Target to query. This is usually a file descriptor but may refer to
	// something else based on the attach type.
	Target int
	// Program to attach.
	Program *ebpf.Program
	// Attach must match the attach type of Program.
	Attach ebpf.AttachType
	// Attach relative to an anchor. Optional.
	Anchor Anchor
	// Flags control the attach behaviour. Specify an Anchor instead of
	// F_LINK, F_ID, F_BEFORE, F_AFTER and F_REPLACE. Optional.
	Flags uint32
	// Only attach if the internal revision matches the given value.
	ExpectedRevision uint64
}

// RawAttachProgram is a low level wrapper around BPF_PROG_ATTACH.
//
// You should use one of the higher level abstractions available in this
// package if possible.
func RawAttachProgram(opts RawAttachProgramOptions) error {
	if opts.Flags&anchorFlags != 0 {
		return fmt.Errorf("disallowed flags: use Anchor to specify attach target")
	}

	attr := sys.ProgAttachAttr{
		TargetFdOrIfindex: uint32(opts.Target),
		AttachBpfFd:       uint32(opts.Program.FD()),
		AttachType:        uint32(opts.Attach),
		AttachFlags:       uint32(opts.Flags),
		ExpectedRevision:  opts.ExpectedRevision,
	}

	if opts.Anchor != nil {
		fdOrID, flags, err := opts.Anchor.anchor()
		if err != nil {
			return fmt.Errorf("attach program: %w", err)
		}

		if flags == sys.BPF_F_REPLACE {
			// Ensure that replacing a program works on old kernels.
			attr.ReplaceBpfFd = fdOrID
		} else {
			attr.RelativeFdOrId = fdOrID
			attr.AttachFlags |= flags
		}
	}

	if err := sys.ProgAttach(&attr); err != nil {
		if haveFeatErr := haveProgAttach(); haveFeatErr != nil {
			return haveFeatErr
		}
		return fmt.Errorf("attach program: %w", err)
	}
	runtime.KeepAlive(opts.Program)

	return nil
}

type RawDetachProgramOptions RawAttachProgramOptions

// RawDetachProgram is a low level wrapper around BPF_PROG_DETACH.
//
// You should use one of the higher level abstractions available in this
// package if possible.
func RawDetachProgram(opts RawDetachProgramOptions) error {
	if opts.Flags&anchorFlags != 0 {
		return fmt.Errorf("disallowed flags: use Anchor to specify attach target")
	}

	attr := sys.ProgDetachAttr{
		TargetFdOrIfindex: uint32(opts.Target),
		AttachBpfFd:       uint32(opts.Program.FD()),
		AttachType:        uint32(opts.Attach),
		ExpectedRevision:  opts.ExpectedRevision,
	}

	if opts.Anchor != nil {
		fdOrID, flags, err := opts.Anchor.anchor()
		if err != nil {
			return fmt.Errorf("detach program: %w", err)
		}

		attr.RelativeFdOrId = fdOrID
		attr.AttachFlags |= flags
	}

	if err := sys.ProgDetach(&attr); err != nil {
		if haveFeatErr := haveProgAttach(); haveFeatErr != nil {
			return haveFeatErr
		}
		return fmt.Errorf("can't detach program: %w", err)
	}

	return nil
}

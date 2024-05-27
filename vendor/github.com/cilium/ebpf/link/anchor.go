package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

const anchorFlags = sys.BPF_F_REPLACE |
	sys.BPF_F_BEFORE |
	sys.BPF_F_AFTER |
	sys.BPF_F_ID |
	sys.BPF_F_LINK_MPROG

// Anchor is a reference to a link or program.
//
// It is used to describe where an attachment or detachment should take place
// for link types which support multiple attachment.
type Anchor interface {
	// anchor returns an fd or ID and a set of flags.
	//
	// By default fdOrID is taken to reference a program, but BPF_F_LINK_MPROG
	// changes this to refer to a link instead.
	//
	// BPF_F_BEFORE, BPF_F_AFTER, BPF_F_REPLACE modify where a link or program
	// is attached. The default behaviour if none of these flags is specified
	// matches BPF_F_AFTER.
	anchor() (fdOrID, flags uint32, _ error)
}

type firstAnchor struct{}

func (firstAnchor) anchor() (fdOrID, flags uint32, _ error) {
	return 0, sys.BPF_F_BEFORE, nil
}

// Head is the position before all other programs or links.
func Head() Anchor {
	return firstAnchor{}
}

type lastAnchor struct{}

func (lastAnchor) anchor() (fdOrID, flags uint32, _ error) {
	return 0, sys.BPF_F_AFTER, nil
}

// Tail is the position after all other programs or links.
func Tail() Anchor {
	return lastAnchor{}
}

// Before is the position just in front of target.
func BeforeLink(target Link) Anchor {
	return anchor{target, sys.BPF_F_BEFORE}
}

// After is the position just after target.
func AfterLink(target Link) Anchor {
	return anchor{target, sys.BPF_F_AFTER}
}

// Before is the position just in front of target.
func BeforeLinkByID(target ID) Anchor {
	return anchor{target, sys.BPF_F_BEFORE}
}

// After is the position just after target.
func AfterLinkByID(target ID) Anchor {
	return anchor{target, sys.BPF_F_AFTER}
}

// Before is the position just in front of target.
func BeforeProgram(target *ebpf.Program) Anchor {
	return anchor{target, sys.BPF_F_BEFORE}
}

// After is the position just after target.
func AfterProgram(target *ebpf.Program) Anchor {
	return anchor{target, sys.BPF_F_AFTER}
}

// Replace the target itself.
func ReplaceProgram(target *ebpf.Program) Anchor {
	return anchor{target, sys.BPF_F_REPLACE}
}

// Before is the position just in front of target.
func BeforeProgramByID(target ebpf.ProgramID) Anchor {
	return anchor{target, sys.BPF_F_BEFORE}
}

// After is the position just after target.
func AfterProgramByID(target ebpf.ProgramID) Anchor {
	return anchor{target, sys.BPF_F_AFTER}
}

// Replace the target itself.
func ReplaceProgramByID(target ebpf.ProgramID) Anchor {
	return anchor{target, sys.BPF_F_REPLACE}
}

type anchor struct {
	target   any
	position uint32
}

func (ap anchor) anchor() (fdOrID, flags uint32, _ error) {
	var typeFlag uint32
	switch target := ap.target.(type) {
	case *ebpf.Program:
		fd := target.FD()
		if fd < 0 {
			return 0, 0, sys.ErrClosedFd
		}
		fdOrID = uint32(fd)
		typeFlag = 0
	case ebpf.ProgramID:
		fdOrID = uint32(target)
		typeFlag = sys.BPF_F_ID
	case interface{ FD() int }:
		fd := target.FD()
		if fd < 0 {
			return 0, 0, sys.ErrClosedFd
		}
		fdOrID = uint32(fd)
		typeFlag = sys.BPF_F_LINK_MPROG
	case ID:
		fdOrID = uint32(target)
		typeFlag = sys.BPF_F_LINK_MPROG | sys.BPF_F_ID
	default:
		return 0, 0, fmt.Errorf("invalid target %T", ap.target)
	}

	return fdOrID, ap.position | typeFlag, nil
}

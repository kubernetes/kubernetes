package link

import (
	"errors"
	"fmt"
	"os"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type cgroupAttachFlags uint32

const (
	// Allow programs attached to sub-cgroups to override the verdict of this
	// program.
	flagAllowOverride cgroupAttachFlags = 1 << iota
	// Allow attaching multiple programs to the cgroup. Only works if the cgroup
	// has zero or more programs attached using the Multi flag. Implies override.
	flagAllowMulti
	// Set automatically by progAttachCgroup.Update(). Used for updating a
	// specific given program attached in multi-mode.
	flagReplace
)

type CgroupOptions struct {
	// Path to a cgroupv2 folder.
	Path string
	// One of the AttachCgroup* constants
	Attach ebpf.AttachType
	// Program must be of type CGroup*, and the attach type must match Attach.
	Program *ebpf.Program
}

// AttachCgroup links a BPF program to a cgroup.
//
// If the running kernel doesn't support bpf_link, attempts to emulate its
// semantics using the legacy PROG_ATTACH mechanism. If bpf_link is not
// available, the returned [Link] will not support pinning to bpffs.
//
// If you need more control over attachment flags or the attachment mechanism
// used, look at [RawAttachProgram] and [AttachRawLink] instead.
func AttachCgroup(opts CgroupOptions) (cg Link, err error) {
	cgroup, err := os.Open(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("can't open cgroup: %s", err)
	}
	defer func() {
		if _, ok := cg.(*progAttachCgroup); ok {
			// Skip closing the cgroup handle if we return a valid progAttachCgroup,
			// where the handle is retained to implement Update().
			return
		}
		cgroup.Close()
	}()

	cg, err = newLinkCgroup(cgroup, opts.Attach, opts.Program)
	if err == nil {
		return cg, nil
	}

	if errors.Is(err, ErrNotSupported) {
		cg, err = newProgAttachCgroup(cgroup, opts.Attach, opts.Program, flagAllowMulti)
	}
	if errors.Is(err, ErrNotSupported) {
		cg, err = newProgAttachCgroup(cgroup, opts.Attach, opts.Program, flagAllowOverride)
	}
	if err != nil {
		return nil, err
	}

	return cg, nil
}

type progAttachCgroup struct {
	cgroup     *os.File
	current    *ebpf.Program
	attachType ebpf.AttachType
	flags      cgroupAttachFlags
}

var _ Link = (*progAttachCgroup)(nil)

func (cg *progAttachCgroup) isLink() {}

// newProgAttachCgroup attaches prog to cgroup using BPF_PROG_ATTACH.
// cgroup and prog are retained by [progAttachCgroup].
func newProgAttachCgroup(cgroup *os.File, attach ebpf.AttachType, prog *ebpf.Program, flags cgroupAttachFlags) (*progAttachCgroup, error) {
	if flags&flagAllowMulti > 0 {
		if err := haveProgAttachReplace(); err != nil {
			return nil, fmt.Errorf("can't support multiple programs: %w", err)
		}
	}

	// Use a program handle that cannot be closed by the caller.
	clone, err := prog.Clone()
	if err != nil {
		return nil, err
	}

	err = RawAttachProgram(RawAttachProgramOptions{
		Target:  int(cgroup.Fd()),
		Program: clone,
		Flags:   uint32(flags),
		Attach:  attach,
	})
	if err != nil {
		clone.Close()
		return nil, fmt.Errorf("cgroup: %w", err)
	}

	return &progAttachCgroup{cgroup, clone, attach, flags}, nil
}

func (cg *progAttachCgroup) Close() error {
	defer cg.cgroup.Close()
	defer cg.current.Close()

	err := RawDetachProgram(RawDetachProgramOptions{
		Target:  int(cg.cgroup.Fd()),
		Program: cg.current,
		Attach:  cg.attachType,
	})
	if err != nil {
		return fmt.Errorf("close cgroup: %s", err)
	}
	return nil
}

func (cg *progAttachCgroup) Update(prog *ebpf.Program) error {
	new, err := prog.Clone()
	if err != nil {
		return err
	}

	args := RawAttachProgramOptions{
		Target:  int(cg.cgroup.Fd()),
		Program: prog,
		Attach:  cg.attachType,
		Flags:   uint32(cg.flags),
	}

	if cg.flags&flagAllowMulti > 0 {
		// Atomically replacing multiple programs requires at least
		// 5.5 (commit 7dd68b3279f17921 "bpf: Support replacing cgroup-bpf
		// program in MULTI mode")
		args.Anchor = ReplaceProgram(cg.current)
	}

	if err := RawAttachProgram(args); err != nil {
		new.Close()
		return fmt.Errorf("can't update cgroup: %s", err)
	}

	cg.current.Close()
	cg.current = new
	return nil
}

func (cg *progAttachCgroup) Pin(string) error {
	return fmt.Errorf("can't pin cgroup: %w", ErrNotSupported)
}

func (cg *progAttachCgroup) Unpin() error {
	return fmt.Errorf("can't unpin cgroup: %w", ErrNotSupported)
}

func (cg *progAttachCgroup) Info() (*Info, error) {
	return nil, fmt.Errorf("can't get cgroup info: %w", ErrNotSupported)
}

type linkCgroup struct {
	RawLink
}

var _ Link = (*linkCgroup)(nil)

// newLinkCgroup attaches prog to cgroup using BPF_LINK_CREATE.
func newLinkCgroup(cgroup *os.File, attach ebpf.AttachType, prog *ebpf.Program) (*linkCgroup, error) {
	link, err := AttachRawLink(RawLinkOptions{
		Target:  int(cgroup.Fd()),
		Program: prog,
		Attach:  attach,
	})
	if err != nil {
		return nil, err
	}

	return &linkCgroup{*link}, err
}

func (cg *linkCgroup) Info() (*Info, error) {
	var info sys.CgroupLinkInfo
	if err := sys.ObjInfo(cg.fd, &info); err != nil {
		return nil, fmt.Errorf("cgroup link info: %s", err)
	}
	extra := &CgroupInfo{
		CgroupId:   info.CgroupId,
		AttachType: info.AttachType,
	}

	return &Info{
		info.Type,
		info.Id,
		ebpf.ProgramID(info.ProgId),
		extra,
	}, nil
}

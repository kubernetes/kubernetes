package link

import (
	"errors"
	"fmt"
	"os"

	"github.com/cilium/ebpf"
)

type cgroupAttachFlags uint32

// cgroup attach flags
const (
	flagAllowOverride cgroupAttachFlags = 1 << iota
	flagAllowMulti
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
func AttachCgroup(opts CgroupOptions) (Link, error) {
	cgroup, err := os.Open(opts.Path)
	if err != nil {
		return nil, fmt.Errorf("can't open cgroup: %s", err)
	}

	clone, err := opts.Program.Clone()
	if err != nil {
		cgroup.Close()
		return nil, err
	}

	var cg Link
	cg, err = newLinkCgroup(cgroup, opts.Attach, clone)
	if errors.Is(err, ErrNotSupported) {
		cg, err = newProgAttachCgroup(cgroup, opts.Attach, clone, flagAllowMulti)
	}
	if errors.Is(err, ErrNotSupported) {
		cg, err = newProgAttachCgroup(cgroup, opts.Attach, clone, flagAllowOverride)
	}
	if err != nil {
		cgroup.Close()
		clone.Close()
		return nil, err
	}

	return cg, nil
}

// LoadPinnedCgroup loads a pinned cgroup from a bpffs.
func LoadPinnedCgroup(fileName string) (Link, error) {
	link, err := LoadPinnedRawLink(fileName)
	if err != nil {
		return nil, err
	}

	return &linkCgroup{link}, nil
}

type progAttachCgroup struct {
	cgroup     *os.File
	current    *ebpf.Program
	attachType ebpf.AttachType
	flags      cgroupAttachFlags
}

var _ Link = (*progAttachCgroup)(nil)

func (cg *progAttachCgroup) isLink() {}

func newProgAttachCgroup(cgroup *os.File, attach ebpf.AttachType, prog *ebpf.Program, flags cgroupAttachFlags) (*progAttachCgroup, error) {
	if flags&flagAllowMulti > 0 {
		if err := haveProgAttachReplace(); err != nil {
			return nil, fmt.Errorf("can't support multiple programs: %w", err)
		}
	}

	err := RawAttachProgram(RawAttachProgramOptions{
		Target:  int(cgroup.Fd()),
		Program: prog,
		Flags:   uint32(flags),
		Attach:  attach,
	})
	if err != nil {
		return nil, fmt.Errorf("cgroup: %w", err)
	}

	return &progAttachCgroup{cgroup, prog, attach, flags}, nil
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
		args.Flags |= uint32(flagReplace)
		args.Replace = cg.current
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

type linkCgroup struct {
	*RawLink
}

var _ Link = (*linkCgroup)(nil)

func (cg *linkCgroup) isLink() {}

func newLinkCgroup(cgroup *os.File, attach ebpf.AttachType, prog *ebpf.Program) (*linkCgroup, error) {
	link, err := AttachRawLink(RawLinkOptions{
		Target:  int(cgroup.Fd()),
		Program: prog,
		Attach:  attach,
	})
	if err != nil {
		return nil, err
	}

	return &linkCgroup{link}, err
}

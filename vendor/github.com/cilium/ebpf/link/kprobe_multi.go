package link

import (
	"errors"
	"fmt"
	"os"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// KprobeMultiOptions defines additional parameters that will be used
// when opening a KprobeMulti Link.
type KprobeMultiOptions struct {
	// Symbols takes a list of kernel symbol names to attach an ebpf program to.
	//
	// Mutually exclusive with Addresses.
	Symbols []string

	// Addresses takes a list of kernel symbol addresses in case they can not
	// be referred to by name.
	//
	// Note that only start addresses can be specified, since the fprobe API
	// limits the attach point to the function entry or return.
	//
	// Mutually exclusive with Symbols.
	Addresses []uintptr

	// Cookies specifies arbitrary values that can be fetched from an eBPF
	// program via `bpf_get_attach_cookie()`.
	//
	// If set, its length should be equal to the length of Symbols or Addresses.
	// Each Cookie is assigned to the Symbol or Address specified at the
	// corresponding slice index.
	Cookies []uint64
}

// KprobeMulti attaches the given eBPF program to the entry point of a given set
// of kernel symbols.
//
// The difference with Kprobe() is that multi-kprobe accomplishes this in a
// single system call, making it significantly faster than attaching many
// probes one at a time.
//
// Requires at least Linux 5.18.
func KprobeMulti(prog *ebpf.Program, opts KprobeMultiOptions) (Link, error) {
	return kprobeMulti(prog, opts, 0)
}

// KretprobeMulti attaches the given eBPF program to the return point of a given
// set of kernel symbols.
//
// The difference with Kretprobe() is that multi-kprobe accomplishes this in a
// single system call, making it significantly faster than attaching many
// probes one at a time.
//
// Requires at least Linux 5.18.
func KretprobeMulti(prog *ebpf.Program, opts KprobeMultiOptions) (Link, error) {
	return kprobeMulti(prog, opts, unix.BPF_F_KPROBE_MULTI_RETURN)
}

func kprobeMulti(prog *ebpf.Program, opts KprobeMultiOptions, flags uint32) (Link, error) {
	if prog == nil {
		return nil, errors.New("cannot attach a nil program")
	}

	syms := uint32(len(opts.Symbols))
	addrs := uint32(len(opts.Addresses))
	cookies := uint32(len(opts.Cookies))

	if syms == 0 && addrs == 0 {
		return nil, fmt.Errorf("one of Symbols or Addresses is required: %w", errInvalidInput)
	}
	if syms != 0 && addrs != 0 {
		return nil, fmt.Errorf("Symbols and Addresses are mutually exclusive: %w", errInvalidInput)
	}
	if cookies > 0 && cookies != syms && cookies != addrs {
		return nil, fmt.Errorf("Cookies must be exactly Symbols or Addresses in length: %w", errInvalidInput)
	}

	if err := haveBPFLinkKprobeMulti(); err != nil {
		return nil, err
	}

	attr := &sys.LinkCreateKprobeMultiAttr{
		ProgFd:           uint32(prog.FD()),
		AttachType:       sys.BPF_TRACE_KPROBE_MULTI,
		KprobeMultiFlags: flags,
	}

	switch {
	case syms != 0:
		attr.Count = syms
		attr.Syms = sys.NewStringSlicePointer(opts.Symbols)

	case addrs != 0:
		attr.Count = addrs
		attr.Addrs = sys.NewPointer(unsafe.Pointer(&opts.Addresses[0]))
	}

	if cookies != 0 {
		attr.Cookies = sys.NewPointer(unsafe.Pointer(&opts.Cookies[0]))
	}

	fd, err := sys.LinkCreateKprobeMulti(attr)
	if errors.Is(err, unix.ESRCH) {
		return nil, fmt.Errorf("couldn't find one or more symbols: %w", os.ErrNotExist)
	}
	if errors.Is(err, unix.EINVAL) {
		return nil, fmt.Errorf("%w (missing kernel symbol or prog's AttachType not AttachTraceKprobeMulti?)", err)
	}
	if err != nil {
		return nil, err
	}

	return &kprobeMultiLink{RawLink{fd, ""}}, nil
}

type kprobeMultiLink struct {
	RawLink
}

var _ Link = (*kprobeMultiLink)(nil)

func (kml *kprobeMultiLink) Update(prog *ebpf.Program) error {
	return fmt.Errorf("update kprobe_multi: %w", ErrNotSupported)
}

func (kml *kprobeMultiLink) Pin(string) error {
	return fmt.Errorf("pin kprobe_multi: %w", ErrNotSupported)
}

func (kml *kprobeMultiLink) Unpin() error {
	return fmt.Errorf("unpin kprobe_multi: %w", ErrNotSupported)
}

var haveBPFLinkKprobeMulti = internal.NewFeatureTest("bpf_link_kprobe_multi", "5.18", func() error {
	prog, err := ebpf.NewProgram(&ebpf.ProgramSpec{
		Name: "probe_kpm_link",
		Type: ebpf.Kprobe,
		Instructions: asm.Instructions{
			asm.Mov.Imm(asm.R0, 0),
			asm.Return(),
		},
		AttachType: ebpf.AttachTraceKprobeMulti,
		License:    "MIT",
	})
	if errors.Is(err, unix.E2BIG) {
		// Kernel doesn't support AttachType field.
		return internal.ErrNotSupported
	}
	if err != nil {
		return err
	}
	defer prog.Close()

	fd, err := sys.LinkCreateKprobeMulti(&sys.LinkCreateKprobeMultiAttr{
		ProgFd:     uint32(prog.FD()),
		AttachType: sys.BPF_TRACE_KPROBE_MULTI,
		Count:      1,
		Syms:       sys.NewStringSlicePointer([]string{"vprintk"}),
	})
	switch {
	case errors.Is(err, unix.EINVAL):
		return internal.ErrNotSupported
	// If CONFIG_FPROBE isn't set.
	case errors.Is(err, unix.EOPNOTSUPP):
		return internal.ErrNotSupported
	case err != nil:
		return err
	}

	fd.Close()

	return nil
})

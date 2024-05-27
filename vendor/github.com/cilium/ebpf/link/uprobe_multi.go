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

// UprobeMultiOptions defines additional parameters that will be used
// when opening a UprobeMulti Link.
type UprobeMultiOptions struct {
	// Symbol addresses. If set, overrides the addresses eventually parsed from
	// the executable. Mutually exclusive with UprobeMulti's symbols argument.
	Addresses []uint64

	// Offsets into functions provided by UprobeMulti's symbols argument.
	// For example: to set uprobes to main+5 and _start+10, call UprobeMulti
	// with:
	//     symbols: "main", "_start"
	//     opt.Offsets: 5, 10
	Offsets []uint64

	// Optional list of associated ref counter offsets.
	RefCtrOffsets []uint64

	// Optional list of associated BPF cookies.
	Cookies []uint64

	// Only set the uprobe_multi link on the given process ID, zero PID means
	// system-wide.
	PID uint32
}

func (ex *Executable) UprobeMulti(symbols []string, prog *ebpf.Program, opts *UprobeMultiOptions) (Link, error) {
	return ex.uprobeMulti(symbols, prog, opts, 0)
}

func (ex *Executable) UretprobeMulti(symbols []string, prog *ebpf.Program, opts *UprobeMultiOptions) (Link, error) {

	// The return probe is not limited for symbols entry, so there's no special
	// setup for return uprobes (other than the extra flag). The symbols, opts.Offsets
	// and opts.Addresses arrays follow the same logic as for entry uprobes.
	return ex.uprobeMulti(symbols, prog, opts, unix.BPF_F_UPROBE_MULTI_RETURN)
}

func (ex *Executable) uprobeMulti(symbols []string, prog *ebpf.Program, opts *UprobeMultiOptions, flags uint32) (Link, error) {
	if prog == nil {
		return nil, errors.New("cannot attach a nil program")
	}

	if opts == nil {
		opts = &UprobeMultiOptions{}
	}

	addresses, err := ex.addresses(symbols, opts.Addresses, opts.Offsets)
	if err != nil {
		return nil, err
	}

	addrs := len(addresses)
	cookies := len(opts.Cookies)
	refCtrOffsets := len(opts.RefCtrOffsets)

	if addrs == 0 {
		return nil, fmt.Errorf("Addresses are required: %w", errInvalidInput)
	}
	if refCtrOffsets > 0 && refCtrOffsets != addrs {
		return nil, fmt.Errorf("RefCtrOffsets must be exactly Addresses in length: %w", errInvalidInput)
	}
	if cookies > 0 && cookies != addrs {
		return nil, fmt.Errorf("Cookies must be exactly Addresses in length: %w", errInvalidInput)
	}

	attr := &sys.LinkCreateUprobeMultiAttr{
		Path:             sys.NewStringPointer(ex.path),
		ProgFd:           uint32(prog.FD()),
		AttachType:       sys.BPF_TRACE_UPROBE_MULTI,
		UprobeMultiFlags: flags,
		Count:            uint32(addrs),
		Offsets:          sys.NewPointer(unsafe.Pointer(&addresses[0])),
		Pid:              opts.PID,
	}

	if refCtrOffsets != 0 {
		attr.RefCtrOffsets = sys.NewPointer(unsafe.Pointer(&opts.RefCtrOffsets[0]))
	}
	if cookies != 0 {
		attr.Cookies = sys.NewPointer(unsafe.Pointer(&opts.Cookies[0]))
	}

	fd, err := sys.LinkCreateUprobeMulti(attr)
	if errors.Is(err, unix.ESRCH) {
		return nil, fmt.Errorf("%w (specified pid not found?)", os.ErrNotExist)
	}
	if errors.Is(err, unix.EINVAL) {
		return nil, fmt.Errorf("%w (missing symbol or prog's AttachType not AttachTraceUprobeMulti?)", err)
	}

	if err != nil {
		if haveFeatErr := haveBPFLinkUprobeMulti(); haveFeatErr != nil {
			return nil, haveFeatErr
		}
		return nil, err
	}

	return &uprobeMultiLink{RawLink{fd, ""}}, nil
}

func (ex *Executable) addresses(symbols []string, addresses, offsets []uint64) ([]uint64, error) {
	n := len(symbols)
	if n == 0 {
		n = len(addresses)
	}

	if n == 0 {
		return nil, fmt.Errorf("%w: neither symbols nor addresses given", errInvalidInput)
	}

	if symbols != nil && len(symbols) != n {
		return nil, fmt.Errorf("%w: have %d symbols but want %d", errInvalidInput, len(symbols), n)
	}

	if addresses != nil && len(addresses) != n {
		return nil, fmt.Errorf("%w: have %d addresses but want %d", errInvalidInput, len(addresses), n)
	}

	if offsets != nil && len(offsets) != n {
		return nil, fmt.Errorf("%w: have %d offsets but want %d", errInvalidInput, len(offsets), n)
	}

	results := make([]uint64, 0, n)
	for i := 0; i < n; i++ {
		var sym string
		if symbols != nil {
			sym = symbols[i]
		}

		var addr, off uint64
		if addresses != nil {
			addr = addresses[i]
		}

		if offsets != nil {
			off = offsets[i]
		}

		result, err := ex.address(sym, addr, off)
		if err != nil {
			return nil, err
		}

		results = append(results, result)
	}

	return results, nil
}

type uprobeMultiLink struct {
	RawLink
}

var _ Link = (*uprobeMultiLink)(nil)

func (kml *uprobeMultiLink) Update(prog *ebpf.Program) error {
	return fmt.Errorf("update uprobe_multi: %w", ErrNotSupported)
}

func (kml *uprobeMultiLink) Pin(string) error {
	return fmt.Errorf("pin uprobe_multi: %w", ErrNotSupported)
}

func (kml *uprobeMultiLink) Unpin() error {
	return fmt.Errorf("unpin uprobe_multi: %w", ErrNotSupported)
}

var haveBPFLinkUprobeMulti = internal.NewFeatureTest("bpf_link_uprobe_multi", "6.6", func() error {
	prog, err := ebpf.NewProgram(&ebpf.ProgramSpec{
		Name: "probe_upm_link",
		Type: ebpf.Kprobe,
		Instructions: asm.Instructions{
			asm.Mov.Imm(asm.R0, 0),
			asm.Return(),
		},
		AttachType: ebpf.AttachTraceUprobeMulti,
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

	// We try to create uprobe multi link on '/' path which results in
	// error with -EBADF in case uprobe multi link is supported.
	fd, err := sys.LinkCreateUprobeMulti(&sys.LinkCreateUprobeMultiAttr{
		ProgFd:     uint32(prog.FD()),
		AttachType: sys.BPF_TRACE_UPROBE_MULTI,
		Path:       sys.NewStringPointer("/"),
		Offsets:    sys.NewPointer(unsafe.Pointer(&[]uint64{0})),
		Count:      1,
	})
	switch {
	case errors.Is(err, unix.EBADF):
		return nil
	case errors.Is(err, unix.EINVAL):
		return internal.ErrNotSupported
	case err != nil:
		return err
	}

	// should not happen
	fd.Close()
	return errors.New("successfully attached uprobe_multi to /, kernel bug?")
})

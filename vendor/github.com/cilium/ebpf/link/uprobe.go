package link

import (
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"sync"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/tracefs"
)

var (
	uprobeRefCtrOffsetPMUPath = "/sys/bus/event_source/devices/uprobe/format/ref_ctr_offset"
	// elixir.bootlin.com/linux/v5.15-rc7/source/kernel/events/core.c#L9799
	uprobeRefCtrOffsetShift = 32
	haveRefCtrOffsetPMU     = internal.NewFeatureTest("RefCtrOffsetPMU", "4.20", func() error {
		_, err := os.Stat(uprobeRefCtrOffsetPMUPath)
		if err != nil {
			return internal.ErrNotSupported
		}
		return nil
	})

	// ErrNoSymbol indicates that the given symbol was not found
	// in the ELF symbols table.
	ErrNoSymbol = errors.New("not found")
)

// Executable defines an executable program on the filesystem.
type Executable struct {
	// Path of the executable on the filesystem.
	path string
	// Parsed ELF and dynamic symbols' addresses.
	addresses map[string]uint64
	// Keep track of symbol table lazy load.
	addressesOnce sync.Once
}

// UprobeOptions defines additional parameters that will be used
// when loading Uprobes.
type UprobeOptions struct {
	// Symbol address. Must be provided in case of external symbols (shared libs).
	// If set, overrides the address eventually parsed from the executable.
	Address uint64
	// The offset relative to given symbol. Useful when tracing an arbitrary point
	// inside the frame of given symbol.
	//
	// Note: this field changed from being an absolute offset to being relative
	// to Address.
	Offset uint64
	// Only set the uprobe on the given process ID. Useful when tracing
	// shared library calls or programs that have many running instances.
	PID int
	// Automatically manage SDT reference counts (semaphores).
	//
	// If this field is set, the Kernel will increment/decrement the
	// semaphore located in the process memory at the provided address on
	// probe attach/detach.
	//
	// See also:
	// sourceware.org/systemtap/wiki/UserSpaceProbeImplementation (Semaphore Handling)
	// github.com/torvalds/linux/commit/1cc33161a83d
	// github.com/torvalds/linux/commit/a6ca88b241d5
	RefCtrOffset uint64
	// Arbitrary value that can be fetched from an eBPF program
	// via `bpf_get_attach_cookie()`.
	//
	// Needs kernel 5.15+.
	Cookie uint64
	// Prefix used for the event name if the uprobe must be attached using tracefs.
	// The group name will be formatted as `<prefix>_<randomstr>`.
	// The default empty string is equivalent to "ebpf" as the prefix.
	TraceFSPrefix string
}

func (uo *UprobeOptions) cookie() uint64 {
	if uo == nil {
		return 0
	}
	return uo.Cookie
}

// To open a new Executable, use:
//
//	OpenExecutable("/bin/bash")
//
// The returned value can then be used to open Uprobe(s).
func OpenExecutable(path string) (*Executable, error) {
	if path == "" {
		return nil, fmt.Errorf("path cannot be empty")
	}

	f, err := internal.OpenSafeELFFile(path)
	if err != nil {
		return nil, fmt.Errorf("parse ELF file: %w", err)
	}
	defer f.Close()

	if f.Type != elf.ET_EXEC && f.Type != elf.ET_DYN {
		// ELF is not an executable or a shared object.
		return nil, errors.New("the given file is not an executable or a shared object")
	}

	return &Executable{
		path:      path,
		addresses: make(map[string]uint64),
	}, nil
}

func (ex *Executable) load(f *internal.SafeELFFile) error {
	syms, err := f.Symbols()
	if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
		return err
	}

	dynsyms, err := f.DynamicSymbols()
	if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
		return err
	}

	syms = append(syms, dynsyms...)

	for _, s := range syms {
		if elf.ST_TYPE(s.Info) != elf.STT_FUNC {
			// Symbol not associated with a function or other executable code.
			continue
		}

		address := s.Value

		// Loop over ELF segments.
		for _, prog := range f.Progs {
			// Skip uninteresting segments.
			if prog.Type != elf.PT_LOAD || (prog.Flags&elf.PF_X) == 0 {
				continue
			}

			if prog.Vaddr <= s.Value && s.Value < (prog.Vaddr+prog.Memsz) {
				// If the symbol value is contained in the segment, calculate
				// the symbol offset.
				//
				// fn symbol offset = fn symbol VA - .text VA + .text offset
				//
				// stackoverflow.com/a/40249502
				address = s.Value - prog.Vaddr + prog.Off
				break
			}
		}

		ex.addresses[s.Name] = address
	}

	return nil
}

// address calculates the address of a symbol in the executable.
//
// opts must not be nil.
func (ex *Executable) address(symbol string, opts *UprobeOptions) (uint64, error) {
	if opts.Address > 0 {
		return opts.Address + opts.Offset, nil
	}

	var err error
	ex.addressesOnce.Do(func() {
		var f *internal.SafeELFFile
		f, err = internal.OpenSafeELFFile(ex.path)
		if err != nil {
			err = fmt.Errorf("parse ELF file: %w", err)
			return
		}
		defer f.Close()

		err = ex.load(f)
	})
	if err != nil {
		return 0, fmt.Errorf("lazy load symbols: %w", err)
	}

	address, ok := ex.addresses[symbol]
	if !ok {
		return 0, fmt.Errorf("symbol %s: %w", symbol, ErrNoSymbol)
	}

	// Symbols with location 0 from section undef are shared library calls and
	// are relocated before the binary is executed. Dynamic linking is not
	// implemented by the library, so mark this as unsupported for now.
	//
	// Since only offset values are stored and not elf.Symbol, if the value is 0,
	// assume it's an external symbol.
	if address == 0 {
		return 0, fmt.Errorf("cannot resolve %s library call '%s': %w "+
			"(consider providing UprobeOptions.Address)", ex.path, symbol, ErrNotSupported)
	}

	return address + opts.Offset, nil
}

// Uprobe attaches the given eBPF program to a perf event that fires when the
// given symbol starts executing in the given Executable.
// For example, /bin/bash::main():
//
//	ex, _ = OpenExecutable("/bin/bash")
//	ex.Uprobe("main", prog, nil)
//
// When using symbols which belongs to shared libraries,
// an offset must be provided via options:
//
//	up, err := ex.Uprobe("main", prog, &UprobeOptions{Offset: 0x123})
//
// Note: Setting the Offset field in the options supersedes the symbol's offset.
//
// Losing the reference to the resulting Link (up) will close the Uprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
//
// Functions provided by shared libraries can currently not be traced and
// will result in an ErrNotSupported.
func (ex *Executable) Uprobe(symbol string, prog *ebpf.Program, opts *UprobeOptions) (Link, error) {
	u, err := ex.uprobe(symbol, prog, opts, false)
	if err != nil {
		return nil, err
	}

	lnk, err := attachPerfEvent(u, prog, opts.cookie())
	if err != nil {
		u.Close()
		return nil, err
	}

	return lnk, nil
}

// Uretprobe attaches the given eBPF program to a perf event that fires right
// before the given symbol exits. For example, /bin/bash::main():
//
//	ex, _ = OpenExecutable("/bin/bash")
//	ex.Uretprobe("main", prog, nil)
//
// When using symbols which belongs to shared libraries,
// an offset must be provided via options:
//
//	up, err := ex.Uretprobe("main", prog, &UprobeOptions{Offset: 0x123})
//
// Note: Setting the Offset field in the options supersedes the symbol's offset.
//
// Losing the reference to the resulting Link (up) will close the Uprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
//
// Functions provided by shared libraries can currently not be traced and
// will result in an ErrNotSupported.
func (ex *Executable) Uretprobe(symbol string, prog *ebpf.Program, opts *UprobeOptions) (Link, error) {
	u, err := ex.uprobe(symbol, prog, opts, true)
	if err != nil {
		return nil, err
	}

	lnk, err := attachPerfEvent(u, prog, opts.cookie())
	if err != nil {
		u.Close()
		return nil, err
	}

	return lnk, nil
}

// uprobe opens a perf event for the given binary/symbol and attaches prog to it.
// If ret is true, create a uretprobe.
func (ex *Executable) uprobe(symbol string, prog *ebpf.Program, opts *UprobeOptions, ret bool) (*perfEvent, error) {
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if prog.Type() != ebpf.Kprobe {
		return nil, fmt.Errorf("eBPF program type %s is not Kprobe: %w", prog.Type(), errInvalidInput)
	}
	if opts == nil {
		opts = &UprobeOptions{}
	}

	offset, err := ex.address(symbol, opts)
	if err != nil {
		return nil, err
	}

	pid := opts.PID
	if pid == 0 {
		pid = perfAllThreads
	}

	if opts.RefCtrOffset != 0 {
		if err := haveRefCtrOffsetPMU(); err != nil {
			return nil, fmt.Errorf("uprobe ref_ctr_offset: %w", err)
		}
	}

	args := tracefs.ProbeArgs{
		Type:         tracefs.Uprobe,
		Symbol:       symbol,
		Path:         ex.path,
		Offset:       offset,
		Pid:          pid,
		RefCtrOffset: opts.RefCtrOffset,
		Ret:          ret,
		Cookie:       opts.Cookie,
		Group:        opts.TraceFSPrefix,
	}

	// Use uprobe PMU if the kernel has it available.
	tp, err := pmuProbe(args)
	if err == nil {
		return tp, nil
	}
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("creating perf_uprobe PMU: %w", err)
	}

	// Use tracefs if uprobe PMU is missing.
	tp, err = tracefsProbe(args)
	if err != nil {
		return nil, fmt.Errorf("creating trace event '%s:%s' in tracefs: %w", ex.path, symbol, err)
	}

	return tp, nil
}

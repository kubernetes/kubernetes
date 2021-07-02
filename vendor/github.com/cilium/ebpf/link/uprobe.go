package link

import (
	"debug/elf"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sync"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
)

var (
	uprobeEventsPath = filepath.Join(tracefsPath, "uprobe_events")

	// rgxUprobeSymbol is used to strip invalid characters from the uprobe symbol
	// as they are not allowed to be used as the EVENT token in tracefs.
	rgxUprobeSymbol = regexp.MustCompile("[^a-zA-Z0-9]+")

	uprobeRetprobeBit = struct {
		once  sync.Once
		value uint64
		err   error
	}{}
)

// Executable defines an executable program on the filesystem.
type Executable struct {
	// Path of the executable on the filesystem.
	path string
	// Parsed ELF symbols and dynamic symbols.
	symbols map[string]elf.Symbol
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

	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file '%s': %w", path, err)
	}
	defer f.Close()

	se, err := internal.NewSafeELFFile(f)
	if err != nil {
		return nil, fmt.Errorf("parse ELF file: %w", err)
	}

	var ex = Executable{
		path:    path,
		symbols: make(map[string]elf.Symbol),
	}
	if err := ex.addSymbols(se.Symbols); err != nil {
		return nil, err
	}

	if err := ex.addSymbols(se.DynamicSymbols); err != nil {
		return nil, err
	}

	return &ex, nil
}

func (ex *Executable) addSymbols(f func() ([]elf.Symbol, error)) error {
	// elf.Symbols and elf.DynamicSymbols return ErrNoSymbols if the section is not found.
	syms, err := f()
	if err != nil && !errors.Is(err, elf.ErrNoSymbols) {
		return err
	}
	for _, s := range syms {
		ex.symbols[s.Name] = s
	}
	return nil
}

func (ex *Executable) symbol(symbol string) (*elf.Symbol, error) {
	if s, ok := ex.symbols[symbol]; ok {
		return &s, nil
	}
	return nil, fmt.Errorf("symbol %s not found", symbol)
}

// Uprobe attaches the given eBPF program to a perf event that fires when the
// given symbol starts executing in the given Executable.
// For example, /bin/bash::main():
//
//  ex, _ = OpenExecutable("/bin/bash")
//  ex.Uprobe("main", prog)
//
// The resulting Link must be Closed during program shutdown to avoid leaking
// system resources. Functions provided by shared libraries can currently not
// be traced and will result in an ErrNotSupported.
func (ex *Executable) Uprobe(symbol string, prog *ebpf.Program) (Link, error) {
	u, err := ex.uprobe(symbol, prog, false)
	if err != nil {
		return nil, err
	}

	err = u.attach(prog)
	if err != nil {
		u.Close()
		return nil, err
	}

	return u, nil
}

// Uretprobe attaches the given eBPF program to a perf event that fires right
// before the given symbol exits. For example, /bin/bash::main():
//
//  ex, _ = OpenExecutable("/bin/bash")
//  ex.Uretprobe("main", prog)
//
// The resulting Link must be Closed during program shutdown to avoid leaking
// system resources. Functions provided by shared libraries can currently not
// be traced and will result in an ErrNotSupported.
func (ex *Executable) Uretprobe(symbol string, prog *ebpf.Program) (Link, error) {
	u, err := ex.uprobe(symbol, prog, true)
	if err != nil {
		return nil, err
	}

	err = u.attach(prog)
	if err != nil {
		u.Close()
		return nil, err
	}

	return u, nil
}

// uprobe opens a perf event for the given binary/symbol and attaches prog to it.
// If ret is true, create a uretprobe.
func (ex *Executable) uprobe(symbol string, prog *ebpf.Program, ret bool) (*perfEvent, error) {
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if prog.Type() != ebpf.Kprobe {
		return nil, fmt.Errorf("eBPF program type %s is not Kprobe: %w", prog.Type(), errInvalidInput)
	}

	sym, err := ex.symbol(symbol)
	if err != nil {
		return nil, fmt.Errorf("symbol '%s' not found in '%s': %w", symbol, ex.path, err)
	}

	// Symbols with location 0 from section undef are shared library calls and
	// are relocated before the binary is executed. Dynamic linking is not
	// implemented by the library, so mark this as unsupported for now.
	if sym.Section == elf.SHN_UNDEF && sym.Value == 0 {
		return nil, fmt.Errorf("cannot resolve %s library call '%s': %w", ex.path, symbol, ErrNotSupported)
	}

	// Use uprobe PMU if the kernel has it available.
	tp, err := pmuUprobe(sym.Name, ex.path, sym.Value, ret)
	if err == nil {
		return tp, nil
	}
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("creating perf_uprobe PMU: %w", err)
	}

	// Use tracefs if uprobe PMU is missing.
	tp, err = tracefsUprobe(uprobeSanitizedSymbol(sym.Name), ex.path, sym.Value, ret)
	if err != nil {
		return nil, fmt.Errorf("creating trace event '%s:%s' in tracefs: %w", ex.path, symbol, err)
	}

	return tp, nil
}

// pmuUprobe opens a perf event based on the uprobe PMU.
func pmuUprobe(symbol, path string, offset uint64, ret bool) (*perfEvent, error) {
	return pmuProbe(uprobeType, symbol, path, offset, ret)
}

// tracefsUprobe creates a Uprobe tracefs entry.
func tracefsUprobe(symbol, path string, offset uint64, ret bool) (*perfEvent, error) {
	return tracefsProbe(uprobeType, symbol, path, offset, ret)
}

// uprobeSanitizedSymbol replaces every invalid characted for the tracefs api with an underscore.
func uprobeSanitizedSymbol(symbol string) string {
	return rgxUprobeSymbol.ReplaceAllString(symbol, "_")
}

// uprobePathOffset creates the PATH:OFFSET token for the tracefs api.
func uprobePathOffset(path string, offset uint64) string {
	return fmt.Sprintf("%s:%#x", path, offset)
}

func uretprobeBit() (uint64, error) {
	uprobeRetprobeBit.once.Do(func() {
		uprobeRetprobeBit.value, uprobeRetprobeBit.err = determineRetprobeBit(uprobeType)
	})
	return uprobeRetprobeBit.value, uprobeRetprobeBit.err
}

package link

import (
	"errors"
	"fmt"
	"os"
	"runtime"
	"strings"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/tracefs"
	"github.com/cilium/ebpf/internal/unix"
)

// KprobeOptions defines additional parameters that will be used
// when loading Kprobes.
type KprobeOptions struct {
	// Arbitrary value that can be fetched from an eBPF program
	// via `bpf_get_attach_cookie()`.
	//
	// Needs kernel 5.15+.
	Cookie uint64
	// Offset of the kprobe relative to the traced symbol.
	// Can be used to insert kprobes at arbitrary offsets in kernel functions,
	// e.g. in places where functions have been inlined.
	Offset uint64
	// Increase the maximum number of concurrent invocations of a kretprobe.
	// Required when tracing some long running functions in the kernel.
	//
	// Deprecated: this setting forces the use of an outdated kernel API and is not portable
	// across kernel versions.
	RetprobeMaxActive int
	// Prefix used for the event name if the kprobe must be attached using tracefs.
	// The group name will be formatted as `<prefix>_<randomstr>`.
	// The default empty string is equivalent to "ebpf" as the prefix.
	TraceFSPrefix string
}

func (ko *KprobeOptions) cookie() uint64 {
	if ko == nil {
		return 0
	}
	return ko.Cookie
}

// Kprobe attaches the given eBPF program to a perf event that fires when the
// given kernel symbol starts executing. See /proc/kallsyms for available
// symbols. For example, printk():
//
//	kp, err := Kprobe("printk", prog, nil)
//
// Losing the reference to the resulting Link (kp) will close the Kprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
//
// If attaching to symbol fails, automatically retries with the running
// platform's syscall prefix (e.g. __x64_) to support attaching to syscalls
// in a portable fashion.
func Kprobe(symbol string, prog *ebpf.Program, opts *KprobeOptions) (Link, error) {
	k, err := kprobe(symbol, prog, opts, false)
	if err != nil {
		return nil, err
	}

	lnk, err := attachPerfEvent(k, prog, opts.cookie())
	if err != nil {
		k.Close()
		return nil, err
	}

	return lnk, nil
}

// Kretprobe attaches the given eBPF program to a perf event that fires right
// before the given kernel symbol exits, with the function stack left intact.
// See /proc/kallsyms for available symbols. For example, printk():
//
//	kp, err := Kretprobe("printk", prog, nil)
//
// Losing the reference to the resulting Link (kp) will close the Kretprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
//
// If attaching to symbol fails, automatically retries with the running
// platform's syscall prefix (e.g. __x64_) to support attaching to syscalls
// in a portable fashion.
//
// On kernels 5.10 and earlier, setting a kretprobe on a nonexistent symbol
// incorrectly returns unix.EINVAL instead of os.ErrNotExist.
func Kretprobe(symbol string, prog *ebpf.Program, opts *KprobeOptions) (Link, error) {
	k, err := kprobe(symbol, prog, opts, true)
	if err != nil {
		return nil, err
	}

	lnk, err := attachPerfEvent(k, prog, opts.cookie())
	if err != nil {
		k.Close()
		return nil, err
	}

	return lnk, nil
}

// isValidKprobeSymbol implements the equivalent of a regex match
// against "^[a-zA-Z_][0-9a-zA-Z_.]*$".
func isValidKprobeSymbol(s string) bool {
	if len(s) < 1 {
		return false
	}

	for i, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c == '_':
		case i > 0 && c >= '0' && c <= '9':

		// Allow `.` in symbol name. GCC-compiled kernel may change symbol name
		// to have a `.isra.$n` suffix, like `udp_send_skb.isra.52`.
		// See: https://gcc.gnu.org/gcc-10/changes.html
		case i > 0 && c == '.':

		default:
			return false
		}
	}

	return true
}

// kprobe opens a perf event on the given symbol and attaches prog to it.
// If ret is true, create a kretprobe.
func kprobe(symbol string, prog *ebpf.Program, opts *KprobeOptions, ret bool) (*perfEvent, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol name cannot be empty: %w", errInvalidInput)
	}
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if !isValidKprobeSymbol(symbol) {
		return nil, fmt.Errorf("symbol '%s' must be a valid symbol in /proc/kallsyms: %w", symbol, errInvalidInput)
	}
	if prog.Type() != ebpf.Kprobe {
		return nil, fmt.Errorf("eBPF program type %s is not a Kprobe: %w", prog.Type(), errInvalidInput)
	}

	args := tracefs.ProbeArgs{
		Type:   tracefs.Kprobe,
		Pid:    perfAllThreads,
		Symbol: symbol,
		Ret:    ret,
	}

	if opts != nil {
		args.RetprobeMaxActive = opts.RetprobeMaxActive
		args.Cookie = opts.Cookie
		args.Offset = opts.Offset
		args.Group = opts.TraceFSPrefix
	}

	// Use kprobe PMU if the kernel has it available.
	tp, err := pmuProbe(args)
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		if prefix := internal.PlatformPrefix(); prefix != "" {
			args.Symbol = prefix + symbol
			tp, err = pmuProbe(args)
		}
	}
	if err == nil {
		return tp, nil
	}
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("creating perf_kprobe PMU (arch-specific fallback for %q): %w", symbol, err)
	}

	// Use tracefs if kprobe PMU is missing.
	args.Symbol = symbol
	tp, err = tracefsProbe(args)
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		if prefix := internal.PlatformPrefix(); prefix != "" {
			args.Symbol = prefix + symbol
			tp, err = tracefsProbe(args)
		}
	}
	if err != nil {
		return nil, fmt.Errorf("creating tracefs event (arch-specific fallback for %q): %w", symbol, err)
	}

	return tp, nil
}

// pmuProbe opens a perf event based on a Performance Monitoring Unit.
//
// Requires at least a 4.17 kernel.
// e12f03d7031a "perf/core: Implement the 'perf_kprobe' PMU"
// 33ea4b24277b "perf/core: Implement the 'perf_uprobe' PMU"
//
// Returns ErrNotSupported if the kernel doesn't support perf_[k,u]probe PMU
func pmuProbe(args tracefs.ProbeArgs) (*perfEvent, error) {
	// Getting the PMU type will fail if the kernel doesn't support
	// the perf_[k,u]probe PMU.
	eventType, err := internal.ReadUint64FromFileOnce("%d\n", "/sys/bus/event_source/devices", args.Type.String(), "type")
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("%s: %w", args.Type, ErrNotSupported)
	}
	if err != nil {
		return nil, err
	}

	// Use tracefs if we want to set kretprobe's retprobeMaxActive.
	if args.RetprobeMaxActive != 0 {
		return nil, fmt.Errorf("pmu probe: non-zero retprobeMaxActive: %w", ErrNotSupported)
	}

	var config uint64
	if args.Ret {
		bit, err := internal.ReadUint64FromFileOnce("config:%d\n", "/sys/bus/event_source/devices", args.Type.String(), "/format/retprobe")
		if err != nil {
			return nil, err
		}
		config |= 1 << bit
	}

	var (
		attr  unix.PerfEventAttr
		sp    unsafe.Pointer
		token string
	)
	switch args.Type {
	case tracefs.Kprobe:
		// Create a pointer to a NUL-terminated string for the kernel.
		sp, err = unsafeStringPtr(args.Symbol)
		if err != nil {
			return nil, err
		}

		token = tracefs.KprobeToken(args)

		attr = unix.PerfEventAttr{
			// The minimum size required for PMU kprobes is PERF_ATTR_SIZE_VER1,
			// since it added the config2 (Ext2) field. Use Ext2 as probe_offset.
			Size:   unix.PERF_ATTR_SIZE_VER1,
			Type:   uint32(eventType),   // PMU event type read from sysfs
			Ext1:   uint64(uintptr(sp)), // Kernel symbol to trace
			Ext2:   args.Offset,         // Kernel symbol offset
			Config: config,              // Retprobe flag
		}
	case tracefs.Uprobe:
		sp, err = unsafeStringPtr(args.Path)
		if err != nil {
			return nil, err
		}

		if args.RefCtrOffset != 0 {
			config |= args.RefCtrOffset << uprobeRefCtrOffsetShift
		}

		token = tracefs.UprobeToken(args)

		attr = unix.PerfEventAttr{
			// The minimum size required for PMU uprobes is PERF_ATTR_SIZE_VER1,
			// since it added the config2 (Ext2) field. The Size field controls the
			// size of the internal buffer the kernel allocates for reading the
			// perf_event_attr argument from userspace.
			Size:   unix.PERF_ATTR_SIZE_VER1,
			Type:   uint32(eventType),   // PMU event type read from sysfs
			Ext1:   uint64(uintptr(sp)), // Uprobe path
			Ext2:   args.Offset,         // Uprobe offset
			Config: config,              // RefCtrOffset, Retprobe flag
		}
	}

	rawFd, err := unix.PerfEventOpen(&attr, args.Pid, 0, -1, unix.PERF_FLAG_FD_CLOEXEC)

	// On some old kernels, kprobe PMU doesn't allow `.` in symbol names and
	// return -EINVAL. Return ErrNotSupported to allow falling back to tracefs.
	// https://github.com/torvalds/linux/blob/94710cac0ef4/kernel/trace/trace_kprobe.c#L340-L343
	if errors.Is(err, unix.EINVAL) && strings.Contains(args.Symbol, ".") {
		return nil, fmt.Errorf("token %s: older kernels don't accept dots: %w", token, ErrNotSupported)
	}
	// Since commit 97c753e62e6c, ENOENT is correctly returned instead of EINVAL
	// when trying to create a retprobe for a missing symbol.
	if errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("token %s: not found: %w", token, err)
	}
	// Since commit ab105a4fb894, EILSEQ is returned when a kprobe sym+offset is resolved
	// to an invalid insn boundary. The exact conditions that trigger this error are
	// arch specific however.
	if errors.Is(err, unix.EILSEQ) {
		return nil, fmt.Errorf("token %s: bad insn boundary: %w", token, os.ErrNotExist)
	}
	// Since at least commit cb9a19fe4aa51, ENOTSUPP is returned
	// when attempting to set a uprobe on a trap instruction.
	if errors.Is(err, sys.ENOTSUPP) {
		return nil, fmt.Errorf("token %s: failed setting uprobe on offset %#x (possible trap insn): %w", token, args.Offset, err)
	}

	if err != nil {
		return nil, fmt.Errorf("token %s: opening perf event: %w", token, err)
	}

	// Ensure the string pointer is not collected before PerfEventOpen returns.
	runtime.KeepAlive(sp)

	fd, err := sys.NewFD(rawFd)
	if err != nil {
		return nil, err
	}

	// Kernel has perf_[k,u]probe PMU available, initialize perf event.
	return newPerfEvent(fd, nil), nil
}

// tracefsProbe creates a trace event by writing an entry to <tracefs>/[k,u]probe_events.
// A new trace event group name is generated on every call to support creating
// multiple trace events for the same kernel or userspace symbol.
// Path and offset are only set in the case of uprobe(s) and are used to set
// the executable/library path on the filesystem and the offset where the probe is inserted.
// A perf event is then opened on the newly-created trace event and returned to the caller.
func tracefsProbe(args tracefs.ProbeArgs) (*perfEvent, error) {
	groupPrefix := "ebpf"
	if args.Group != "" {
		groupPrefix = args.Group
	}

	// Generate a random string for each trace event we attempt to create.
	// This value is used as the 'group' token in tracefs to allow creating
	// multiple kprobe trace events with the same name.
	group, err := tracefs.RandomGroup(groupPrefix)
	if err != nil {
		return nil, fmt.Errorf("randomizing group name: %w", err)
	}
	args.Group = group

	// Create the [k,u]probe trace event using tracefs.
	evt, err := tracefs.NewEvent(args)
	if err != nil {
		return nil, fmt.Errorf("creating probe entry on tracefs: %w", err)
	}

	// Kprobes are ephemeral tracepoints and share the same perf event type.
	fd, err := openTracepointPerfEvent(evt.ID(), args.Pid)
	if err != nil {
		// Make sure we clean up the created tracefs event when we return error.
		// If a livepatch handler is already active on the symbol, the write to
		// tracefs will succeed, a trace event will show up, but creating the
		// perf event will fail with EBUSY.
		_ = evt.Close()
		return nil, err
	}

	return newPerfEvent(fd, evt), nil
}

package link

import (
	"bytes"
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"sync"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"
)

var (
	kprobeEventsPath = filepath.Join(tracefsPath, "kprobe_events")

	kprobeRetprobeBit = struct {
		once  sync.Once
		value uint64
		err   error
	}{}
)

type probeType uint8

const (
	kprobeType probeType = iota
	uprobeType
)

func (pt probeType) String() string {
	if pt == kprobeType {
		return "kprobe"
	}
	return "uprobe"
}

func (pt probeType) EventsPath() string {
	if pt == kprobeType {
		return kprobeEventsPath
	}
	return uprobeEventsPath
}

func (pt probeType) PerfEventType(ret bool) perfEventType {
	if pt == kprobeType {
		if ret {
			return kretprobeEvent
		}
		return kprobeEvent
	}
	if ret {
		return uretprobeEvent
	}
	return uprobeEvent
}

func (pt probeType) RetprobeBit() (uint64, error) {
	if pt == kprobeType {
		return kretprobeBit()
	}
	return uretprobeBit()
}

// Kprobe attaches the given eBPF program to a perf event that fires when the
// given kernel symbol starts executing. See /proc/kallsyms for available
// symbols. For example, printk():
//
//	kp, err := Kprobe("printk", prog)
//
// Losing the reference to the resulting Link (kp) will close the Kprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
func Kprobe(symbol string, prog *ebpf.Program) (Link, error) {
	k, err := kprobe(symbol, prog, false)
	if err != nil {
		return nil, err
	}

	err = k.attach(prog)
	if err != nil {
		k.Close()
		return nil, err
	}

	return k, nil
}

// Kretprobe attaches the given eBPF program to a perf event that fires right
// before the given kernel symbol exits, with the function stack left intact.
// See /proc/kallsyms for available symbols. For example, printk():
//
//	kp, err := Kretprobe("printk", prog)
//
// Losing the reference to the resulting Link (kp) will close the Kretprobe
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
func Kretprobe(symbol string, prog *ebpf.Program) (Link, error) {
	k, err := kprobe(symbol, prog, true)
	if err != nil {
		return nil, err
	}

	err = k.attach(prog)
	if err != nil {
		k.Close()
		return nil, err
	}

	return k, nil
}

// kprobe opens a perf event on the given symbol and attaches prog to it.
// If ret is true, create a kretprobe.
func kprobe(symbol string, prog *ebpf.Program, ret bool) (*perfEvent, error) {
	if symbol == "" {
		return nil, fmt.Errorf("symbol name cannot be empty: %w", errInvalidInput)
	}
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if !rgxTraceEvent.MatchString(symbol) {
		return nil, fmt.Errorf("symbol '%s' must be alphanumeric or underscore: %w", symbol, errInvalidInput)
	}
	if prog.Type() != ebpf.Kprobe {
		return nil, fmt.Errorf("eBPF program type %s is not a Kprobe: %w", prog.Type(), errInvalidInput)
	}

	// Use kprobe PMU if the kernel has it available.
	tp, err := pmuKprobe(platformPrefix(symbol), ret)
	if errors.Is(err, os.ErrNotExist) {
		tp, err = pmuKprobe(symbol, ret)
	}
	if err == nil {
		return tp, nil
	}
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("creating perf_kprobe PMU: %w", err)
	}

	// Use tracefs if kprobe PMU is missing.
	tp, err = tracefsKprobe(platformPrefix(symbol), ret)
	if errors.Is(err, os.ErrNotExist) {
		tp, err = tracefsKprobe(symbol, ret)
	}
	if err != nil {
		return nil, fmt.Errorf("creating trace event '%s' in tracefs: %w", symbol, err)
	}

	return tp, nil
}

// pmuKprobe opens a perf event based on the kprobe PMU.
// Returns os.ErrNotExist if the given symbol does not exist in the kernel.
func pmuKprobe(symbol string, ret bool) (*perfEvent, error) {
	return pmuProbe(kprobeType, symbol, "", 0, perfAllThreads, ret)
}

// pmuProbe opens a perf event based on a Performance Monitoring Unit.
//
// Requires at least a 4.17 kernel.
// e12f03d7031a "perf/core: Implement the 'perf_kprobe' PMU"
// 33ea4b24277b "perf/core: Implement the 'perf_uprobe' PMU"
//
// Returns ErrNotSupported if the kernel doesn't support perf_[k,u]probe PMU
func pmuProbe(typ probeType, symbol, path string, offset uint64, pid int, ret bool) (*perfEvent, error) {
	// Getting the PMU type will fail if the kernel doesn't support
	// the perf_[k,u]probe PMU.
	et, err := getPMUEventType(typ)
	if err != nil {
		return nil, err
	}

	var config uint64
	if ret {
		bit, err := typ.RetprobeBit()
		if err != nil {
			return nil, err
		}
		config |= 1 << bit
	}

	var (
		attr unix.PerfEventAttr
		sp   unsafe.Pointer
	)
	switch typ {
	case kprobeType:
		// Create a pointer to a NUL-terminated string for the kernel.
		sp, err = unsafeStringPtr(symbol)
		if err != nil {
			return nil, err
		}

		attr = unix.PerfEventAttr{
			Type:   uint32(et),          // PMU event type read from sysfs
			Ext1:   uint64(uintptr(sp)), // Kernel symbol to trace
			Config: config,              // Retprobe flag
		}
	case uprobeType:
		sp, err = unsafeStringPtr(path)
		if err != nil {
			return nil, err
		}

		attr = unix.PerfEventAttr{
			// The minimum size required for PMU uprobes is PERF_ATTR_SIZE_VER1,
			// since it added the config2 (Ext2) field. The Size field controls the
			// size of the internal buffer the kernel allocates for reading the
			// perf_event_attr argument from userspace.
			Size:   unix.PERF_ATTR_SIZE_VER1,
			Type:   uint32(et),          // PMU event type read from sysfs
			Ext1:   uint64(uintptr(sp)), // Uprobe path
			Ext2:   offset,              // Uprobe offset
			Config: config,              // Retprobe flag
		}
	}

	fd, err := unix.PerfEventOpen(&attr, pid, 0, -1, unix.PERF_FLAG_FD_CLOEXEC)

	// Since commit 97c753e62e6c, ENOENT is correctly returned instead of EINVAL
	// when trying to create a kretprobe for a missing symbol. Make sure ENOENT
	// is returned to the caller.
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		return nil, fmt.Errorf("symbol '%s' not found: %w", symbol, os.ErrNotExist)
	}
	// Since at least commit cb9a19fe4aa51, ENOTSUPP is returned
	// when attempting to set a uprobe on a trap instruction.
	if errors.Is(err, unix.ENOTSUPP) {
		return nil, fmt.Errorf("failed setting uprobe on offset %#x (possible trap insn): %w", offset, err)
	}
	if err != nil {
		return nil, fmt.Errorf("opening perf event: %w", err)
	}

	// Ensure the string pointer is not collected before PerfEventOpen returns.
	runtime.KeepAlive(sp)

	// Kernel has perf_[k,u]probe PMU available, initialize perf event.
	return &perfEvent{
		fd:    internal.NewFD(uint32(fd)),
		pmuID: et,
		name:  symbol,
		typ:   typ.PerfEventType(ret),
	}, nil
}

// tracefsKprobe creates a Kprobe tracefs entry.
func tracefsKprobe(symbol string, ret bool) (*perfEvent, error) {
	return tracefsProbe(kprobeType, symbol, "", 0, perfAllThreads, ret)
}

// tracefsProbe creates a trace event by writing an entry to <tracefs>/[k,u]probe_events.
// A new trace event group name is generated on every call to support creating
// multiple trace events for the same kernel or userspace symbol.
// Path and offset are only set in the case of uprobe(s) and are used to set
// the executable/library path on the filesystem and the offset where the probe is inserted.
// A perf event is then opened on the newly-created trace event and returned to the caller.
func tracefsProbe(typ probeType, symbol, path string, offset uint64, pid int, ret bool) (*perfEvent, error) {
	// Generate a random string for each trace event we attempt to create.
	// This value is used as the 'group' token in tracefs to allow creating
	// multiple kprobe trace events with the same name.
	group, err := randomGroup("ebpf")
	if err != nil {
		return nil, fmt.Errorf("randomizing group name: %w", err)
	}

	// Before attempting to create a trace event through tracefs,
	// check if an event with the same group and name already exists.
	// Kernels 4.x and earlier don't return os.ErrExist on writing a duplicate
	// entry, so we need to rely on reads for detecting uniqueness.
	_, err = getTraceEventID(group, symbol)
	if err == nil {
		return nil, fmt.Errorf("trace event already exists: %s/%s", group, symbol)
	}
	if err != nil && !errors.Is(err, os.ErrNotExist) {
		return nil, fmt.Errorf("checking trace event %s/%s: %w", group, symbol, err)
	}

	// Create the [k,u]probe trace event using tracefs.
	if err := createTraceFSProbeEvent(typ, group, symbol, path, offset, ret); err != nil {
		return nil, fmt.Errorf("creating probe entry on tracefs: %w", err)
	}

	// Get the newly-created trace event's id.
	tid, err := getTraceEventID(group, symbol)
	if err != nil {
		return nil, fmt.Errorf("getting trace event id: %w", err)
	}

	// Kprobes are ephemeral tracepoints and share the same perf event type.
	fd, err := openTracepointPerfEvent(tid, pid)
	if err != nil {
		return nil, err
	}

	return &perfEvent{
		fd:        fd,
		group:     group,
		name:      symbol,
		tracefsID: tid,
		typ:       typ.PerfEventType(ret),
	}, nil
}

// createTraceFSProbeEvent creates a new ephemeral trace event by writing to
// <tracefs>/[k,u]probe_events. Returns os.ErrNotExist if symbol is not a valid
// kernel symbol, or if it is not traceable with kprobes. Returns os.ErrExist
// if a probe with the same group and symbol already exists.
func createTraceFSProbeEvent(typ probeType, group, symbol, path string, offset uint64, ret bool) error {
	// Open the kprobe_events file in tracefs.
	f, err := os.OpenFile(typ.EventsPath(), os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		return fmt.Errorf("error opening '%s': %w", typ.EventsPath(), err)
	}
	defer f.Close()

	var pe string
	switch typ {
	case kprobeType:
		// The kprobe_events syntax is as follows (see Documentation/trace/kprobetrace.txt):
		// p[:[GRP/]EVENT] [MOD:]SYM[+offs]|MEMADDR [FETCHARGS] : Set a probe
		// r[MAXACTIVE][:[GRP/]EVENT] [MOD:]SYM[+0] [FETCHARGS] : Set a return probe
		// -:[GRP/]EVENT                                        : Clear a probe
		//
		// Some examples:
		// r:ebpf_1234/r_my_kretprobe nf_conntrack_destroy
		// p:ebpf_5678/p_my_kprobe __x64_sys_execve
		//
		// Leaving the kretprobe's MAXACTIVE set to 0 (or absent) will make the
		// kernel default to NR_CPUS. This is desired in most eBPF cases since
		// subsampling or rate limiting logic can be more accurately implemented in
		// the eBPF program itself.
		// See Documentation/kprobes.txt for more details.
		pe = fmt.Sprintf("%s:%s/%s %s", probePrefix(ret), group, symbol, symbol)
	case uprobeType:
		// The uprobe_events syntax is as follows:
		// p[:[GRP/]EVENT] PATH:OFFSET [FETCHARGS] : Set a probe
		// r[:[GRP/]EVENT] PATH:OFFSET [FETCHARGS] : Set a return probe
		// -:[GRP/]EVENT                           : Clear a probe
		//
		// Some examples:
		// r:ebpf_1234/readline /bin/bash:0x12345
		// p:ebpf_5678/main_mySymbol /bin/mybin:0x12345
		//
		// See Documentation/trace/uprobetracer.txt for more details.
		pathOffset := uprobePathOffset(path, offset)
		pe = fmt.Sprintf("%s:%s/%s %s", probePrefix(ret), group, symbol, pathOffset)
	}
	_, err = f.WriteString(pe)
	// Since commit 97c753e62e6c, ENOENT is correctly returned instead of EINVAL
	// when trying to create a kretprobe for a missing symbol. Make sure ENOENT
	// is returned to the caller.
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		return fmt.Errorf("symbol %s not found: %w", symbol, os.ErrNotExist)
	}
	if err != nil {
		return fmt.Errorf("writing '%s' to '%s': %w", pe, typ.EventsPath(), err)
	}

	return nil
}

// closeTraceFSProbeEvent removes the [k,u]probe with the given type, group and symbol
// from <tracefs>/[k,u]probe_events.
func closeTraceFSProbeEvent(typ probeType, group, symbol string) error {
	f, err := os.OpenFile(typ.EventsPath(), os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		return fmt.Errorf("error opening %s: %w", typ.EventsPath(), err)
	}
	defer f.Close()

	// See [k,u]probe_events syntax above. The probe type does not need to be specified
	// for removals.
	pe := fmt.Sprintf("-:%s/%s", group, symbol)
	if _, err = f.WriteString(pe); err != nil {
		return fmt.Errorf("writing '%s' to '%s': %w", pe, typ.EventsPath(), err)
	}

	return nil
}

// randomGroup generates a pseudorandom string for use as a tracefs group name.
// Returns an error when the output string would exceed 63 characters (kernel
// limitation), when rand.Read() fails or when prefix contains characters not
// allowed by rgxTraceEvent.
func randomGroup(prefix string) (string, error) {
	if !rgxTraceEvent.MatchString(prefix) {
		return "", fmt.Errorf("prefix '%s' must be alphanumeric or underscore: %w", prefix, errInvalidInput)
	}

	b := make([]byte, 8)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("reading random bytes: %w", err)
	}

	group := fmt.Sprintf("%s_%x", prefix, b)
	if len(group) > 63 {
		return "", fmt.Errorf("group name '%s' cannot be longer than 63 characters: %w", group, errInvalidInput)
	}

	return group, nil
}

func probePrefix(ret bool) string {
	if ret {
		return "r"
	}
	return "p"
}

// determineRetprobeBit reads a Performance Monitoring Unit's retprobe bit
// from /sys/bus/event_source/devices/<pmu>/format/retprobe.
func determineRetprobeBit(typ probeType) (uint64, error) {
	p := filepath.Join("/sys/bus/event_source/devices/", typ.String(), "/format/retprobe")

	data, err := os.ReadFile(p)
	if err != nil {
		return 0, err
	}

	var rp uint64
	n, err := fmt.Sscanf(string(bytes.TrimSpace(data)), "config:%d", &rp)
	if err != nil {
		return 0, fmt.Errorf("parse retprobe bit: %w", err)
	}
	if n != 1 {
		return 0, fmt.Errorf("parse retprobe bit: expected 1 item, got %d", n)
	}

	return rp, nil
}

func kretprobeBit() (uint64, error) {
	kprobeRetprobeBit.once.Do(func() {
		kprobeRetprobeBit.value, kprobeRetprobeBit.err = determineRetprobeBit(kprobeType)
	})
	return kprobeRetprobeBit.value, kprobeRetprobeBit.err
}

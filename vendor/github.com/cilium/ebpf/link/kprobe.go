package link

import (
	"crypto/rand"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"
)

var (
	kprobeEventsPath = filepath.Join(tracefsPath, "kprobe_events")
)

// Kprobe attaches the given eBPF program to a perf event that fires when the
// given kernel symbol starts executing. See /proc/kallsyms for available
// symbols. For example, printk():
//
//	Kprobe("printk")
//
// The resulting Link must be Closed during program shutdown to avoid leaking
// system resources.
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
//	Kretprobe("printk")
//
// The resulting Link must be Closed during program shutdown to avoid leaking
// system resources.
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
	tp, err := pmuKprobe(symbol, ret)
	if err == nil {
		return tp, nil
	}
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("creating perf_kprobe PMU: %w", err)
	}

	// Use tracefs if kprobe PMU is missing.
	tp, err = tracefsKprobe(symbol, ret)
	if err != nil {
		return nil, fmt.Errorf("creating trace event '%s' in tracefs: %w", symbol, err)
	}

	return tp, nil
}

// pmuKprobe opens a perf event based on a Performance Monitoring Unit.
// Requires at least 4.17 (e12f03d7031a "perf/core: Implement the
// 'perf_kprobe' PMU").
// Returns ErrNotSupported if the kernel doesn't support perf_kprobe PMU,
// or os.ErrNotExist if the given symbol does not exist in the kernel.
func pmuKprobe(symbol string, ret bool) (*perfEvent, error) {

	// Getting the PMU type will fail if the kernel doesn't support
	// the perf_kprobe PMU.
	et, err := getPMUEventType("kprobe")
	if err != nil {
		return nil, err
	}

	// Create a pointer to a NUL-terminated string for the kernel.
	sp, err := unsafeStringPtr(symbol)
	if err != nil {
		return nil, err
	}

	// TODO: Parse the position of the bit from /sys/bus/event_source/devices/%s/format/retprobe.
	config := 0
	if ret {
		config = 1
	}

	attr := unix.PerfEventAttr{
		Type:   uint32(et),          // PMU event type read from sysfs
		Ext1:   uint64(uintptr(sp)), // Kernel symbol to trace
		Config: uint64(config),      // perf_kprobe PMU treats config as flags
	}

	fd, err := unix.PerfEventOpen(&attr, perfAllThreads, 0, -1, unix.PERF_FLAG_FD_CLOEXEC)

	// Since commit 97c753e62e6c, ENOENT is correctly returned instead of EINVAL
	// when trying to create a kretprobe for a missing symbol. Make sure ENOENT
	// is returned to the caller.
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		return nil, fmt.Errorf("symbol '%s' not found: %w", symbol, os.ErrNotExist)
	}
	if err != nil {
		return nil, fmt.Errorf("opening perf event: %w", err)
	}

	// Ensure the string pointer is not collected before PerfEventOpen returns.
	runtime.KeepAlive(sp)

	// Kernel has perf_kprobe PMU available, initialize perf event.
	return &perfEvent{
		fd:       internal.NewFD(uint32(fd)),
		pmuID:    et,
		name:     symbol,
		ret:      ret,
		progType: ebpf.Kprobe,
	}, nil
}

// tracefsKprobe creates a trace event by writing an entry to <tracefs>/kprobe_events.
// A new trace event group name is generated on every call to support creating
// multiple trace events for the same kernel symbol. A perf event is then opened
// on the newly-created trace event and returned to the caller.
func tracefsKprobe(symbol string, ret bool) (*perfEvent, error) {

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
	// The read is expected to fail with ErrNotSupported due to a non-existing event.
	if err != nil && !errors.Is(err, ErrNotSupported) {
		return nil, fmt.Errorf("checking trace event %s/%s: %w", group, symbol, err)
	}

	// Create the kprobe trace event using tracefs.
	if err := createTraceFSKprobeEvent(group, symbol, ret); err != nil {
		return nil, fmt.Errorf("creating kprobe event on tracefs: %w", err)
	}

	// Get the newly-created trace event's id.
	tid, err := getTraceEventID(group, symbol)
	if err != nil {
		return nil, fmt.Errorf("getting trace event id: %w", err)
	}

	// Kprobes are ephemeral tracepoints and share the same perf event type.
	fd, err := openTracepointPerfEvent(tid)
	if err != nil {
		return nil, err
	}

	return &perfEvent{
		fd:        fd,
		group:     group,
		name:      symbol,
		ret:       ret,
		tracefsID: tid,
		progType:  ebpf.Kprobe, // kernel only allows attaching kprobe programs to kprobe events
	}, nil
}

// createTraceFSKprobeEvent creates a new ephemeral trace event by writing to
// <tracefs>/kprobe_events. Returns ErrNotSupported if symbol is not a valid
// kernel symbol, or if it is not traceable with kprobes.
func createTraceFSKprobeEvent(group, symbol string, ret bool) error {
	// Open the kprobe_events file in tracefs.
	f, err := os.OpenFile(kprobeEventsPath, os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		return fmt.Errorf("error opening kprobe_events: %w", err)
	}
	defer f.Close()

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
	// the eBPF program itself. See Documentation/kprobes.txt for more details.
	pe := fmt.Sprintf("%s:%s/%s %s", kprobePrefix(ret), group, symbol, symbol)
	_, err = f.WriteString(pe)
	// Since commit 97c753e62e6c, ENOENT is correctly returned instead of EINVAL
	// when trying to create a kretprobe for a missing symbol. Make sure ENOENT
	// is returned to the caller.
	if errors.Is(err, os.ErrNotExist) || errors.Is(err, unix.EINVAL) {
		return fmt.Errorf("kernel symbol %s not found: %w", symbol, os.ErrNotExist)
	}
	if err != nil {
		return fmt.Errorf("writing '%s' to kprobe_events: %w", pe, err)
	}

	return nil
}

// closeTraceFSKprobeEvent removes the kprobe with the given group, symbol and kind
// from <tracefs>/kprobe_events.
func closeTraceFSKprobeEvent(group, symbol string) error {
	f, err := os.OpenFile(kprobeEventsPath, os.O_APPEND|os.O_WRONLY, 0666)
	if err != nil {
		return fmt.Errorf("error opening kprobe_events: %w", err)
	}
	defer f.Close()

	// See kprobe_events syntax above. Kprobe type does not need to be specified
	// for removals.
	pe := fmt.Sprintf("-:%s/%s", group, symbol)
	if _, err = f.WriteString(pe); err != nil {
		return fmt.Errorf("writing '%s' to kprobe_events: %w", pe, err)
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

func kprobePrefix(ret bool) string {
	if ret {
		return "r"
	}
	return "p"
}

package link

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/unix"
)

// Getting the terminology right is usually the hardest part. For posterity and
// for staying sane during implementation:
//
// - trace event: Representation of a kernel runtime hook. Filesystem entries
//   under <tracefs>/events. Can be tracepoints (static), kprobes or uprobes.
//   Can be instantiated into perf events (see below).
// - tracepoint: A predetermined hook point in the kernel. Exposed as trace
//   events in (sub)directories under <tracefs>/events. Cannot be closed or
//   removed, they are static.
// - k(ret)probe: Ephemeral trace events based on entry or exit points of
//   exported kernel symbols. kprobe-based (tracefs) trace events can be
//   created system-wide by writing to the <tracefs>/kprobe_events file, or
//   they can be scoped to the current process by creating PMU perf events.
// - perf event: An object instantiated based on an existing trace event or
//   kernel symbol. Referred to by fd in userspace.
//   Exactly one eBPF program can be attached to a perf event. Multiple perf
//   events can be created from a single trace event. Closing a perf event
//   stops any further invocations of the attached eBPF program.

var (
	tracefsPath = "/sys/kernel/debug/tracing"

	// Trace event groups, names and kernel symbols must adhere to this set
	// of characters. Non-empty, first character must not be a number, all
	// characters must be alphanumeric or underscore.
	rgxTraceEvent = regexp.MustCompile("^[a-zA-Z_][0-9a-zA-Z_]*$")

	errInvalidInput = errors.New("invalid input")
)

const (
	perfAllThreads = -1
)

// A perfEvent represents a perf event kernel object. Exactly one eBPF program
// can be attached to it. It is created based on a tracefs trace event or a
// Performance Monitoring Unit (PMU).
type perfEvent struct {

	// Group and name of the tracepoint/kprobe/uprobe.
	group string
	name  string

	// PMU event ID read from sysfs. Valid IDs are non-zero.
	pmuID uint64
	// ID of the trace event read from tracefs. Valid IDs are non-zero.
	tracefsID uint64

	// True for kretprobes/uretprobes.
	ret bool

	fd       *internal.FD
	progType ebpf.ProgramType
}

func (pe *perfEvent) isLink() {}

func (pe *perfEvent) Pin(string) error {
	return fmt.Errorf("pin perf event: %w", ErrNotSupported)
}

func (pe *perfEvent) Unpin() error {
	return fmt.Errorf("unpin perf event: %w", ErrNotSupported)
}

// Since 4.15 (e87c6bc3852b "bpf: permit multiple bpf attachments for a single perf event"),
// calling PERF_EVENT_IOC_SET_BPF appends the given program to a prog_array
// owned by the perf event, which means multiple programs can be attached
// simultaneously.
//
// Before 4.15, calling PERF_EVENT_IOC_SET_BPF more than once on a perf event
// returns EEXIST.
//
// Detaching a program from a perf event is currently not possible, so a
// program replacement mechanism cannot be implemented for perf events.
func (pe *perfEvent) Update(prog *ebpf.Program) error {
	return fmt.Errorf("can't replace eBPF program in perf event: %w", ErrNotSupported)
}

func (pe *perfEvent) Close() error {
	if pe.fd == nil {
		return nil
	}

	pfd, err := pe.fd.Value()
	if err != nil {
		return fmt.Errorf("getting perf event fd: %w", err)
	}

	err = unix.IoctlSetInt(int(pfd), unix.PERF_EVENT_IOC_DISABLE, 0)
	if err != nil {
		return fmt.Errorf("disabling perf event: %w", err)
	}

	err = pe.fd.Close()
	if err != nil {
		return fmt.Errorf("closing perf event fd: %w", err)
	}

	switch t := pe.progType; t {
	case ebpf.Kprobe:
		// For kprobes created using tracefs, clean up the <tracefs>/kprobe_events entry.
		if pe.tracefsID != 0 {
			return closeTraceFSKprobeEvent(pe.group, pe.name)
		}
	case ebpf.TracePoint:
		// Tracepoint trace events don't hold any extra resources.
		return nil
	}

	return nil
}

// attach the given eBPF prog to the perf event stored in pe.
// pe must contain a valid perf event fd.
// prog's type must match the program type stored in pe.
func (pe *perfEvent) attach(prog *ebpf.Program) error {
	if prog == nil {
		return errors.New("cannot attach a nil program")
	}
	if pe.fd == nil {
		return errors.New("cannot attach to nil perf event")
	}
	if t := prog.Type(); t != pe.progType {
		return fmt.Errorf("invalid program type (expected %s): %s", pe.progType, t)
	}
	if prog.FD() < 0 {
		return fmt.Errorf("invalid program: %w", internal.ErrClosedFd)
	}

	// The ioctl below will fail when the fd is invalid.
	kfd, _ := pe.fd.Value()

	// Assign the eBPF program to the perf event.
	err := unix.IoctlSetInt(int(kfd), unix.PERF_EVENT_IOC_SET_BPF, prog.FD())
	if err != nil {
		return fmt.Errorf("setting perf event bpf program: %w", err)
	}

	// PERF_EVENT_IOC_ENABLE and _DISABLE ignore their given values.
	if err := unix.IoctlSetInt(int(kfd), unix.PERF_EVENT_IOC_ENABLE, 0); err != nil {
		return fmt.Errorf("enable perf event: %s", err)
	}

	// Close the perf event when its reference is lost to avoid leaking system resources.
	runtime.SetFinalizer(pe, (*perfEvent).Close)
	return nil
}

// unsafeStringPtr returns an unsafe.Pointer to a NUL-terminated copy of str.
func unsafeStringPtr(str string) (unsafe.Pointer, error) {
	p, err := unix.BytePtrFromString(str)
	if err != nil {
		return nil, err
	}
	return unsafe.Pointer(p), nil
}

// getTraceEventID reads a trace event's ID from tracefs given its group and name.
// group and name must be alphanumeric or underscore, as required by the kernel.
func getTraceEventID(group, name string) (uint64, error) {
	tid, err := uint64FromFile(tracefsPath, "events", group, name, "id")
	if errors.Is(err, ErrNotSupported) {
		return 0, fmt.Errorf("trace event %s/%s: %w", group, name, ErrNotSupported)
	}
	if err != nil {
		return 0, fmt.Errorf("reading trace event ID of %s/%s: %w", group, name, err)
	}

	return tid, nil
}

// getPMUEventType reads a Performance Monitoring Unit's type (numeric identifier)
// from /sys/bus/event_source/devices/<pmu>/type.
func getPMUEventType(pmu string) (uint64, error) {
	et, err := uint64FromFile("/sys/bus/event_source/devices", pmu, "type")
	if errors.Is(err, ErrNotSupported) {
		return 0, fmt.Errorf("pmu type %s: %w", pmu, ErrNotSupported)
	}
	if err != nil {
		return 0, fmt.Errorf("reading pmu type %s: %w", pmu, err)
	}

	return et, nil
}

// openTracepointPerfEvent opens a tracepoint-type perf event. System-wide
// kprobes created by writing to <tracefs>/kprobe_events are tracepoints
// behind the scenes, and can be attached to using these perf events.
func openTracepointPerfEvent(tid uint64) (*internal.FD, error) {
	attr := unix.PerfEventAttr{
		Type:        unix.PERF_TYPE_TRACEPOINT,
		Config:      tid,
		Sample_type: unix.PERF_SAMPLE_RAW,
		Sample:      1,
		Wakeup:      1,
	}

	fd, err := unix.PerfEventOpen(&attr, perfAllThreads, 0, -1, unix.PERF_FLAG_FD_CLOEXEC)
	if err != nil {
		return nil, fmt.Errorf("opening tracepoint perf event: %w", err)
	}

	return internal.NewFD(uint32(fd)), nil
}

// uint64FromFile reads a uint64 from a file. All elements of path are sanitized
// and joined onto base. Returns error if base no longer prefixes the path after
// joining all components.
func uint64FromFile(base string, path ...string) (uint64, error) {

	// Resolve leaf path separately for error feedback. Makes the join onto
	// base more readable (can't mix with variadic args).
	l := filepath.Join(path...)

	p := filepath.Join(base, l)
	if !strings.HasPrefix(p, base) {
		return 0, fmt.Errorf("path '%s' attempts to escape base path '%s': %w", l, base, errInvalidInput)
	}

	data, err := ioutil.ReadFile(p)
	if os.IsNotExist(err) {
		// Only echo leaf path, the base path can be prepended at the call site
		// if more verbosity is required.
		return 0, fmt.Errorf("symbol %s: %w", l, ErrNotSupported)
	}
	if err != nil {
		return 0, fmt.Errorf("reading file %s: %w", p, err)
	}

	et := bytes.TrimSpace(data)
	return strconv.ParseUint(string(et), 10, 64)
}

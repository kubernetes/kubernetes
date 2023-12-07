package link

import (
	"bytes"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
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
// - u(ret)probe: Ephemeral trace events based on user provides ELF binaries
//   and offsets. uprobe-based (tracefs) trace events can be
//   created system-wide by writing to the <tracefs>/uprobe_events file, or
//   they can be scoped to the current process by creating PMU perf events.
// - perf event: An object instantiated based on an existing trace event or
//   kernel symbol. Referred to by fd in userspace.
//   Exactly one eBPF program can be attached to a perf event. Multiple perf
//   events can be created from a single trace event. Closing a perf event
//   stops any further invocations of the attached eBPF program.

var (
	tracefsPath = "/sys/kernel/debug/tracing"

	errInvalidInput = errors.New("invalid input")
)

const (
	perfAllThreads = -1
)

type perfEventType uint8

const (
	tracepointEvent perfEventType = iota
	kprobeEvent
	kretprobeEvent
	uprobeEvent
	uretprobeEvent
)

// A perfEvent represents a perf event kernel object. Exactly one eBPF program
// can be attached to it. It is created based on a tracefs trace event or a
// Performance Monitoring Unit (PMU).
type perfEvent struct {
	// The event type determines the types of programs that can be attached.
	typ perfEventType

	// Group and name of the tracepoint/kprobe/uprobe.
	group string
	name  string

	// PMU event ID read from sysfs. Valid IDs are non-zero.
	pmuID uint64
	// ID of the trace event read from tracefs. Valid IDs are non-zero.
	tracefsID uint64

	// User provided arbitrary value.
	cookie uint64

	// This is the perf event FD.
	fd *sys.FD
}

func (pe *perfEvent) Close() error {
	if err := pe.fd.Close(); err != nil {
		return fmt.Errorf("closing perf event fd: %w", err)
	}

	switch pe.typ {
	case kprobeEvent, kretprobeEvent:
		// Clean up kprobe tracefs entry.
		if pe.tracefsID != 0 {
			return closeTraceFSProbeEvent(kprobeType, pe.group, pe.name)
		}
	case uprobeEvent, uretprobeEvent:
		// Clean up uprobe tracefs entry.
		if pe.tracefsID != 0 {
			return closeTraceFSProbeEvent(uprobeType, pe.group, pe.name)
		}
	case tracepointEvent:
		// Tracepoint trace events don't hold any extra resources.
		return nil
	}

	return nil
}

// perfEventLink represents a bpf perf link.
type perfEventLink struct {
	RawLink
	pe *perfEvent
}

func (pl *perfEventLink) isLink() {}

// Pinning requires the underlying perf event FD to stay open.
//
// | PerfEvent FD | BpfLink FD | Works |
// |--------------|------------|-------|
// | Open         | Open       | Yes   |
// | Closed       | Open       | No    |
// | Open         | Closed     | No (Pin() -> EINVAL) |
// | Closed       | Closed     | No (Pin() -> EINVAL) |
//
// There is currently no pretty way to recover the perf event FD
// when loading a pinned link, so leave as not supported for now.
func (pl *perfEventLink) Pin(string) error {
	return fmt.Errorf("perf event link pin: %w", ErrNotSupported)
}

func (pl *perfEventLink) Unpin() error {
	return fmt.Errorf("perf event link unpin: %w", ErrNotSupported)
}

func (pl *perfEventLink) Close() error {
	if err := pl.pe.Close(); err != nil {
		return fmt.Errorf("perf event link close: %w", err)
	}
	return pl.fd.Close()
}

func (pl *perfEventLink) Update(prog *ebpf.Program) error {
	return fmt.Errorf("perf event link update: %w", ErrNotSupported)
}

// perfEventIoctl implements Link and handles the perf event lifecycle
// via ioctl().
type perfEventIoctl struct {
	*perfEvent
}

func (pi *perfEventIoctl) isLink() {}

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
func (pi *perfEventIoctl) Update(prog *ebpf.Program) error {
	return fmt.Errorf("perf event ioctl update: %w", ErrNotSupported)
}

func (pi *perfEventIoctl) Pin(string) error {
	return fmt.Errorf("perf event ioctl pin: %w", ErrNotSupported)
}

func (pi *perfEventIoctl) Unpin() error {
	return fmt.Errorf("perf event ioctl unpin: %w", ErrNotSupported)
}

func (pi *perfEventIoctl) Info() (*Info, error) {
	return nil, fmt.Errorf("perf event ioctl info: %w", ErrNotSupported)
}

// attach the given eBPF prog to the perf event stored in pe.
// pe must contain a valid perf event fd.
// prog's type must match the program type stored in pe.
func attachPerfEvent(pe *perfEvent, prog *ebpf.Program) (Link, error) {
	if prog == nil {
		return nil, errors.New("cannot attach a nil program")
	}
	if prog.FD() < 0 {
		return nil, fmt.Errorf("invalid program: %w", sys.ErrClosedFd)
	}

	switch pe.typ {
	case kprobeEvent, kretprobeEvent, uprobeEvent, uretprobeEvent:
		if t := prog.Type(); t != ebpf.Kprobe {
			return nil, fmt.Errorf("invalid program type (expected %s): %s", ebpf.Kprobe, t)
		}
	case tracepointEvent:
		if t := prog.Type(); t != ebpf.TracePoint {
			return nil, fmt.Errorf("invalid program type (expected %s): %s", ebpf.TracePoint, t)
		}
	default:
		return nil, fmt.Errorf("unknown perf event type: %d", pe.typ)
	}

	if err := haveBPFLinkPerfEvent(); err == nil {
		return attachPerfEventLink(pe, prog)
	}
	return attachPerfEventIoctl(pe, prog)
}

func attachPerfEventIoctl(pe *perfEvent, prog *ebpf.Program) (*perfEventIoctl, error) {
	if pe.cookie != 0 {
		return nil, fmt.Errorf("cookies are not supported: %w", ErrNotSupported)
	}

	// Assign the eBPF program to the perf event.
	err := unix.IoctlSetInt(pe.fd.Int(), unix.PERF_EVENT_IOC_SET_BPF, prog.FD())
	if err != nil {
		return nil, fmt.Errorf("setting perf event bpf program: %w", err)
	}

	// PERF_EVENT_IOC_ENABLE and _DISABLE ignore their given values.
	if err := unix.IoctlSetInt(pe.fd.Int(), unix.PERF_EVENT_IOC_ENABLE, 0); err != nil {
		return nil, fmt.Errorf("enable perf event: %s", err)
	}

	pi := &perfEventIoctl{pe}

	// Close the perf event when its reference is lost to avoid leaking system resources.
	runtime.SetFinalizer(pi, (*perfEventIoctl).Close)
	return pi, nil
}

// Use the bpf api to attach the perf event (BPF_LINK_TYPE_PERF_EVENT, 5.15+).
//
// https://github.com/torvalds/linux/commit/b89fbfbb854c9afc3047e8273cc3a694650b802e
func attachPerfEventLink(pe *perfEvent, prog *ebpf.Program) (*perfEventLink, error) {
	fd, err := sys.LinkCreatePerfEvent(&sys.LinkCreatePerfEventAttr{
		ProgFd:     uint32(prog.FD()),
		TargetFd:   pe.fd.Uint(),
		AttachType: sys.BPF_PERF_EVENT,
		BpfCookie:  pe.cookie,
	})
	if err != nil {
		return nil, fmt.Errorf("cannot create bpf perf link: %v", err)
	}

	pl := &perfEventLink{RawLink{fd: fd}, pe}

	// Close the perf event when its reference is lost to avoid leaking system resources.
	runtime.SetFinalizer(pl, (*perfEventLink).Close)
	return pl, nil
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
// The kernel requires group and name to be alphanumeric or underscore.
//
// name automatically has its invalid symbols converted to underscores so the caller
// can pass a raw symbol name, e.g. a kernel symbol containing dots.
func getTraceEventID(group, name string) (uint64, error) {
	name = sanitizeSymbol(name)
	tid, err := uint64FromFile(tracefsPath, "events", group, name, "id")
	if errors.Is(err, os.ErrNotExist) {
		return 0, fmt.Errorf("trace event %s/%s: %w", group, name, os.ErrNotExist)
	}
	if err != nil {
		return 0, fmt.Errorf("reading trace event ID of %s/%s: %w", group, name, err)
	}

	return tid, nil
}

// getPMUEventType reads a Performance Monitoring Unit's type (numeric identifier)
// from /sys/bus/event_source/devices/<pmu>/type.
//
// Returns ErrNotSupported if the pmu type is not supported.
func getPMUEventType(typ probeType) (uint64, error) {
	et, err := uint64FromFile("/sys/bus/event_source/devices", typ.String(), "type")
	if errors.Is(err, os.ErrNotExist) {
		return 0, fmt.Errorf("pmu type %s: %w", typ, ErrNotSupported)
	}
	if err != nil {
		return 0, fmt.Errorf("reading pmu type %s: %w", typ, err)
	}

	return et, nil
}

// openTracepointPerfEvent opens a tracepoint-type perf event. System-wide
// [k,u]probes created by writing to <tracefs>/[k,u]probe_events are tracepoints
// behind the scenes, and can be attached to using these perf events.
func openTracepointPerfEvent(tid uint64, pid int) (*sys.FD, error) {
	attr := unix.PerfEventAttr{
		Type:        unix.PERF_TYPE_TRACEPOINT,
		Config:      tid,
		Sample_type: unix.PERF_SAMPLE_RAW,
		Sample:      1,
		Wakeup:      1,
	}

	fd, err := unix.PerfEventOpen(&attr, pid, 0, -1, unix.PERF_FLAG_FD_CLOEXEC)
	if err != nil {
		return nil, fmt.Errorf("opening tracepoint perf event: %w", err)
	}

	return sys.NewFD(fd)
}

// uint64FromFile reads a uint64 from a file. All elements of path are sanitized
// and joined onto base. Returns error if base no longer prefixes the path after
// joining all components.
func uint64FromFile(base string, path ...string) (uint64, error) {
	l := filepath.Join(path...)
	p := filepath.Join(base, l)
	if !strings.HasPrefix(p, base) {
		return 0, fmt.Errorf("path '%s' attempts to escape base path '%s': %w", l, base, errInvalidInput)
	}

	data, err := os.ReadFile(p)
	if err != nil {
		return 0, fmt.Errorf("reading file %s: %w", p, err)
	}

	et := bytes.TrimSpace(data)
	return strconv.ParseUint(string(et), 10, 64)
}

// Probe BPF perf link.
//
// https://elixir.bootlin.com/linux/v5.16.8/source/kernel/bpf/syscall.c#L4307
// https://github.com/torvalds/linux/commit/b89fbfbb854c9afc3047e8273cc3a694650b802e
var haveBPFLinkPerfEvent = internal.FeatureTest("bpf_link_perf_event", "5.15", func() error {
	prog, err := ebpf.NewProgram(&ebpf.ProgramSpec{
		Name: "probe_bpf_perf_link",
		Type: ebpf.Kprobe,
		Instructions: asm.Instructions{
			asm.Mov.Imm(asm.R0, 0),
			asm.Return(),
		},
		License: "MIT",
	})
	if err != nil {
		return err
	}
	defer prog.Close()

	_, err = sys.LinkCreatePerfEvent(&sys.LinkCreatePerfEventAttr{
		ProgFd:     uint32(prog.FD()),
		AttachType: sys.BPF_PERF_EVENT,
	})
	if errors.Is(err, unix.EINVAL) {
		return internal.ErrNotSupported
	}
	if errors.Is(err, unix.EBADF) {
		return nil
	}
	return err
})

// isValidTraceID implements the equivalent of a regex match
// against "^[a-zA-Z_][0-9a-zA-Z_]*$".
//
// Trace event groups, names and kernel symbols must adhere to this set
// of characters. Non-empty, first character must not be a number, all
// characters must be alphanumeric or underscore.
func isValidTraceID(s string) bool {
	if len(s) < 1 {
		return false
	}
	for i, c := range []byte(s) {
		switch {
		case c >= 'a' && c <= 'z':
		case c >= 'A' && c <= 'Z':
		case c == '_':
		case i > 0 && c >= '0' && c <= '9':

		default:
			return false
		}
	}

	return true
}

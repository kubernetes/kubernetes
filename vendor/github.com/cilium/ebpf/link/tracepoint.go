package link

import (
	"fmt"

	"github.com/cilium/ebpf"
)

// TracepointOptions defines additional parameters that will be used
// when loading Tracepoints.
type TracepointOptions struct {
	// Arbitrary value that can be fetched from an eBPF program
	// via `bpf_get_attach_cookie()`.
	//
	// Needs kernel 5.15+.
	Cookie uint64
}

// Tracepoint attaches the given eBPF program to the tracepoint with the given
// group and name. See /sys/kernel/debug/tracing/events to find available
// tracepoints. The top-level directory is the group, the event's subdirectory
// is the name. Example:
//
//	tp, err := Tracepoint("syscalls", "sys_enter_fork", prog, nil)
//
// Losing the reference to the resulting Link (tp) will close the Tracepoint
// and prevent further execution of prog. The Link must be Closed during
// program shutdown to avoid leaking system resources.
//
// Note that attaching eBPF programs to syscalls (sys_enter_*/sys_exit_*) is
// only possible as of kernel 4.14 (commit cf5f5ce).
func Tracepoint(group, name string, prog *ebpf.Program, opts *TracepointOptions) (Link, error) {
	if group == "" || name == "" {
		return nil, fmt.Errorf("group and name cannot be empty: %w", errInvalidInput)
	}
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if !isValidTraceID(group) || !isValidTraceID(name) {
		return nil, fmt.Errorf("group and name '%s/%s' must be alphanumeric or underscore: %w", group, name, errInvalidInput)
	}
	if prog.Type() != ebpf.TracePoint {
		return nil, fmt.Errorf("eBPF program type %s is not a Tracepoint: %w", prog.Type(), errInvalidInput)
	}

	tid, err := getTraceEventID(group, name)
	if err != nil {
		return nil, err
	}

	fd, err := openTracepointPerfEvent(tid, perfAllThreads)
	if err != nil {
		return nil, err
	}

	var cookie uint64
	if opts != nil {
		cookie = opts.Cookie
	}

	pe := &perfEvent{
		typ:       tracepointEvent,
		group:     group,
		name:      name,
		tracefsID: tid,
		cookie:    cookie,
		fd:        fd,
	}

	lnk, err := attachPerfEvent(pe, prog)
	if err != nil {
		pe.Close()
		return nil, err
	}

	return lnk, nil
}

package link

import (
	"fmt"

	"github.com/cilium/ebpf"
)

// Tracepoint attaches the given eBPF program to the tracepoint with the given
// group and name. See /sys/kernel/debug/tracing/events to find available
// tracepoints. The top-level directory is the group, the event's subdirectory
// is the name. Example:
//
//	Tracepoint("syscalls", "sys_enter_fork", prog)
//
// Note that attaching eBPF programs to syscalls (sys_enter_*/sys_exit_*) is
// only possible as of kernel 4.14 (commit cf5f5ce).
func Tracepoint(group, name string, prog *ebpf.Program) (Link, error) {
	if group == "" || name == "" {
		return nil, fmt.Errorf("group and name cannot be empty: %w", errInvalidInput)
	}
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if !rgxTraceEvent.MatchString(group) || !rgxTraceEvent.MatchString(name) {
		return nil, fmt.Errorf("group and name '%s/%s' must be alphanumeric or underscore: %w", group, name, errInvalidInput)
	}
	if prog.Type() != ebpf.TracePoint {
		return nil, fmt.Errorf("eBPF program type %s is not a Tracepoint: %w", prog.Type(), errInvalidInput)
	}

	tid, err := getTraceEventID(group, name)
	if err != nil {
		return nil, err
	}

	fd, err := openTracepointPerfEvent(tid)
	if err != nil {
		return nil, err
	}

	pe := &perfEvent{
		fd:        fd,
		tracefsID: tid,
		group:     group,
		name:      name,
		typ:       tracepointEvent,
	}

	if err := pe.attach(prog); err != nil {
		pe.Close()
		return nil, err
	}

	return pe, nil
}

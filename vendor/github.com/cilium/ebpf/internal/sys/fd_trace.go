package sys

import (
	"bytes"
	"fmt"
	"runtime"
	"sync"
)

// OnLeakFD controls tracing [FD] lifetime to detect resources that are not
// closed by Close().
//
// If fn is not nil, tracing is enabled for all FDs created going forward. fn is
// invoked for all FDs that are closed by the garbage collector instead of an
// explicit Close() by a caller. Calling OnLeakFD twice with a non-nil fn
// (without disabling tracing in the meantime) will cause a panic.
//
// If fn is nil, tracing will be disabled. Any FDs that have not been closed are
// considered to be leaked, fn will be invoked for them, and the process will be
// terminated.
//
// fn will be invoked at most once for every unique sys.FD allocation since a
// runtime.Frames can only be unwound once.
func OnLeakFD(fn func(*runtime.Frames)) {
	// Enable leak tracing if new fn is provided.
	if fn != nil {
		if onLeakFD != nil {
			panic("OnLeakFD called twice with non-nil fn")
		}

		onLeakFD = fn
		return
	}

	// fn is nil past this point.

	if onLeakFD == nil {
		return
	}

	// Call onLeakFD for all open fds.
	if fs := flushFrames(); len(fs) != 0 {
		for _, f := range fs {
			onLeakFD(f)
		}
	}

	onLeakFD = nil
}

var onLeakFD func(*runtime.Frames)

// fds is a registry of all file descriptors wrapped into sys.fds that were
// created while an fd tracer was active.
var fds sync.Map // map[int]*runtime.Frames

// flushFrames removes all elements from fds and returns them as a slice. This
// deals with the fact that a runtime.Frames can only be unwound once using
// Next().
func flushFrames() []*runtime.Frames {
	var frames []*runtime.Frames
	fds.Range(func(key, value any) bool {
		frames = append(frames, value.(*runtime.Frames))
		fds.Delete(key)
		return true
	})
	return frames
}

func callersFrames() *runtime.Frames {
	c := make([]uintptr, 32)

	// Skip runtime.Callers and this function.
	i := runtime.Callers(2, c)
	if i == 0 {
		return nil
	}

	return runtime.CallersFrames(c)
}

// FormatFrames formats a runtime.Frames as a human-readable string.
func FormatFrames(fs *runtime.Frames) string {
	var b bytes.Buffer
	for {
		f, more := fs.Next()
		b.WriteString(fmt.Sprintf("\t%s+%#x\n\t\t%s:%d\n", f.Function, f.PC-f.Entry, f.File, f.Line))
		if !more {
			break
		}
	}
	return b.String()
}

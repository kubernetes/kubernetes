// Copyright (c) 2016 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package zapcore

import (
	"fmt"
	"runtime"
	"strings"
	"time"

	"go.uber.org/multierr"
	"go.uber.org/zap/internal/bufferpool"
	"go.uber.org/zap/internal/exit"
	"go.uber.org/zap/internal/pool"
)

var _cePool = pool.New(func() *CheckedEntry {
	// Pre-allocate some space for cores.
	return &CheckedEntry{
		cores: make([]Core, 4),
	}
})

func getCheckedEntry() *CheckedEntry {
	ce := _cePool.Get()
	ce.reset()
	return ce
}

func putCheckedEntry(ce *CheckedEntry) {
	if ce == nil {
		return
	}
	_cePool.Put(ce)
}

// NewEntryCaller makes an EntryCaller from the return signature of
// runtime.Caller.
func NewEntryCaller(pc uintptr, file string, line int, ok bool) EntryCaller {
	if !ok {
		return EntryCaller{}
	}
	return EntryCaller{
		PC:      pc,
		File:    file,
		Line:    line,
		Defined: true,
	}
}

// EntryCaller represents the caller of a logging function.
type EntryCaller struct {
	Defined  bool
	PC       uintptr
	File     string
	Line     int
	Function string
}

// String returns the full path and line number of the caller.
func (ec EntryCaller) String() string {
	return ec.FullPath()
}

// FullPath returns a /full/path/to/package/file:line description of the
// caller.
func (ec EntryCaller) FullPath() string {
	if !ec.Defined {
		return "undefined"
	}
	buf := bufferpool.Get()
	buf.AppendString(ec.File)
	buf.AppendByte(':')
	buf.AppendInt(int64(ec.Line))
	caller := buf.String()
	buf.Free()
	return caller
}

// TrimmedPath returns a package/file:line description of the caller,
// preserving only the leaf directory name and file name.
func (ec EntryCaller) TrimmedPath() string {
	if !ec.Defined {
		return "undefined"
	}
	// nb. To make sure we trim the path correctly on Windows too, we
	// counter-intuitively need to use '/' and *not* os.PathSeparator here,
	// because the path given originates from Go stdlib, specifically
	// runtime.Caller() which (as of Mar/17) returns forward slashes even on
	// Windows.
	//
	// See https://github.com/golang/go/issues/3335
	// and https://github.com/golang/go/issues/18151
	//
	// for discussion on the issue on Go side.
	//
	// Find the last separator.
	//
	idx := strings.LastIndexByte(ec.File, '/')
	if idx == -1 {
		return ec.FullPath()
	}
	// Find the penultimate separator.
	idx = strings.LastIndexByte(ec.File[:idx], '/')
	if idx == -1 {
		return ec.FullPath()
	}
	buf := bufferpool.Get()
	// Keep everything after the penultimate separator.
	buf.AppendString(ec.File[idx+1:])
	buf.AppendByte(':')
	buf.AppendInt(int64(ec.Line))
	caller := buf.String()
	buf.Free()
	return caller
}

// An Entry represents a complete log message. The entry's structured context
// is already serialized, but the log level, time, message, and call site
// information are available for inspection and modification. Any fields left
// empty will be omitted when encoding.
//
// Entries are pooled, so any functions that accept them MUST be careful not to
// retain references to them.
type Entry struct {
	Level      Level
	Time       time.Time
	LoggerName string
	Message    string
	Caller     EntryCaller
	Stack      string
}

// CheckWriteHook is a custom action that may be executed after an entry is
// written.
//
// Register one on a CheckedEntry with the After method.
//
//	if ce := logger.Check(...); ce != nil {
//	  ce = ce.After(hook)
//	  ce.Write(...)
//	}
//
// You can configure the hook for Fatal log statements at the logger level with
// the zap.WithFatalHook option.
type CheckWriteHook interface {
	// OnWrite is invoked with the CheckedEntry that was written and a list
	// of fields added with that entry.
	//
	// The list of fields DOES NOT include fields that were already added
	// to the logger with the With method.
	OnWrite(*CheckedEntry, []Field)
}

// CheckWriteAction indicates what action to take after a log entry is
// processed. Actions are ordered in increasing severity.
type CheckWriteAction uint8

const (
	// WriteThenNoop indicates that nothing special needs to be done. It's the
	// default behavior.
	WriteThenNoop CheckWriteAction = iota
	// WriteThenGoexit runs runtime.Goexit after Write.
	WriteThenGoexit
	// WriteThenPanic causes a panic after Write.
	WriteThenPanic
	// WriteThenFatal causes an os.Exit(1) after Write.
	WriteThenFatal
)

// OnWrite implements the OnWrite method to keep CheckWriteAction compatible
// with the new CheckWriteHook interface which deprecates CheckWriteAction.
func (a CheckWriteAction) OnWrite(ce *CheckedEntry, _ []Field) {
	switch a {
	case WriteThenGoexit:
		runtime.Goexit()
	case WriteThenPanic:
		panic(ce.Message)
	case WriteThenFatal:
		exit.With(1)
	}
}

var _ CheckWriteHook = CheckWriteAction(0)

// CheckedEntry is an Entry together with a collection of Cores that have
// already agreed to log it.
//
// CheckedEntry references should be created by calling AddCore or After on a
// nil *CheckedEntry. References are returned to a pool after Write, and MUST
// NOT be retained after calling their Write method.
type CheckedEntry struct {
	Entry
	ErrorOutput WriteSyncer
	dirty       bool // best-effort detection of pool misuse
	after       CheckWriteHook
	cores       []Core
}

func (ce *CheckedEntry) reset() {
	ce.Entry = Entry{}
	ce.ErrorOutput = nil
	ce.dirty = false
	ce.after = nil
	for i := range ce.cores {
		// don't keep references to cores
		ce.cores[i] = nil
	}
	ce.cores = ce.cores[:0]
}

// Write writes the entry to the stored Cores, returns any errors, and returns
// the CheckedEntry reference to a pool for immediate re-use. Finally, it
// executes any required CheckWriteAction.
func (ce *CheckedEntry) Write(fields ...Field) {
	if ce == nil {
		return
	}

	if ce.dirty {
		if ce.ErrorOutput != nil {
			// Make a best effort to detect unsafe re-use of this CheckedEntry.
			// If the entry is dirty, log an internal error; because the
			// CheckedEntry is being used after it was returned to the pool,
			// the message may be an amalgamation from multiple call sites.
			fmt.Fprintf(ce.ErrorOutput, "%v Unsafe CheckedEntry re-use near Entry %+v.\n", ce.Time, ce.Entry)
			_ = ce.ErrorOutput.Sync() // ignore error
		}
		return
	}
	ce.dirty = true

	var err error
	for i := range ce.cores {
		err = multierr.Append(err, ce.cores[i].Write(ce.Entry, fields))
	}
	if err != nil && ce.ErrorOutput != nil {
		fmt.Fprintf(ce.ErrorOutput, "%v write error: %v\n", ce.Time, err)
		_ = ce.ErrorOutput.Sync() // ignore error
	}

	hook := ce.after
	if hook != nil {
		hook.OnWrite(ce, fields)
	}
	putCheckedEntry(ce)
}

// AddCore adds a Core that has agreed to log this CheckedEntry. It's intended to be
// used by Core.Check implementations, and is safe to call on nil CheckedEntry
// references.
func (ce *CheckedEntry) AddCore(ent Entry, core Core) *CheckedEntry {
	if ce == nil {
		ce = getCheckedEntry()
		ce.Entry = ent
	}
	ce.cores = append(ce.cores, core)
	return ce
}

// Should sets this CheckedEntry's CheckWriteAction, which controls whether a
// Core will panic or fatal after writing this log entry. Like AddCore, it's
// safe to call on nil CheckedEntry references.
//
// Deprecated: Use [CheckedEntry.After] instead.
func (ce *CheckedEntry) Should(ent Entry, should CheckWriteAction) *CheckedEntry {
	return ce.After(ent, should)
}

// After sets this CheckEntry's CheckWriteHook, which will be called after this
// log entry has been written. It's safe to call this on nil CheckedEntry
// references.
func (ce *CheckedEntry) After(ent Entry, hook CheckWriteHook) *CheckedEntry {
	if ce == nil {
		ce = getCheckedEntry()
		ce.Entry = ent
	}
	ce.after = hook
	return ce
}

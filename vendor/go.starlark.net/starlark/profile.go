// Copyright 2019 The Bazel Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package starlark

// This file defines a simple execution-time profiler for Starlark.
// It measures the wall time spent executing Starlark code, and emits a
// gzipped protocol message in pprof format (github.com/google/pprof).
//
// When profiling is enabled, the interpreter calls the profiler to
// indicate the start and end of each "span" or time interval. A leaf
// function (whether Go or Starlark) has a single span. A function that
// calls another function has spans for each interval in which it is the
// top of the stack. (A LOAD instruction also ends a span.)
//
// At the start of a span, the interpreter records the current time in
// the thread's topmost frame. At the end of the span, it obtains the
// time again and subtracts the span start time. The difference is added
// to an accumulator variable in the thread. If the accumulator exceeds
// some fixed quantum (10ms, say), the profiler records the current call
// stack and sends it to the profiler goroutine, along with the number
// of quanta, which are subtracted. For example, if the accumulator
// holds 3ms and then a completed span adds 25ms to it, its value is 28ms,
// which exceeeds 10ms. The profiler records a stack with the value 20ms
// (2 quanta), and the accumulator is left with 8ms.
//
// The profiler goroutine converts the stacks into the pprof format and
// emits a gzip-compressed protocol message to the designated output
// file. We use a hand-written streaming proto encoder to avoid
// dependencies on pprof and proto, and to avoid the need to
// materialize the profile data structure in memory.
//
// A limitation of this profiler is that it measures wall time, which
// does not necessarily correspond to CPU time. A CPU profiler requires
// that only running (not runnable) threads are sampled; this is
// commonly achieved by having the kernel deliver a (PROF) signal to an
// arbitrary running thread, through setitimer(2). The CPU profiler in the
// Go runtime uses this mechanism, but it is not possible for a Go
// application to register a SIGPROF handler, nor is it possible for a
// Go handler for some other signal to read the stack pointer of
// the interrupted thread.
//
// Two caveats:
// (1) it is tempting to send the leaf Frame directly to the profiler
// goroutine instead of making a copy of the stack, since a Frame is a
// spaghetti stack--a linked list. However, as soon as execution
// resumes, the stack's Frame.pc values may be mutated, so Frames are
// not safe to share with the asynchronous profiler goroutine.
// (2) it is tempting to use Callables as keys in a map when tabulating
// the pprof protocols's Function entities. However, we cannot assume
// that Callables are valid map keys, and furthermore we must not
// pin function values in memory indefinitely as this may cause lambda
// values to keep their free variables live much longer than necessary.

// TODO(adonovan):
// - make Start/Stop fully thread-safe.
// - fix the pc hack.
// - experiment with other values of quantum.

import (
	"bufio"
	"bytes"
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"reflect"
	"sync/atomic"
	"time"
	"unsafe"

	"go.starlark.net/syntax"
)

// StartProfile enables time profiling of all Starlark threads,
// and writes a profile in pprof format to w.
// It must be followed by a call to StopProfiler to stop
// the profiler and finalize the profile.
//
// StartProfile returns an error if profiling was already enabled.
//
// StartProfile must not be called concurrently with Starlark execution.
func StartProfile(w io.Writer) error {
	if !atomic.CompareAndSwapUint32(&profiler.on, 0, 1) {
		return fmt.Errorf("profiler already running")
	}

	// TODO(adonovan): make the API fully concurrency-safe.
	// The main challenge is racy reads/writes of profiler.events,
	// and of send/close races on the channel it refers to.
	// It's easy to solve them with a mutex but harder to do
	// it efficiently.

	profiler.events = make(chan *profEvent, 1)
	profiler.done = make(chan error)

	go profile(w)

	return nil
}

// StopProfiler stops the profiler started by a prior call to
// StartProfile and finalizes the profile. It returns an error if the
// profile could not be completed.
//
// StopProfiler must not be called concurrently with Starlark execution.
func StopProfile() error {
	// Terminate the profiler goroutine and get its result.
	close(profiler.events)
	err := <-profiler.done

	profiler.done = nil
	profiler.events = nil
	atomic.StoreUint32(&profiler.on, 0)

	return err
}

// globals
var profiler struct {
	on     uint32          // nonzero => profiler running
	events chan *profEvent // profile events from interpreter threads
	done   chan error      // indicates profiler goroutine is ready
}

func (thread *Thread) beginProfSpan() {
	if profiler.events == nil {
		return // profiling not enabled
	}

	thread.frameAt(0).spanStart = nanotime()
}

// TODO(adonovan): experiment with smaller values,
// which trade space and time for greater precision.
const quantum = 10 * time.Millisecond

func (thread *Thread) endProfSpan() {
	if profiler.events == nil {
		return // profiling not enabled
	}

	// Add the span to the thread's accumulator.
	thread.proftime += time.Duration(nanotime() - thread.frameAt(0).spanStart)
	if thread.proftime < quantum {
		return
	}

	// Only record complete quanta.
	n := thread.proftime / quantum
	thread.proftime -= n * quantum

	// Copy the stack.
	// (We can't save thread.frame because its pc will change.)
	ev := &profEvent{
		thread: thread,
		time:   n * quantum,
	}
	ev.stack = ev.stackSpace[:0]
	for i := range thread.stack {
		fr := thread.frameAt(i)
		ev.stack = append(ev.stack, profFrame{
			pos: fr.Position(),
			fn:  fr.Callable(),
			pc:  fr.pc,
		})
	}

	profiler.events <- ev
}

type profEvent struct {
	thread     *Thread // currently unused
	time       time.Duration
	stack      []profFrame
	stackSpace [8]profFrame // initial space for stack
}

type profFrame struct {
	fn  Callable        // don't hold this live for too long (prevents GC of lambdas)
	pc  uint32          // program counter (Starlark frames only)
	pos syntax.Position // position of pc within this frame
}

// profile is the profiler goroutine.
// It runs until StopProfiler is called.
func profile(w io.Writer) {
	// Field numbers from pprof protocol.
	// See https://github.com/google/pprof/blob/master/proto/profile.proto
	const (
		Profile_sample_type    = 1  // repeated ValueType
		Profile_sample         = 2  // repeated Sample
		Profile_mapping        = 3  // repeated Mapping
		Profile_location       = 4  // repeated Location
		Profile_function       = 5  // repeated Function
		Profile_string_table   = 6  // repeated string
		Profile_time_nanos     = 9  // int64
		Profile_duration_nanos = 10 // int64
		Profile_period_type    = 11 // ValueType
		Profile_period         = 12 // int64

		ValueType_type = 1 // int64
		ValueType_unit = 2 // int64

		Sample_location_id = 1 // repeated uint64
		Sample_value       = 2 // repeated int64
		Sample_label       = 3 // repeated Label

		Label_key      = 1 // int64
		Label_str      = 2 // int64
		Label_num      = 3 // int64
		Label_num_unit = 4 // int64

		Location_id         = 1 // uint64
		Location_mapping_id = 2 // uint64
		Location_address    = 3 // uint64
		Location_line       = 4 // repeated Line

		Line_function_id = 1 // uint64
		Line_line        = 2 // int64

		Function_id          = 1 // uint64
		Function_name        = 2 // int64
		Function_system_name = 3 // int64
		Function_filename    = 4 // int64
		Function_start_line  = 5 // int64
	)

	bufw := bufio.NewWriter(w) // write file in 4KB (not 240B flate-sized) chunks
	gz := gzip.NewWriter(bufw)
	enc := protoEncoder{w: gz}

	// strings
	stringIndex := make(map[string]int64)
	str := func(s string) int64 {
		i, ok := stringIndex[s]
		if !ok {
			i = int64(len(stringIndex))
			enc.string(Profile_string_table, s)
			stringIndex[s] = i
		}
		return i
	}
	str("") // entry 0

	// functions
	//
	// function returns the ID of a Callable for use in Line.FunctionId.
	// The ID is the same as the function's logical address,
	// which is supplied by the caller to avoid the need to recompute it.
	functionId := make(map[uintptr]uint64)
	function := func(fn Callable, addr uintptr) uint64 {
		id, ok := functionId[addr]
		if !ok {
			id = uint64(addr)

			var pos syntax.Position
			if fn, ok := fn.(callableWithPosition); ok {
				pos = fn.Position()
			}

			name := fn.Name()
			if name == "<toplevel>" {
				name = pos.Filename()
			}

			nameIndex := str(name)

			fun := new(bytes.Buffer)
			funenc := protoEncoder{w: fun}
			funenc.uint(Function_id, id)
			funenc.int(Function_name, nameIndex)
			funenc.int(Function_system_name, nameIndex)
			funenc.int(Function_filename, str(pos.Filename()))
			funenc.int(Function_start_line, int64(pos.Line))
			enc.bytes(Profile_function, fun.Bytes())

			functionId[addr] = id
		}
		return id
	}

	// locations
	//
	// location returns the ID of the location denoted by fr.
	// For Starlark frames, this is the Frame pc.
	locationId := make(map[uintptr]uint64)
	location := func(fr profFrame) uint64 {
		fnAddr := profFuncAddr(fr.fn)

		// For Starlark functions, the frame position
		// represents the current PC value.
		// Mix it into the low bits of the address.
		// This is super hacky and may result in collisions
		// in large functions or if functions are numerous.
		// TODO(adonovan): fix: try making this cleaner by treating
		// each bytecode segment as a Profile.Mapping.
		pcAddr := fnAddr
		if _, ok := fr.fn.(*Function); ok {
			pcAddr = (pcAddr << 16) ^ uintptr(fr.pc)
		}

		id, ok := locationId[pcAddr]
		if !ok {
			id = uint64(pcAddr)

			line := new(bytes.Buffer)
			lineenc := protoEncoder{w: line}
			lineenc.uint(Line_function_id, function(fr.fn, fnAddr))
			lineenc.int(Line_line, int64(fr.pos.Line))
			loc := new(bytes.Buffer)
			locenc := protoEncoder{w: loc}
			locenc.uint(Location_id, id)
			locenc.uint(Location_address, uint64(pcAddr))
			locenc.bytes(Location_line, line.Bytes())
			enc.bytes(Profile_location, loc.Bytes())

			locationId[pcAddr] = id
		}
		return id
	}

	wallNanos := new(bytes.Buffer)
	wnenc := protoEncoder{w: wallNanos}
	wnenc.int(ValueType_type, str("wall"))
	wnenc.int(ValueType_unit, str("nanoseconds"))

	// informational fields of Profile
	enc.bytes(Profile_sample_type, wallNanos.Bytes())
	enc.int(Profile_period, quantum.Nanoseconds())     // magnitude of sampling period
	enc.bytes(Profile_period_type, wallNanos.Bytes())  // dimension and unit of period
	enc.int(Profile_time_nanos, time.Now().UnixNano()) // start (real) time of profile

	startNano := nanotime()

	// Read profile events from the channel
	// until it is closed by StopProfiler.
	for e := range profiler.events {
		sample := new(bytes.Buffer)
		sampleenc := protoEncoder{w: sample}
		sampleenc.int(Sample_value, e.time.Nanoseconds()) // wall nanoseconds
		for _, fr := range e.stack {
			sampleenc.uint(Sample_location_id, location(fr))
		}
		enc.bytes(Profile_sample, sample.Bytes())
	}

	endNano := nanotime()
	enc.int(Profile_duration_nanos, endNano-startNano)

	err := gz.Close() // Close reports any prior write error
	if flushErr := bufw.Flush(); err == nil {
		err = flushErr
	}
	profiler.done <- err
}

// nanotime returns the time in nanoseconds since epoch.
// It is implemented by runtime.nanotime using the linkname hack;
// runtime.nanotime is defined for all OSs/ARCHS and uses the
// monotonic system clock, which there is no portable way to access.
// Should that function ever go away, these alternatives exist:
//
// 	// POSIX only. REALTIME not MONOTONIC. 17ns.
// 	var tv syscall.Timeval
// 	syscall.Gettimeofday(&tv) // can't fail
// 	return tv.Nano()
//
// 	// Portable. REALTIME not MONOTONIC. 46ns.
// 	return time.Now().Nanoseconds()
//
//      // POSIX only. Adds a dependency.
//	import "golang.org/x/sys/unix"
//	var ts unix.Timespec
// 	unix.ClockGettime(CLOCK_MONOTONIC, &ts) // can't fail
//	return unix.TimespecToNsec(ts)
//
//go:linkname nanotime runtime.nanotime
func nanotime() int64

// profFuncAddr returns the canonical "address"
// of a Callable for use by the profiler.
func profFuncAddr(fn Callable) uintptr {
	switch fn := fn.(type) {
	case *Builtin:
		return reflect.ValueOf(fn.fn).Pointer()
	case *Function:
		return uintptr(unsafe.Pointer(fn.funcode))
	}

	// User-defined callable types are typically of
	// of kind pointer-to-struct. Handle them specially.
	if v := reflect.ValueOf(fn); v.Type().Kind() == reflect.Ptr {
		return v.Pointer()
	}

	// Address zero is reserved by the protocol.
	// Use 1 for callables we don't recognize.
	log.Printf("Starlark profiler: no address for Callable %T", fn)
	return 1
}

// We encode the protocol message by hand to avoid making
// the interpreter depend on both github.com/google/pprof
// and github.com/golang/protobuf.
//
// This also avoids the need to materialize a protocol message object
// tree of unbounded size and serialize it all at the end.
// The pprof format appears to have been designed to
// permit streaming implementations such as this one.
//
// See https://developers.google.com/protocol-buffers/docs/encoding.
type protoEncoder struct {
	w   io.Writer // *bytes.Buffer or *gzip.Writer
	tmp [binary.MaxVarintLen64]byte
}

func (e *protoEncoder) uvarint(x uint64) {
	n := binary.PutUvarint(e.tmp[:], x)
	e.w.Write(e.tmp[:n])
}

func (e *protoEncoder) tag(field, wire uint) {
	e.uvarint(uint64(field<<3 | wire))
}

func (e *protoEncoder) string(field uint, s string) {
	e.tag(field, 2) // length-delimited
	e.uvarint(uint64(len(s)))
	io.WriteString(e.w, s)
}

func (e *protoEncoder) bytes(field uint, b []byte) {
	e.tag(field, 2) // length-delimited
	e.uvarint(uint64(len(b)))
	e.w.Write(b)
}

func (e *protoEncoder) uint(field uint, x uint64) {
	e.tag(field, 0) // varint
	e.uvarint(x)
}

func (e *protoEncoder) int(field uint, x int64) {
	e.tag(field, 0) // varint
	e.uvarint(uint64(x))
}

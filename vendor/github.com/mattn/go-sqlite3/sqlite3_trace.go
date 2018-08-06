// Copyright (C) 2016 Yasuhiro Matsumoto <mattn.jp@gmail.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.
// +build trace

package sqlite3

/*
#ifndef USE_LIBSQLITE3
#include <sqlite3-binding.h>
#else
#include <sqlite3.h>
#endif
#include <stdlib.h>

int traceCallbackTrampoline(unsigned int traceEventCode, void *ctx, void *p, void *x);
*/
import "C"

import (
	"fmt"
	"strings"
	"sync"
	"unsafe"
)

// Trace... constants identify the possible events causing callback invocation.
// Values are same as the corresponding SQLite Trace Event Codes.
const (
	TraceStmt    = C.SQLITE_TRACE_STMT
	TraceProfile = C.SQLITE_TRACE_PROFILE
	TraceRow     = C.SQLITE_TRACE_ROW
	TraceClose   = C.SQLITE_TRACE_CLOSE
)

type TraceInfo struct {
	// Pack together the shorter fields, to keep the struct smaller.
	// On a 64-bit machine there would be padding
	// between EventCode and ConnHandle; having AutoCommit here is "free":
	EventCode  uint32
	AutoCommit bool
	ConnHandle uintptr

	// Usually filled, unless EventCode = TraceClose = SQLITE_TRACE_CLOSE:
	// identifier for a prepared statement:
	StmtHandle uintptr

	// Two strings filled when EventCode = TraceStmt = SQLITE_TRACE_STMT:
	// (1) either the unexpanded SQL text of the prepared statement, or
	//     an SQL comment that indicates the invocation of a trigger;
	// (2) expanded SQL, if requested and if (1) is not an SQL comment.
	StmtOrTrigger string
	ExpandedSQL   string // only if requested (TraceConfig.WantExpandedSQL = true)

	// filled when EventCode = TraceProfile = SQLITE_TRACE_PROFILE:
	// estimated number of nanoseconds that the prepared statement took to run:
	RunTimeNanosec int64

	DBError Error
}

// TraceUserCallback gives the signature for a trace function
// provided by the user (Go application programmer).
// SQLite 3.14 documentation (as of September 2, 2016)
// for SQL Trace Hook = sqlite3_trace_v2():
// The integer return value from the callback is currently ignored,
// though this may change in future releases. Callback implementations
// should return zero to ensure future compatibility.
type TraceUserCallback func(TraceInfo) int

type TraceConfig struct {
	Callback        TraceUserCallback
	EventMask       C.uint
	WantExpandedSQL bool
}

func fillDBError(dbErr *Error, db *C.sqlite3) {
	// See SQLiteConn.lastError(), in file 'sqlite3.go' at the time of writing (Sept 5, 2016)
	dbErr.Code = ErrNo(C.sqlite3_errcode(db))
	dbErr.ExtendedCode = ErrNoExtended(C.sqlite3_extended_errcode(db))
	dbErr.err = C.GoString(C.sqlite3_errmsg(db))
}

func fillExpandedSQL(info *TraceInfo, db *C.sqlite3, pStmt unsafe.Pointer) {
	if pStmt == nil {
		panic("No SQLite statement pointer in P arg of trace_v2 callback")
	}

	expSQLiteCStr := C.sqlite3_expanded_sql((*C.sqlite3_stmt)(pStmt))
	if expSQLiteCStr == nil {
		fillDBError(&info.DBError, db)
		return
	}
	info.ExpandedSQL = C.GoString(expSQLiteCStr)
}

//export traceCallbackTrampoline
func traceCallbackTrampoline(
	traceEventCode C.uint,
	// Parameter named 'C' in SQLite docs = Context given at registration:
	ctx unsafe.Pointer,
	// Parameter named 'P' in SQLite docs (Primary event data?):
	p unsafe.Pointer,
	// Parameter named 'X' in SQLite docs (eXtra event data?):
	xValue unsafe.Pointer) C.int {

	if ctx == nil {
		panic(fmt.Sprintf("No context (ev 0x%x)", traceEventCode))
	}

	contextDB := (*C.sqlite3)(ctx)
	connHandle := uintptr(ctx)

	var traceConf TraceConfig
	var found bool
	if traceEventCode == TraceClose {
		// clean up traceMap: 'pop' means get and delete
		traceConf, found = popTraceMapping(connHandle)
	} else {
		traceConf, found = lookupTraceMapping(connHandle)
	}

	if !found {
		panic(fmt.Sprintf("Mapping not found for handle 0x%x (ev 0x%x)",
			connHandle, traceEventCode))
	}

	var info TraceInfo

	info.EventCode = uint32(traceEventCode)
	info.AutoCommit = (int(C.sqlite3_get_autocommit(contextDB)) != 0)
	info.ConnHandle = connHandle

	switch traceEventCode {
	case TraceStmt:
		info.StmtHandle = uintptr(p)

		var xStr string
		if xValue != nil {
			xStr = C.GoString((*C.char)(xValue))
		}
		info.StmtOrTrigger = xStr
		if !strings.HasPrefix(xStr, "--") {
			// Not SQL comment, therefore the current event
			// is not related to a trigger.
			// The user might want to receive the expanded SQL;
			// let's check:
			if traceConf.WantExpandedSQL {
				fillExpandedSQL(&info, contextDB, p)
			}
		}

	case TraceProfile:
		info.StmtHandle = uintptr(p)

		if xValue == nil {
			panic("NULL pointer in X arg of trace_v2 callback for SQLITE_TRACE_PROFILE event")
		}

		info.RunTimeNanosec = *(*int64)(xValue)

		// sample the error //TODO: is it safe? is it useful?
		fillDBError(&info.DBError, contextDB)

	case TraceRow:
		info.StmtHandle = uintptr(p)

	case TraceClose:
		handle := uintptr(p)
		if handle != info.ConnHandle {
			panic(fmt.Sprintf("Different conn handle 0x%x (expected 0x%x) in SQLITE_TRACE_CLOSE event.",
				handle, info.ConnHandle))
		}

	default:
		// Pass unsupported events to the user callback (if configured);
		// let the user callback decide whether to panic or ignore them.
	}

	// Do not execute user callback when the event was not requested by user!
	// Remember that the Close event is always selected when
	// registering this callback trampoline with SQLite --- for cleanup.
	// In the future there may be more events forced to "selected" in SQLite
	// for the driver's needs.
	if traceConf.EventMask&traceEventCode == 0 {
		return 0
	}

	r := 0
	if traceConf.Callback != nil {
		r = traceConf.Callback(info)
	}
	return C.int(r)
}

type traceMapEntry struct {
	config TraceConfig
}

var traceMapLock sync.Mutex
var traceMap = make(map[uintptr]traceMapEntry)

func addTraceMapping(connHandle uintptr, traceConf TraceConfig) {
	traceMapLock.Lock()
	defer traceMapLock.Unlock()

	oldEntryCopy, found := traceMap[connHandle]
	if found {
		panic(fmt.Sprintf("Adding trace config %v: handle 0x%x already registered (%v).",
			traceConf, connHandle, oldEntryCopy.config))
	}
	traceMap[connHandle] = traceMapEntry{config: traceConf}
	fmt.Printf("Added trace config %v: handle 0x%x.\n", traceConf, connHandle)
}

func lookupTraceMapping(connHandle uintptr) (TraceConfig, bool) {
	traceMapLock.Lock()
	defer traceMapLock.Unlock()

	entryCopy, found := traceMap[connHandle]
	return entryCopy.config, found
}

// 'pop' = get and delete from map before returning the value to the caller
func popTraceMapping(connHandle uintptr) (TraceConfig, bool) {
	traceMapLock.Lock()
	defer traceMapLock.Unlock()

	entryCopy, found := traceMap[connHandle]
	if found {
		delete(traceMap, connHandle)
		fmt.Printf("Pop handle 0x%x: deleted trace config %v.\n", connHandle, entryCopy.config)
	}
	return entryCopy.config, found
}

// SetTrace installs or removes the trace callback for the given database connection.
// It's not named 'RegisterTrace' because only one callback can be kept and called.
// Calling SetTrace a second time on same database connection
// overrides (cancels) any prior callback and all its settings:
// event mask, etc.
func (c *SQLiteConn) SetTrace(requested *TraceConfig) error {
	connHandle := uintptr(unsafe.Pointer(c.db))

	_, _ = popTraceMapping(connHandle)

	if requested == nil {
		// The traceMap entry was deleted already by popTraceMapping():
		// can disable all events now, no need to watch for TraceClose.
		err := c.setSQLiteTrace(0)
		return err
	}

	reqCopy := *requested

	// Disable potentially expensive operations
	// if their result will not be used. We are doing this
	// just in case the caller provided nonsensical input.
	if reqCopy.EventMask&TraceStmt == 0 {
		reqCopy.WantExpandedSQL = false
	}

	addTraceMapping(connHandle, reqCopy)

	// The callback trampoline function does cleanup on Close event,
	// regardless of the presence or absence of the user callback.
	// Therefore it needs the Close event to be selected:
	actualEventMask := uint(reqCopy.EventMask | TraceClose)
	err := c.setSQLiteTrace(actualEventMask)
	return err
}

func (c *SQLiteConn) setSQLiteTrace(sqliteEventMask uint) error {
	rv := C.sqlite3_trace_v2(c.db,
		C.uint(sqliteEventMask),
		(*[0]byte)(unsafe.Pointer(C.traceCallbackTrampoline)),
		unsafe.Pointer(c.db)) // Fourth arg is same as first: we are
	// passing the database connection handle as callback context.

	if rv != C.SQLITE_OK {
		return c.lastError()
	}
	return nil
}

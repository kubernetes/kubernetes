// Copyright 2013 The win_pdh Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build windows

package win_pdh

import (
	"syscall"
	"unsafe"
)

// Error codes
const (
	ERROR_SUCCESS             = 0
	ERROR_INVALID_FUNCTION    = 1
)

type (
	HANDLE    uintptr
)

// PDH error codes, which can be returned by all Pdh* functions. Taken from mingw-w64 pdhmsg.h
const (
	PDH_CSTATUS_VALID_DATA                     = 0x00000000 // The returned data is valid.
	PDH_CSTATUS_NEW_DATA                       = 0x00000001 // The return data value is valid and different from the last sample.
	PDH_CSTATUS_NO_MACHINE                     = 0x800007D0 // Unable to connect to the specified computer, or the computer is offline.
	PDH_CSTATUS_NO_INSTANCE                    = 0x800007D1
	PDH_MORE_DATA                              = 0x800007D2 // The PdhGetFormattedCounterArray* function can return this if there's 'more data to be displayed'.
	PDH_CSTATUS_ITEM_NOT_VALIDATED             = 0x800007D3
	PDH_RETRY                                  = 0x800007D4
	PDH_NO_DATA                                = 0x800007D5 // The query does not currently contain any counters (for example, limited access)
	PDH_CALC_NEGATIVE_DENOMINATOR              = 0x800007D6
	PDH_CALC_NEGATIVE_TIMEBASE                 = 0x800007D7
	PDH_CALC_NEGATIVE_VALUE                    = 0x800007D8
	PDH_DIALOG_CANCELLED                       = 0x800007D9
	PDH_END_OF_LOG_FILE                        = 0x800007DA
	PDH_ASYNC_QUERY_TIMEOUT                    = 0x800007DB
	PDH_CANNOT_SET_DEFAULT_REALTIME_DATASOURCE = 0x800007DC
	PDH_CSTATUS_NO_OBJECT                      = 0xC0000BB8
	PDH_CSTATUS_NO_COUNTER                     = 0xC0000BB9 // The specified counter could not be found.
	PDH_CSTATUS_INVALID_DATA                   = 0xC0000BBA // The counter was successfully found, but the data returned is not valid.
	PDH_MEMORY_ALLOCATION_FAILURE              = 0xC0000BBB
	PDH_INVALID_HANDLE                         = 0xC0000BBC
	PDH_INVALID_ARGUMENT                       = 0xC0000BBD // Required argument is missing or incorrect.
	PDH_FUNCTION_NOT_FOUND                     = 0xC0000BBE
	PDH_CSTATUS_NO_COUNTERNAME                 = 0xC0000BBF
	PDH_CSTATUS_BAD_COUNTERNAME                = 0xC0000BC0 // Unable to parse the counter path. Check the format and syntax of the specified path.
	PDH_INVALID_BUFFER                         = 0xC0000BC1
	PDH_INSUFFICIENT_BUFFER                    = 0xC0000BC2
	PDH_CANNOT_CONNECT_MACHINE                 = 0xC0000BC3
	PDH_INVALID_PATH                           = 0xC0000BC4
	PDH_INVALID_INSTANCE                       = 0xC0000BC5
	PDH_INVALID_DATA                           = 0xC0000BC6 // specified counter does not contain valid data or a successful status code.
	PDH_NO_DIALOG_DATA                         = 0xC0000BC7
	PDH_CANNOT_READ_NAME_STRINGS               = 0xC0000BC8
	PDH_LOG_FILE_CREATE_ERROR                  = 0xC0000BC9
	PDH_LOG_FILE_OPEN_ERROR                    = 0xC0000BCA
	PDH_LOG_TYPE_NOT_FOUND                     = 0xC0000BCB
	PDH_NO_MORE_DATA                           = 0xC0000BCC
	PDH_ENTRY_NOT_IN_LOG_FILE                  = 0xC0000BCD
	PDH_DATA_SOURCE_IS_LOG_FILE                = 0xC0000BCE
	PDH_DATA_SOURCE_IS_REAL_TIME               = 0xC0000BCF
	PDH_UNABLE_READ_LOG_HEADER                 = 0xC0000BD0
	PDH_FILE_NOT_FOUND                         = 0xC0000BD1
	PDH_FILE_ALREADY_EXISTS                    = 0xC0000BD2
	PDH_NOT_IMPLEMENTED                        = 0xC0000BD3
	PDH_STRING_NOT_FOUND                       = 0xC0000BD4
	PDH_UNABLE_MAP_NAME_FILES                  = 0x80000BD5
	PDH_UNKNOWN_LOG_FORMAT                     = 0xC0000BD6
	PDH_UNKNOWN_LOGSVC_COMMAND                 = 0xC0000BD7
	PDH_LOGSVC_QUERY_NOT_FOUND                 = 0xC0000BD8
	PDH_LOGSVC_NOT_OPENED                      = 0xC0000BD9
	PDH_WBEM_ERROR                             = 0xC0000BDA
	PDH_ACCESS_DENIED                          = 0xC0000BDB
	PDH_LOG_FILE_TOO_SMALL                     = 0xC0000BDC
	PDH_INVALID_DATASOURCE                     = 0xC0000BDD
	PDH_INVALID_SQLDB                          = 0xC0000BDE
	PDH_NO_COUNTERS                            = 0xC0000BDF
	PDH_SQL_ALLOC_FAILED                       = 0xC0000BE0
	PDH_SQL_ALLOCCON_FAILED                    = 0xC0000BE1
	PDH_SQL_EXEC_DIRECT_FAILED                 = 0xC0000BE2
	PDH_SQL_FETCH_FAILED                       = 0xC0000BE3
	PDH_SQL_ROWCOUNT_FAILED                    = 0xC0000BE4
	PDH_SQL_MORE_RESULTS_FAILED                = 0xC0000BE5
	PDH_SQL_CONNECT_FAILED                     = 0xC0000BE6
	PDH_SQL_BIND_FAILED                        = 0xC0000BE7
	PDH_CANNOT_CONNECT_WMI_SERVER              = 0xC0000BE8
	PDH_PLA_COLLECTION_ALREADY_RUNNING         = 0xC0000BE9
	PDH_PLA_ERROR_SCHEDULE_OVERLAP             = 0xC0000BEA
	PDH_PLA_COLLECTION_NOT_FOUND               = 0xC0000BEB
	PDH_PLA_ERROR_SCHEDULE_ELAPSED             = 0xC0000BEC
	PDH_PLA_ERROR_NOSTART                      = 0xC0000BED
	PDH_PLA_ERROR_ALREADY_EXISTS               = 0xC0000BEE
	PDH_PLA_ERROR_TYPE_MISMATCH                = 0xC0000BEF
	PDH_PLA_ERROR_FILEPATH                     = 0xC0000BF0
	PDH_PLA_SERVICE_ERROR                      = 0xC0000BF1
	PDH_PLA_VALIDATION_ERROR                   = 0xC0000BF2
	PDH_PLA_VALIDATION_WARNING                 = 0x80000BF3
	PDH_PLA_ERROR_NAME_TOO_LONG                = 0xC0000BF4
	PDH_INVALID_SQL_LOG_FORMAT                 = 0xC0000BF5
	PDH_COUNTER_ALREADY_IN_QUERY               = 0xC0000BF6
	PDH_BINARY_LOG_CORRUPT                     = 0xC0000BF7
	PDH_LOG_SAMPLE_TOO_SMALL                   = 0xC0000BF8
	PDH_OS_LATER_VERSION                       = 0xC0000BF9
	PDH_OS_EARLIER_VERSION                     = 0xC0000BFA
	PDH_INCORRECT_APPEND_TIME                  = 0xC0000BFB
	PDH_UNMATCHED_APPEND_COUNTER               = 0xC0000BFC
	PDH_SQL_ALTER_DETAIL_FAILED                = 0xC0000BFD
	PDH_QUERY_PERF_DATA_TIMEOUT                = 0xC0000BFE
)

// Formatting options for GetFormattedCounterValue().
const (
	PDH_FMT_RAW          = 0x00000010
	PDH_FMT_ANSI         = 0x00000020
	PDH_FMT_UNICODE      = 0x00000040
	PDH_FMT_LONG         = 0x00000100 // Return data as a long int.
	PDH_FMT_DOUBLE       = 0x00000200 // Return data as a double precision floating point real.
	PDH_FMT_LARGE        = 0x00000400 // Return data as a 64 bit integer.
	PDH_FMT_NOSCALE      = 0x00001000 // can be OR-ed: Do not apply the counter's default scaling factor.
	PDH_FMT_1000         = 0x00002000 // can be OR-ed: multiply the actual value by 1,000.
	PDH_FMT_NODATA       = 0x00004000 // can be OR-ed: unknown what this is for, MSDN says nothing.
	PDH_FMT_NOCAP100     = 0x00008000 // can be OR-ed: do not cap values > 100.
	PERF_DETAIL_COSTLY   = 0x00010000
	PERF_DETAIL_STANDARD = 0x0000FFFF
)

type (
	PDH_HQUERY   HANDLE // query handle
	PDH_HCOUNTER HANDLE // counter handle
)

// Union specialization for double values
type PDH_FMT_COUNTERVALUE_DOUBLE struct {
	CStatus     uint32
	DoubleValue float64
}

// Union specialization for 64 bit integer values
type PDH_FMT_COUNTERVALUE_LARGE struct {
	CStatus    uint32
	LargeValue int64
}

// Union specialization for long values
type PDH_FMT_COUNTERVALUE_LONG struct {
	CStatus   uint32
	LongValue int32
	padding   [4]byte
}

// Union specialization for double values, used by PdhGetFormattedCounterArrayDouble()
type PDH_FMT_COUNTERVALUE_ITEM_DOUBLE struct {
	SzName   *uint16 // pointer to a string
	FmtValue PDH_FMT_COUNTERVALUE_DOUBLE
}

// Union specialization for 'large' values, used by PdhGetFormattedCounterArrayLarge()
type PDH_FMT_COUNTERVALUE_ITEM_LARGE struct {
	SzName   *uint16 // pointer to a string
	FmtValue PDH_FMT_COUNTERVALUE_LARGE
}

// Union specialization for long values, used by PdhGetFormattedCounterArrayLong()
type PDH_FMT_COUNTERVALUE_ITEM_LONG struct {
	SzName   *uint16 // pointer to a string
	FmtValue PDH_FMT_COUNTERVALUE_LONG
}

var (
	// Library
	libpdhDll *syscall.DLL

	// Functions
	pdh_AddCounterW               *syscall.Proc
	pdh_AddEnglishCounterW        *syscall.Proc
	pdh_CloseQuery                *syscall.Proc
	pdh_CollectQueryData          *syscall.Proc
	pdh_GetFormattedCounterValue  *syscall.Proc
	pdh_GetFormattedCounterArrayW *syscall.Proc
	pdh_OpenQuery                 *syscall.Proc
	pdh_ValidatePathW             *syscall.Proc
)

func init() {
	// Library
	libpdhDll = syscall.MustLoadDLL("pdh.dll")

	// Functions
	pdh_AddCounterW = libpdhDll.MustFindProc("PdhAddCounterW")
	pdh_AddEnglishCounterW, _ = libpdhDll.FindProc("PdhAddEnglishCounterW") // XXX: only supported on versions > Vista.
	pdh_CloseQuery = libpdhDll.MustFindProc("PdhCloseQuery")
	pdh_CollectQueryData = libpdhDll.MustFindProc("PdhCollectQueryData")
	pdh_GetFormattedCounterValue = libpdhDll.MustFindProc("PdhGetFormattedCounterValue")
	pdh_GetFormattedCounterArrayW = libpdhDll.MustFindProc("PdhGetFormattedCounterArrayW")
	pdh_OpenQuery = libpdhDll.MustFindProc("PdhOpenQuery")
	pdh_ValidatePathW = libpdhDll.MustFindProc("PdhValidatePathW")
}

// Adds the specified counter to the query. This is the internationalized version. Preferably, use the
// function PdhAddEnglishCounter instead. hQuery is the query handle, which has been fetched by PdhOpenQuery.
// szFullCounterPath is a full, internationalized counter path (this will differ per Windows language version).
// dwUserData is a 'user-defined value', which becomes part of the counter information. To retrieve this value
// later, call PdhGetCounterInfo() and access dwQueryUserData of the PDH_COUNTER_INFO structure.
//
// Examples of szFullCounterPath (in an English version of Windows):
//
//	\\Processor(_Total)\\% Idle Time
//	\\Processor(_Total)\\% Processor Time
//	\\LogicalDisk(C:)\% Free Space
//
// To view all (internationalized...) counters on a system, there are three non-programmatic ways: perfmon utility,
// the typeperf command, and the the registry editor. perfmon.exe is perhaps the easiest way, because it's basically a
// full implemention of the pdh.dll API, except with a GUI and all that. The registry setting also provides an
// interface to the available counters, and can be found at the following key:
//
// 	HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion\Perflib\CurrentLanguage
//
// This registry key contains several values as follows:
//
//	1
//	1847
//	2
//	System
//	4
//	Memory
//	6
//	% Processor Time
//	... many, many more
//
// Somehow, these numeric values can be used as szFullCounterPath too:
//
//	\2\6 will correspond to \\System\% Processor Time
//
// The typeperf command may also be pretty easy. To find all performance counters, simply execute:
//
//	typeperf -qx
func PdhAddCounter(hQuery PDH_HQUERY, szFullCounterPath string, dwUserData uintptr, phCounter *PDH_HCOUNTER) uint32 {
	ptxt, _ := syscall.UTF16PtrFromString(szFullCounterPath)
	ret, _, _ := pdh_AddCounterW.Call(
		uintptr(hQuery),
		uintptr(unsafe.Pointer(ptxt)),
		dwUserData,
		uintptr(unsafe.Pointer(phCounter)))

	return uint32(ret)
}

// Adds the specified language-neutral counter to the query. See the PdhAddCounter function. This function only exists on
// Windows versions higher than Vista.
func PdhAddEnglishCounter(hQuery PDH_HQUERY, szFullCounterPath string, dwUserData uintptr, phCounter *PDH_HCOUNTER) uint32 {
	if pdh_AddEnglishCounterW == nil {
		return ERROR_INVALID_FUNCTION
	}

	ptxt, _ := syscall.UTF16PtrFromString(szFullCounterPath)
	ret, _, _ := pdh_AddEnglishCounterW.Call(
		uintptr(hQuery),
		uintptr(unsafe.Pointer(ptxt)),
		dwUserData,
		uintptr(unsafe.Pointer(phCounter)))

	return uint32(ret)
}

// Closes all counters contained in the specified query, closes all handles related to the query,
// and frees all memory associated with the query.
func PdhCloseQuery(hQuery PDH_HQUERY) uint32 {
	ret, _, _ := pdh_CloseQuery.Call(uintptr(hQuery))

	return uint32(ret)
}

// Collects the current raw data value for all counters in the specified query and updates the status
// code of each counter. With some counters, this function needs to be repeatedly called before the value
// of the counter can be extracted with PdhGetFormattedCounterValue(). For example, the following code
// requires at least two calls:
//
// 	var handle win.PDH_HQUERY
// 	var counterHandle win.PDH_HCOUNTER
// 	ret := win.PdhOpenQuery(0, 0, &handle)
//	ret = win.PdhAddEnglishCounter(handle, "\\Processor(_Total)\\% Idle Time", 0, &counterHandle)
//	var derp win.PDH_FMT_COUNTERVALUE_DOUBLE
//
//	ret = win.PdhCollectQueryData(handle)
//	fmt.Printf("Collect return code is %x\n", ret) // return code will be PDH_CSTATUS_INVALID_DATA
//	ret = win.PdhGetFormattedCounterValueDouble(counterHandle, 0, &derp)
//
//	ret = win.PdhCollectQueryData(handle)
//	fmt.Printf("Collect return code is %x\n", ret) // return code will be ERROR_SUCCESS
//	ret = win.PdhGetFormattedCounterValueDouble(counterHandle, 0, &derp)
//
// The PdhCollectQueryData will return an error in the first call because it needs two values for
// displaying the correct data for the processor idle time. The second call will have a 0 return code.
func PdhCollectQueryData(hQuery PDH_HQUERY) uint32 {
	ret, _, _ := pdh_CollectQueryData.Call(uintptr(hQuery))

	return uint32(ret)
}

// Formats the given hCounter using a 'double'. The result is set into the specialized union struct pValue.
// This function does not directly translate to a Windows counterpart due to union specialization tricks.
func PdhGetFormattedCounterValueDouble(hCounter PDH_HCOUNTER, lpdwType *uint32, pValue *PDH_FMT_COUNTERVALUE_DOUBLE) uint32 {
	ret, _, _ := pdh_GetFormattedCounterValue.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_DOUBLE),
		uintptr(unsafe.Pointer(lpdwType)),
		uintptr(unsafe.Pointer(pValue)))

	return uint32(ret)
}

// Formats the given hCounter using a large int (int64). The result is set into the specialized union struct pValue.
// This function does not directly translate to a Windows counterpart due to union specialization tricks.
func PdhGetFormattedCounterValueLarge(hCounter PDH_HCOUNTER, lpdwType *uint32, pValue *PDH_FMT_COUNTERVALUE_LARGE) uint32 {
	ret, _, _ := pdh_GetFormattedCounterValue.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_LARGE),
		uintptr(unsafe.Pointer(lpdwType)),
		uintptr(unsafe.Pointer(pValue)))

	return uint32(ret)
}

// Formats the given hCounter using a 'long'. The result is set into the specialized union struct pValue.
// This function does not directly translate to a Windows counterpart due to union specialization tricks.
//
// BUG(krpors): Testing this function on multiple systems yielded inconsistent results. For instance,
// the pValue.LongValue kept the value '192' on test system A, but on B this was '0', while the padding
// bytes of the struct got the correct value. Until someone can figure out this behaviour, prefer to use
// the Double or Large counterparts instead. These functions provide actually the same data, except in
// a different, working format.
func PdhGetFormattedCounterValueLong(hCounter PDH_HCOUNTER, lpdwType *uint32, pValue *PDH_FMT_COUNTERVALUE_LONG) uint32 {
	ret, _, _ := pdh_GetFormattedCounterValue.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_LONG),
		uintptr(unsafe.Pointer(lpdwType)),
		uintptr(unsafe.Pointer(pValue)))

	return uint32(ret)
}

// Returns an array of formatted counter values. Use this function when you want to format the counter values of a
// counter that contains a wildcard character for the instance name. The itemBuffer must a slice of type PDH_FMT_COUNTERVALUE_ITEM_DOUBLE.
// An example of how this function can be used:
//
//	okPath := "\\Process(*)\\% Processor Time" // notice the wildcard * character
//
//	// ommitted all necessary stuff ...
//
//	var bufSize uint32
//	var bufCount uint32
//	var size uint32 = uint32(unsafe.Sizeof(win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE{}))
//	var emptyBuf [1]win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE // need at least 1 addressable null ptr.
//
//	for {
//		// collect
//		ret := win.PdhCollectQueryData(queryHandle)
//		if ret == win.ERROR_SUCCESS {
//			ret = win.PdhGetFormattedCounterArrayDouble(counterHandle, &bufSize, &bufCount, &emptyBuf[0]) // uses null ptr here according to MSDN.
//			if ret == win.PDH_MORE_DATA {
//				filledBuf := make([]win.PDH_FMT_COUNTERVALUE_ITEM_DOUBLE, bufCount*size)
//				ret = win.PdhGetFormattedCounterArrayDouble(counterHandle, &bufSize, &bufCount, &filledBuf[0])
//				for i := 0; i < int(bufCount); i++ {
//					c := filledBuf[i]
//					var s string = win.UTF16PtrToString(c.SzName)
//					fmt.Printf("Index %d -> %s, value %v\n", i, s, c.FmtValue.DoubleValue)
//				}
//
//				filledBuf = nil
//				// Need to at least set bufSize to zero, because if not, the function will not
//				// return PDH_MORE_DATA and will not set the bufSize.
//				bufCount = 0
//				bufSize = 0
//			}
//
//			time.Sleep(2000 * time.Millisecond)
//		}
//	}
func PdhGetFormattedCounterArrayDouble(hCounter PDH_HCOUNTER, lpdwBufferSize *uint32, lpdwBufferCount *uint32, itemBuffer *PDH_FMT_COUNTERVALUE_ITEM_DOUBLE) uint32 {
	ret, _, _ := pdh_GetFormattedCounterArrayW.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_DOUBLE),
		uintptr(unsafe.Pointer(lpdwBufferSize)),
		uintptr(unsafe.Pointer(lpdwBufferCount)),
		uintptr(unsafe.Pointer(itemBuffer)))

	return uint32(ret)
}

// Returns an array of formatted counter values. Use this function when you want to format the counter values of a
// counter that contains a wildcard character for the instance name. The itemBuffer must a slice of type PDH_FMT_COUNTERVALUE_ITEM_LARGE.
// For an example usage, see PdhGetFormattedCounterArrayDouble.
func PdhGetFormattedCounterArrayLarge(hCounter PDH_HCOUNTER, lpdwBufferSize *uint32, lpdwBufferCount *uint32, itemBuffer *PDH_FMT_COUNTERVALUE_ITEM_LARGE) uint32 {
	ret, _, _ := pdh_GetFormattedCounterArrayW.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_LARGE),
		uintptr(unsafe.Pointer(lpdwBufferSize)),
		uintptr(unsafe.Pointer(lpdwBufferCount)),
		uintptr(unsafe.Pointer(itemBuffer)))

	return uint32(ret)
}

// Returns an array of formatted counter values. Use this function when you want to format the counter values of a
// counter that contains a wildcard character for the instance name. The itemBuffer must a slice of type PDH_FMT_COUNTERVALUE_ITEM_LONG.
// For an example usage, see PdhGetFormattedCounterArrayDouble.
//
// BUG(krpors): See description of PdhGetFormattedCounterValueLong().
func PdhGetFormattedCounterArrayLong(hCounter PDH_HCOUNTER, lpdwBufferSize *uint32, lpdwBufferCount *uint32, itemBuffer *PDH_FMT_COUNTERVALUE_ITEM_LONG) uint32 {
	ret, _, _ := pdh_GetFormattedCounterArrayW.Call(
		uintptr(hCounter),
		uintptr(PDH_FMT_LONG),
		uintptr(unsafe.Pointer(lpdwBufferSize)),
		uintptr(unsafe.Pointer(lpdwBufferCount)),
		uintptr(unsafe.Pointer(itemBuffer)))

	return uint32(ret)
}

// Creates a new query that is used to manage the collection of performance data.
// szDataSource is a null terminated string that specifies the name of the log file from which to
// retrieve the performance data. If 0, performance data is collected from a real-time data source.
// dwUserData is a user-defined value to associate with this query. To retrieve the user data later,
// call PdhGetCounterInfo and access dwQueryUserData of the PDH_COUNTER_INFO structure. phQuery is
// the handle to the query, and must be used in subsequent calls. This function returns a PDH_
// constant error code, or ERROR_SUCCESS if the call succeeded.
func PdhOpenQuery(szDataSource uintptr, dwUserData uintptr, phQuery *PDH_HQUERY) uint32 {
	ret, _, _ := pdh_OpenQuery.Call(
		szDataSource,
		dwUserData,
		uintptr(unsafe.Pointer(phQuery)))

	return uint32(ret)
}

// Validates a path. Will return ERROR_SUCCESS when ok, or PDH_CSTATUS_BAD_COUNTERNAME when the path is
// erroneous.
func PdhValidatePath(path string) uint32 {
	ptxt, _ := syscall.UTF16PtrFromString(path)
	ret, _, _ := pdh_ValidatePathW.Call(uintptr(unsafe.Pointer(ptxt)))

	return uint32(ret)
}

func UTF16PtrToString(s *uint16) string {
	if s == nil {
		return ""
	}
	return syscall.UTF16ToString((*[1 << 29]uint16)(unsafe.Pointer(s))[0:])
}

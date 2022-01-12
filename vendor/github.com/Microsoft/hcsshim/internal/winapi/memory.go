package winapi

// VOID RtlMoveMemory(
// 	_Out_       VOID UNALIGNED *Destination,
// 	_In_  const VOID UNALIGNED *Source,
// 	_In_        SIZE_T         Length
// );
//sys RtlMoveMemory(destination *byte, source *byte, length uintptr) (err error) = kernel32.RtlMoveMemory

//sys LocalAlloc(flags uint32, size int) (ptr uintptr) = kernel32.LocalAlloc
//sys LocalFree(ptr uintptr) = kernel32.LocalFree

// BOOL QueryWorkingSet(
//	HANDLE hProcess,
//	PVOID  pv,
//	DWORD  cb
// );
//sys QueryWorkingSet(handle windows.Handle, pv uintptr, cb uint32) (err error) = psapi.QueryWorkingSet

type PSAPI_WORKING_SET_INFORMATION struct {
	NumberOfEntries uintptr
	WorkingSetInfo  [1]PSAPI_WORKING_SET_BLOCK
}

type PSAPI_WORKING_SET_BLOCK struct {
	Flags uintptr
}

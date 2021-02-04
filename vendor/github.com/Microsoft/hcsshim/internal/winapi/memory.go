package winapi

// VOID RtlMoveMemory(
// 	_Out_       VOID UNALIGNED *Destination,
// 	_In_  const VOID UNALIGNED *Source,
// 	_In_        SIZE_T         Length
// );
//sys RtlMoveMemory(destination *byte, source *byte, length uintptr) (err error) = kernel32.RtlMoveMemory

//sys LocalAlloc(flags uint32, size int) (ptr uintptr) = kernel32.LocalAlloc
//sys LocalFree(ptr uintptr) = kernel32.LocalFree

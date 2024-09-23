package winapi

const (
	BINDFLT_FLAG_READ_ONLY_MAPPING        uint32 = 0x00000001
	BINDFLT_FLAG_MERGED_BIND_MAPPING      uint32 = 0x00000002
	BINDFLT_FLAG_USE_CURRENT_SILO_MAPPING uint32 = 0x00000004
)

// HRESULT
// BfSetupFilter(
//     _In_opt_ HANDLE JobHandle,
//     _In_ ULONG Flags,
//     _In_ LPCWSTR VirtualizationRootPath,
//     _In_ LPCWSTR VirtualizationTargetPath,
//     _In_reads_opt_( VirtualizationExceptionPathCount ) LPCWSTR* VirtualizationExceptionPaths,
//     _In_opt_ ULONG VirtualizationExceptionPathCount
// );
//
//sys BfSetupFilter(jobHandle windows.Handle, flags uint32, virtRootPath *uint16, virtTargetPath *uint16, virtExceptions **uint16, virtExceptionPathCount uint32) (hr error) = bindfltapi.BfSetupFilter?

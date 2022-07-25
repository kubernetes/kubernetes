package winapi

import "golang.org/x/sys/windows"

const SystemProcessInformation = 5

const STATUS_INFO_LENGTH_MISMATCH = 0xC0000004

// __kernel_entry NTSTATUS NtQuerySystemInformation(
// 	SYSTEM_INFORMATION_CLASS SystemInformationClass,
// 	PVOID                    SystemInformation,
// 	ULONG                    SystemInformationLength,
// 	PULONG                   ReturnLength
// );
//sys NtQuerySystemInformation(systemInfoClass int, systemInformation uintptr, systemInfoLength uint32, returnLength *uint32) (status uint32) = ntdll.NtQuerySystemInformation

type SYSTEM_PROCESS_INFORMATION struct {
	NextEntryOffset              uint32         // ULONG
	NumberOfThreads              uint32         // ULONG
	WorkingSetPrivateSize        int64          // LARGE_INTEGER
	HardFaultCount               uint32         // ULONG
	NumberOfThreadsHighWatermark uint32         // ULONG
	CycleTime                    uint64         // ULONGLONG
	CreateTime                   int64          // LARGE_INTEGER
	UserTime                     int64          // LARGE_INTEGER
	KernelTime                   int64          // LARGE_INTEGER
	ImageName                    UnicodeString  // UNICODE_STRING
	BasePriority                 int32          // KPRIORITY
	UniqueProcessID              windows.Handle // HANDLE
	InheritedFromUniqueProcessID windows.Handle // HANDLE
	HandleCount                  uint32         // ULONG
	SessionID                    uint32         // ULONG
	UniqueProcessKey             *uint32        // ULONG_PTR
	PeakVirtualSize              uintptr        // SIZE_T
	VirtualSize                  uintptr        // SIZE_T
	PageFaultCount               uint32         // ULONG
	PeakWorkingSetSize           uintptr        // SIZE_T
	WorkingSetSize               uintptr        // SIZE_T
	QuotaPeakPagedPoolUsage      uintptr        // SIZE_T
	QuotaPagedPoolUsage          uintptr        // SIZE_T
	QuotaPeakNonPagedPoolUsage   uintptr        // SIZE_T
	QuotaNonPagedPoolUsage       uintptr        // SIZE_T
	PagefileUsage                uintptr        // SIZE_T
	PeakPagefileUsage            uintptr        // SIZE_T
	PrivatePageCount             uintptr        // SIZE_T
	ReadOperationCount           int64          // LARGE_INTEGER
	WriteOperationCount          int64          // LARGE_INTEGER
	OtherOperationCount          int64          // LARGE_INTEGER
	ReadTransferCount            int64          // LARGE_INTEGER
	WriteTransferCount           int64          // LARGE_INTEGER
	OtherTransferCount           int64          // LARGE_INTEGER
}

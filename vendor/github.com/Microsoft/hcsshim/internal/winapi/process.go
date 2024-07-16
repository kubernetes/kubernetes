package winapi

const PROCESS_ALL_ACCESS uint32 = 2097151

const (
	PROC_THREAD_ATTRIBUTE_PSEUDOCONSOLE = 0x20016
	PROC_THREAD_ATTRIBUTE_JOB_LIST      = 0x2000D
)

// ProcessVmCounters corresponds to the _VM_COUNTERS_EX and _VM_COUNTERS_EX2 structures.
const ProcessVmCounters = 3

// __kernel_entry NTSTATUS NtQueryInformationProcess(
// 	[in]            HANDLE           ProcessHandle,
// 	[in]            PROCESSINFOCLASS ProcessInformationClass,
// 	[out]           PVOID            ProcessInformation,
// 	[in]            ULONG            ProcessInformationLength,
// 	[out, optional] PULONG           ReturnLength
// );
//
//sys NtQueryInformationProcess(processHandle windows.Handle, processInfoClass uint32, processInfo unsafe.Pointer, processInfoLength uint32, returnLength *uint32) (status uint32) = ntdll.NtQueryInformationProcess

//	typedef struct _VM_COUNTERS_EX {
//		   SIZE_T PeakVirtualSize;
//		   SIZE_T VirtualSize;
//		   ULONG PageFaultCount;
//		   SIZE_T PeakWorkingSetSize;
//		   SIZE_T WorkingSetSize;
//		   SIZE_T QuotaPeakPagedPoolUsage;
//		   SIZE_T QuotaPagedPoolUsage;
//		   SIZE_T QuotaPeakNonPagedPoolUsage;
//		   SIZE_T QuotaNonPagedPoolUsage;
//		   SIZE_T PagefileUsage;
//		   SIZE_T PeakPagefileUsage;
//		   SIZE_T PrivateUsage;
//	} VM_COUNTERS_EX, *PVM_COUNTERS_EX;
type VM_COUNTERS_EX struct {
	PeakVirtualSize            uintptr
	VirtualSize                uintptr
	PageFaultCount             uint32
	PeakWorkingSetSize         uintptr
	WorkingSetSize             uintptr
	QuotaPeakPagedPoolUsage    uintptr
	QuotaPagedPoolUsage        uintptr
	QuotaPeakNonPagedPoolUsage uintptr
	QuotaNonPagedPoolUsage     uintptr
	PagefileUsage              uintptr
	PeakPagefileUsage          uintptr
	PrivateUsage               uintptr
}

//	typedef struct _VM_COUNTERS_EX2 {
//		   VM_COUNTERS_EX CountersEx;
//		   SIZE_T PrivateWorkingSetSize;
//		   SIZE_T SharedCommitUsage;
//	} VM_COUNTERS_EX2, *PVM_COUNTERS_EX2;
type VM_COUNTERS_EX2 struct {
	CountersEx            VM_COUNTERS_EX
	PrivateWorkingSetSize uintptr
	SharedCommitUsage     uintptr
}

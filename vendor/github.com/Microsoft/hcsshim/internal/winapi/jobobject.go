package winapi

import (
	"unsafe"

	"golang.org/x/sys/windows"
)

// Messages that can be received from an assigned io completion port.
// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port
const (
	JOB_OBJECT_MSG_END_OF_JOB_TIME       uint32 = 1
	JOB_OBJECT_MSG_END_OF_PROCESS_TIME   uint32 = 2
	JOB_OBJECT_MSG_ACTIVE_PROCESS_LIMIT  uint32 = 3
	JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO   uint32 = 4
	JOB_OBJECT_MSG_NEW_PROCESS           uint32 = 6
	JOB_OBJECT_MSG_EXIT_PROCESS          uint32 = 7
	JOB_OBJECT_MSG_ABNORMAL_EXIT_PROCESS uint32 = 8
	JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT  uint32 = 9
	JOB_OBJECT_MSG_JOB_MEMORY_LIMIT      uint32 = 10
	JOB_OBJECT_MSG_NOTIFICATION_LIMIT    uint32 = 11
)

// Access rights for creating or opening job objects.
//
// https://docs.microsoft.com/en-us/windows/win32/procthread/job-object-security-and-access-rights
const JOB_OBJECT_ALL_ACCESS = 0x1F001F

// IO limit flags
//
// https://docs.microsoft.com/en-us/windows/win32/api/jobapi2/ns-jobapi2-jobobject_io_rate_control_information
const JOB_OBJECT_IO_RATE_CONTROL_ENABLE = 0x1

const JOBOBJECT_IO_ATTRIBUTION_CONTROL_ENABLE uint32 = 0x1

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
const (
	JOB_OBJECT_CPU_RATE_CONTROL_ENABLE uint32 = 1 << iota
	JOB_OBJECT_CPU_RATE_CONTROL_WEIGHT_BASED
	JOB_OBJECT_CPU_RATE_CONTROL_HARD_CAP
	JOB_OBJECT_CPU_RATE_CONTROL_NOTIFY
	JOB_OBJECT_CPU_RATE_CONTROL_MIN_MAX_RATE
)

// JobObjectInformationClass values. Used for a call to QueryInformationJobObject
//
// https://docs.microsoft.com/en-us/windows/win32/api/jobapi2/nf-jobapi2-queryinformationjobobject
const (
	JobObjectBasicAccountingInformation      uint32 = 1
	JobObjectBasicProcessIdList              uint32 = 3
	JobObjectBasicAndIoAccountingInformation uint32 = 8
	JobObjectLimitViolationInformation       uint32 = 13
	JobObjectMemoryUsageInformation          uint32 = 28
	JobObjectNotificationLimitInformation2   uint32 = 33
	JobObjectIoAttribution                   uint32 = 42
)

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_limit_information
type JOBOBJECT_BASIC_LIMIT_INFORMATION struct {
	PerProcessUserTimeLimit int64
	PerJobUserTimeLimit     int64
	LimitFlags              uint32
	MinimumWorkingSetSize   uintptr
	MaximumWorkingSetSize   uintptr
	ActiveProcessLimit      uint32
	Affinity                uintptr
	PriorityClass           uint32
	SchedulingClass         uint32
}

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_cpu_rate_control_information
type JOBOBJECT_CPU_RATE_CONTROL_INFORMATION struct {
	ControlFlags uint32
	Value        uint32
}

// https://docs.microsoft.com/en-us/windows/win32/api/jobapi2/ns-jobapi2-jobobject_io_rate_control_information
type JOBOBJECT_IO_RATE_CONTROL_INFORMATION struct {
	MaxIops         int64
	MaxBandwidth    int64
	ReservationIops int64
	BaseIOSize      uint32
	VolumeName      string
	ControlFlags    uint32
}

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_process_id_list
type JOBOBJECT_BASIC_PROCESS_ID_LIST struct {
	NumberOfAssignedProcesses uint32
	NumberOfProcessIdsInList  uint32
	ProcessIdList             [1]uintptr
}

// AllPids returns all the process Ids in the job object.
func (p *JOBOBJECT_BASIC_PROCESS_ID_LIST) AllPids() []uintptr {
	return (*[(1 << 27) - 1]uintptr)(unsafe.Pointer(&p.ProcessIdList[0]))[:p.NumberOfProcessIdsInList]
}

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_accounting_information
type JOBOBJECT_BASIC_ACCOUNTING_INFORMATION struct {
	TotalUserTime             int64
	TotalKernelTime           int64
	ThisPeriodTotalUserTime   int64
	ThisPeriodTotalKernelTime int64
	TotalPageFaultCount       uint32
	TotalProcesses            uint32
	ActiveProcesses           uint32
	TotalTerminateProcesses   uint32
}

//https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_basic_and_io_accounting_information
type JOBOBJECT_BASIC_AND_IO_ACCOUNTING_INFORMATION struct {
	BasicInfo JOBOBJECT_BASIC_ACCOUNTING_INFORMATION
	IoInfo    windows.IO_COUNTERS
}

// typedef struct _JOBOBJECT_MEMORY_USAGE_INFORMATION {
//     ULONG64 JobMemory;
//     ULONG64 PeakJobMemoryUsed;
// } JOBOBJECT_MEMORY_USAGE_INFORMATION, *PJOBOBJECT_MEMORY_USAGE_INFORMATION;
//
type JOBOBJECT_MEMORY_USAGE_INFORMATION struct {
	JobMemory         uint64
	PeakJobMemoryUsed uint64
}

// typedef struct _JOBOBJECT_IO_ATTRIBUTION_STATS {
//     ULONG_PTR IoCount;
//     ULONGLONG TotalNonOverlappedQueueTime;
//     ULONGLONG TotalNonOverlappedServiceTime;
//     ULONGLONG TotalSize;
// } JOBOBJECT_IO_ATTRIBUTION_STATS, *PJOBOBJECT_IO_ATTRIBUTION_STATS;
//
type JOBOBJECT_IO_ATTRIBUTION_STATS struct {
	IoCount                       uintptr
	TotalNonOverlappedQueueTime   uint64
	TotalNonOverlappedServiceTime uint64
	TotalSize                     uint64
}

// typedef struct _JOBOBJECT_IO_ATTRIBUTION_INFORMATION {
//     ULONG ControlFlags;
//     JOBOBJECT_IO_ATTRIBUTION_STATS ReadStats;
//     JOBOBJECT_IO_ATTRIBUTION_STATS WriteStats;
// } JOBOBJECT_IO_ATTRIBUTION_INFORMATION, *PJOBOBJECT_IO_ATTRIBUTION_INFORMATION;
//
type JOBOBJECT_IO_ATTRIBUTION_INFORMATION struct {
	ControlFlags uint32
	ReadStats    JOBOBJECT_IO_ATTRIBUTION_STATS
	WriteStats   JOBOBJECT_IO_ATTRIBUTION_STATS
}

// https://docs.microsoft.com/en-us/windows/win32/api/winnt/ns-winnt-jobobject_associate_completion_port
type JOBOBJECT_ASSOCIATE_COMPLETION_PORT struct {
	CompletionKey  windows.Handle
	CompletionPort windows.Handle
}

// BOOL IsProcessInJob(
// 		HANDLE ProcessHandle,
// 		HANDLE JobHandle,
// 		PBOOL  Result
// );
//
//sys IsProcessInJob(procHandle windows.Handle, jobHandle windows.Handle, result *bool) (err error) = kernel32.IsProcessInJob

// BOOL QueryInformationJobObject(
//		HANDLE             hJob,
//		JOBOBJECTINFOCLASS JobObjectInformationClass,
//		LPVOID             lpJobObjectInformation,
//		DWORD              cbJobObjectInformationLength,
//		LPDWORD            lpReturnLength
// );
//
//sys QueryInformationJobObject(jobHandle windows.Handle, infoClass uint32, jobObjectInfo uintptr, jobObjectInformationLength uint32, lpReturnLength *uint32) (err error) = kernel32.QueryInformationJobObject

// HANDLE OpenJobObjectW(
//		DWORD   dwDesiredAccess,
//		BOOL    bInheritHandle,
//		LPCWSTR lpName
// );
//
//sys OpenJobObject(desiredAccess uint32, inheritHandle bool, lpName *uint16) (handle windows.Handle, err error) = kernel32.OpenJobObjectW

// DWORD SetIoRateControlInformationJobObject(
//		HANDLE                                hJob,
//		JOBOBJECT_IO_RATE_CONTROL_INFORMATION *IoRateControlInfo
// );
//
//sys SetIoRateControlInformationJobObject(jobHandle windows.Handle, ioRateControlInfo *JOBOBJECT_IO_RATE_CONTROL_INFORMATION) (ret uint32, err error) = kernel32.SetIoRateControlInformationJobObject

// DWORD QueryIoRateControlInformationJobObject(
// 		HANDLE                                hJob,
// 		PCWSTR                                VolumeName,
//		JOBOBJECT_IO_RATE_CONTROL_INFORMATION **InfoBlocks,
// 		ULONG                                 *InfoBlockCount
// );
//sys QueryIoRateControlInformationJobObject(jobHandle windows.Handle, volumeName *uint16, ioRateControlInfo **JOBOBJECT_IO_RATE_CONTROL_INFORMATION, infoBlockCount *uint32) (ret uint32, err error) = kernel32.QueryIoRateControlInformationJobObject

// NTSTATUS
// NtOpenJobObject (
//     _Out_ PHANDLE JobHandle,
//     _In_ ACCESS_MASK DesiredAccess,
//     _In_ POBJECT_ATTRIBUTES ObjectAttributes
// );
//sys NtOpenJobObject(jobHandle *windows.Handle, desiredAccess uint32, objAttributes *ObjectAttributes) (status uint32) = ntdll.NtOpenJobObject

// NTSTATUS
// NTAPI
// NtCreateJobObject (
//     _Out_ PHANDLE JobHandle,
//     _In_ ACCESS_MASK DesiredAccess,
//     _In_opt_ POBJECT_ATTRIBUTES ObjectAttributes
// );
//sys NtCreateJobObject(jobHandle *windows.Handle, desiredAccess uint32, objAttributes *ObjectAttributes) (status uint32) = ntdll.NtCreateJobObject

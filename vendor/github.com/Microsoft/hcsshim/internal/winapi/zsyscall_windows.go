// Code generated mksyscall_windows.exe DO NOT EDIT

package winapi

import (
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var _ unsafe.Pointer

// Do the interface allocations only once for common
// Errno values.
const (
	errnoERROR_IO_PENDING = 997
)

var (
	errERROR_IO_PENDING error = syscall.Errno(errnoERROR_IO_PENDING)
)

// errnoErr returns common boxed Errno values, to prevent
// allocations at runtime.
func errnoErr(e syscall.Errno) error {
	switch e {
	case 0:
		return nil
	case errnoERROR_IO_PENDING:
		return errERROR_IO_PENDING
	}
	// TODO: add more here, after collecting data on the common
	// error values see on Windows. (perhaps when running
	// all.bat?)
	return e
}

var (
	modntdll    = windows.NewLazySystemDLL("ntdll.dll")
	modiphlpapi = windows.NewLazySystemDLL("iphlpapi.dll")
	modkernel32 = windows.NewLazySystemDLL("kernel32.dll")
	modadvapi32 = windows.NewLazySystemDLL("advapi32.dll")
	modpsapi    = windows.NewLazySystemDLL("psapi.dll")
	modcfgmgr32 = windows.NewLazySystemDLL("cfgmgr32.dll")

	procNtQuerySystemInformation               = modntdll.NewProc("NtQuerySystemInformation")
	procSetJobCompartmentId                    = modiphlpapi.NewProc("SetJobCompartmentId")
	procSearchPathW                            = modkernel32.NewProc("SearchPathW")
	procCreateRemoteThread                     = modkernel32.NewProc("CreateRemoteThread")
	procGetQueuedCompletionStatus              = modkernel32.NewProc("GetQueuedCompletionStatus")
	procIsProcessInJob                         = modkernel32.NewProc("IsProcessInJob")
	procQueryInformationJobObject              = modkernel32.NewProc("QueryInformationJobObject")
	procOpenJobObjectW                         = modkernel32.NewProc("OpenJobObjectW")
	procSetIoRateControlInformationJobObject   = modkernel32.NewProc("SetIoRateControlInformationJobObject")
	procQueryIoRateControlInformationJobObject = modkernel32.NewProc("QueryIoRateControlInformationJobObject")
	procNtOpenJobObject                        = modntdll.NewProc("NtOpenJobObject")
	procNtCreateJobObject                      = modntdll.NewProc("NtCreateJobObject")
	procLogonUserW                             = modadvapi32.NewProc("LogonUserW")
	procRtlMoveMemory                          = modkernel32.NewProc("RtlMoveMemory")
	procLocalAlloc                             = modkernel32.NewProc("LocalAlloc")
	procLocalFree                              = modkernel32.NewProc("LocalFree")
	procQueryWorkingSet                        = modpsapi.NewProc("QueryWorkingSet")
	procGetProcessImageFileNameW               = modkernel32.NewProc("GetProcessImageFileNameW")
	procGetActiveProcessorCount                = modkernel32.NewProc("GetActiveProcessorCount")
	procCM_Get_Device_ID_List_SizeA            = modcfgmgr32.NewProc("CM_Get_Device_ID_List_SizeA")
	procCM_Get_Device_ID_ListA                 = modcfgmgr32.NewProc("CM_Get_Device_ID_ListA")
	procCM_Locate_DevNodeW                     = modcfgmgr32.NewProc("CM_Locate_DevNodeW")
	procCM_Get_DevNode_PropertyW               = modcfgmgr32.NewProc("CM_Get_DevNode_PropertyW")
	procNtCreateFile                           = modntdll.NewProc("NtCreateFile")
	procNtSetInformationFile                   = modntdll.NewProc("NtSetInformationFile")
	procNtOpenDirectoryObject                  = modntdll.NewProc("NtOpenDirectoryObject")
	procNtQueryDirectoryObject                 = modntdll.NewProc("NtQueryDirectoryObject")
	procRtlNtStatusToDosError                  = modntdll.NewProc("RtlNtStatusToDosError")
)

func NtQuerySystemInformation(systemInfoClass int, systemInformation uintptr, systemInfoLength uint32, returnLength *uint32) (status uint32) {
	r0, _, _ := syscall.Syscall6(procNtQuerySystemInformation.Addr(), 4, uintptr(systemInfoClass), uintptr(systemInformation), uintptr(systemInfoLength), uintptr(unsafe.Pointer(returnLength)), 0, 0)
	status = uint32(r0)
	return
}

func SetJobCompartmentId(handle windows.Handle, compartmentId uint32) (win32Err error) {
	r0, _, _ := syscall.Syscall(procSetJobCompartmentId.Addr(), 2, uintptr(handle), uintptr(compartmentId), 0)
	if r0 != 0 {
		win32Err = syscall.Errno(r0)
	}
	return
}

func SearchPath(lpPath *uint16, lpFileName *uint16, lpExtension *uint16, nBufferLength uint32, lpBuffer *uint16, lpFilePath *uint16) (size uint32, err error) {
	r0, _, e1 := syscall.Syscall6(procSearchPathW.Addr(), 6, uintptr(unsafe.Pointer(lpPath)), uintptr(unsafe.Pointer(lpFileName)), uintptr(unsafe.Pointer(lpExtension)), uintptr(nBufferLength), uintptr(unsafe.Pointer(lpBuffer)), uintptr(unsafe.Pointer(lpFilePath)))
	size = uint32(r0)
	if size == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func CreateRemoteThread(process windows.Handle, sa *windows.SecurityAttributes, stackSize uint32, startAddr uintptr, parameter uintptr, creationFlags uint32, threadID *uint32) (handle windows.Handle, err error) {
	r0, _, e1 := syscall.Syscall9(procCreateRemoteThread.Addr(), 7, uintptr(process), uintptr(unsafe.Pointer(sa)), uintptr(stackSize), uintptr(startAddr), uintptr(parameter), uintptr(creationFlags), uintptr(unsafe.Pointer(threadID)), 0, 0)
	handle = windows.Handle(r0)
	if handle == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func GetQueuedCompletionStatus(cphandle windows.Handle, qty *uint32, key *uintptr, overlapped **windows.Overlapped, timeout uint32) (err error) {
	r1, _, e1 := syscall.Syscall6(procGetQueuedCompletionStatus.Addr(), 5, uintptr(cphandle), uintptr(unsafe.Pointer(qty)), uintptr(unsafe.Pointer(key)), uintptr(unsafe.Pointer(overlapped)), uintptr(timeout), 0)
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func IsProcessInJob(procHandle windows.Handle, jobHandle windows.Handle, result *bool) (err error) {
	r1, _, e1 := syscall.Syscall(procIsProcessInJob.Addr(), 3, uintptr(procHandle), uintptr(jobHandle), uintptr(unsafe.Pointer(result)))
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func QueryInformationJobObject(jobHandle windows.Handle, infoClass uint32, jobObjectInfo uintptr, jobObjectInformationLength uint32, lpReturnLength *uint32) (err error) {
	r1, _, e1 := syscall.Syscall6(procQueryInformationJobObject.Addr(), 5, uintptr(jobHandle), uintptr(infoClass), uintptr(jobObjectInfo), uintptr(jobObjectInformationLength), uintptr(unsafe.Pointer(lpReturnLength)), 0)
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func OpenJobObject(desiredAccess uint32, inheritHandle bool, lpName *uint16) (handle windows.Handle, err error) {
	var _p0 uint32
	if inheritHandle {
		_p0 = 1
	} else {
		_p0 = 0
	}
	r0, _, e1 := syscall.Syscall(procOpenJobObjectW.Addr(), 3, uintptr(desiredAccess), uintptr(_p0), uintptr(unsafe.Pointer(lpName)))
	handle = windows.Handle(r0)
	if handle == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func SetIoRateControlInformationJobObject(jobHandle windows.Handle, ioRateControlInfo *JOBOBJECT_IO_RATE_CONTROL_INFORMATION) (ret uint32, err error) {
	r0, _, e1 := syscall.Syscall(procSetIoRateControlInformationJobObject.Addr(), 2, uintptr(jobHandle), uintptr(unsafe.Pointer(ioRateControlInfo)), 0)
	ret = uint32(r0)
	if ret == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func QueryIoRateControlInformationJobObject(jobHandle windows.Handle, volumeName *uint16, ioRateControlInfo **JOBOBJECT_IO_RATE_CONTROL_INFORMATION, infoBlockCount *uint32) (ret uint32, err error) {
	r0, _, e1 := syscall.Syscall6(procQueryIoRateControlInformationJobObject.Addr(), 4, uintptr(jobHandle), uintptr(unsafe.Pointer(volumeName)), uintptr(unsafe.Pointer(ioRateControlInfo)), uintptr(unsafe.Pointer(infoBlockCount)), 0, 0)
	ret = uint32(r0)
	if ret == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func NtOpenJobObject(jobHandle *windows.Handle, desiredAccess uint32, objAttributes *ObjectAttributes) (status uint32) {
	r0, _, _ := syscall.Syscall(procNtOpenJobObject.Addr(), 3, uintptr(unsafe.Pointer(jobHandle)), uintptr(desiredAccess), uintptr(unsafe.Pointer(objAttributes)))
	status = uint32(r0)
	return
}

func NtCreateJobObject(jobHandle *windows.Handle, desiredAccess uint32, objAttributes *ObjectAttributes) (status uint32) {
	r0, _, _ := syscall.Syscall(procNtCreateJobObject.Addr(), 3, uintptr(unsafe.Pointer(jobHandle)), uintptr(desiredAccess), uintptr(unsafe.Pointer(objAttributes)))
	status = uint32(r0)
	return
}

func LogonUser(username *uint16, domain *uint16, password *uint16, logonType uint32, logonProvider uint32, token *windows.Token) (err error) {
	r1, _, e1 := syscall.Syscall6(procLogonUserW.Addr(), 6, uintptr(unsafe.Pointer(username)), uintptr(unsafe.Pointer(domain)), uintptr(unsafe.Pointer(password)), uintptr(logonType), uintptr(logonProvider), uintptr(unsafe.Pointer(token)))
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func RtlMoveMemory(destination *byte, source *byte, length uintptr) (err error) {
	r1, _, e1 := syscall.Syscall(procRtlMoveMemory.Addr(), 3, uintptr(unsafe.Pointer(destination)), uintptr(unsafe.Pointer(source)), uintptr(length))
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func LocalAlloc(flags uint32, size int) (ptr uintptr) {
	r0, _, _ := syscall.Syscall(procLocalAlloc.Addr(), 2, uintptr(flags), uintptr(size), 0)
	ptr = uintptr(r0)
	return
}

func LocalFree(ptr uintptr) {
	syscall.Syscall(procLocalFree.Addr(), 1, uintptr(ptr), 0, 0)
	return
}

func QueryWorkingSet(handle windows.Handle, pv uintptr, cb uint32) (err error) {
	r1, _, e1 := syscall.Syscall(procQueryWorkingSet.Addr(), 3, uintptr(handle), uintptr(pv), uintptr(cb))
	if r1 == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func GetProcessImageFileName(hProcess windows.Handle, imageFileName *uint16, nSize uint32) (size uint32, err error) {
	r0, _, e1 := syscall.Syscall(procGetProcessImageFileNameW.Addr(), 3, uintptr(hProcess), uintptr(unsafe.Pointer(imageFileName)), uintptr(nSize))
	size = uint32(r0)
	if size == 0 {
		if e1 != 0 {
			err = errnoErr(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return
}

func GetActiveProcessorCount(groupNumber uint16) (amount uint32) {
	r0, _, _ := syscall.Syscall(procGetActiveProcessorCount.Addr(), 1, uintptr(groupNumber), 0, 0)
	amount = uint32(r0)
	return
}

func CMGetDeviceIDListSize(pulLen *uint32, pszFilter *byte, uFlags uint32) (hr error) {
	r0, _, _ := syscall.Syscall(procCM_Get_Device_ID_List_SizeA.Addr(), 3, uintptr(unsafe.Pointer(pulLen)), uintptr(unsafe.Pointer(pszFilter)), uintptr(uFlags))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func CMGetDeviceIDList(pszFilter *byte, buffer *byte, bufferLen uint32, uFlags uint32) (hr error) {
	r0, _, _ := syscall.Syscall6(procCM_Get_Device_ID_ListA.Addr(), 4, uintptr(unsafe.Pointer(pszFilter)), uintptr(unsafe.Pointer(buffer)), uintptr(bufferLen), uintptr(uFlags), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func CMLocateDevNode(pdnDevInst *uint32, pDeviceID string, uFlags uint32) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(pDeviceID)
	if hr != nil {
		return
	}
	return _CMLocateDevNode(pdnDevInst, _p0, uFlags)
}

func _CMLocateDevNode(pdnDevInst *uint32, pDeviceID *uint16, uFlags uint32) (hr error) {
	r0, _, _ := syscall.Syscall(procCM_Locate_DevNodeW.Addr(), 3, uintptr(unsafe.Pointer(pdnDevInst)), uintptr(unsafe.Pointer(pDeviceID)), uintptr(uFlags))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func CMGetDevNodeProperty(dnDevInst uint32, propertyKey *DevPropKey, propertyType *uint32, propertyBuffer *uint16, propertyBufferSize *uint32, uFlags uint32) (hr error) {
	r0, _, _ := syscall.Syscall6(procCM_Get_DevNode_PropertyW.Addr(), 6, uintptr(dnDevInst), uintptr(unsafe.Pointer(propertyKey)), uintptr(unsafe.Pointer(propertyType)), uintptr(unsafe.Pointer(propertyBuffer)), uintptr(unsafe.Pointer(propertyBufferSize)), uintptr(uFlags))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func NtCreateFile(handle *uintptr, accessMask uint32, oa *ObjectAttributes, iosb *IOStatusBlock, allocationSize *uint64, fileAttributes uint32, shareAccess uint32, createDisposition uint32, createOptions uint32, eaBuffer *byte, eaLength uint32) (status uint32) {
	r0, _, _ := syscall.Syscall12(procNtCreateFile.Addr(), 11, uintptr(unsafe.Pointer(handle)), uintptr(accessMask), uintptr(unsafe.Pointer(oa)), uintptr(unsafe.Pointer(iosb)), uintptr(unsafe.Pointer(allocationSize)), uintptr(fileAttributes), uintptr(shareAccess), uintptr(createDisposition), uintptr(createOptions), uintptr(unsafe.Pointer(eaBuffer)), uintptr(eaLength), 0)
	status = uint32(r0)
	return
}

func NtSetInformationFile(handle uintptr, iosb *IOStatusBlock, information uintptr, length uint32, class uint32) (status uint32) {
	r0, _, _ := syscall.Syscall6(procNtSetInformationFile.Addr(), 5, uintptr(handle), uintptr(unsafe.Pointer(iosb)), uintptr(information), uintptr(length), uintptr(class), 0)
	status = uint32(r0)
	return
}

func NtOpenDirectoryObject(handle *uintptr, accessMask uint32, oa *ObjectAttributes) (status uint32) {
	r0, _, _ := syscall.Syscall(procNtOpenDirectoryObject.Addr(), 3, uintptr(unsafe.Pointer(handle)), uintptr(accessMask), uintptr(unsafe.Pointer(oa)))
	status = uint32(r0)
	return
}

func NtQueryDirectoryObject(handle uintptr, buffer *byte, length uint32, singleEntry bool, restartScan bool, context *uint32, returnLength *uint32) (status uint32) {
	var _p0 uint32
	if singleEntry {
		_p0 = 1
	} else {
		_p0 = 0
	}
	var _p1 uint32
	if restartScan {
		_p1 = 1
	} else {
		_p1 = 0
	}
	r0, _, _ := syscall.Syscall9(procNtQueryDirectoryObject.Addr(), 7, uintptr(handle), uintptr(unsafe.Pointer(buffer)), uintptr(length), uintptr(_p0), uintptr(_p1), uintptr(unsafe.Pointer(context)), uintptr(unsafe.Pointer(returnLength)), 0, 0)
	status = uint32(r0)
	return
}

func RtlNtStatusToDosError(status uint32) (winerr error) {
	r0, _, _ := syscall.Syscall(procRtlNtStatusToDosError.Addr(), 1, uintptr(status), 0, 0)
	if r0 != 0 {
		winerr = syscall.Errno(r0)
	}
	return
}

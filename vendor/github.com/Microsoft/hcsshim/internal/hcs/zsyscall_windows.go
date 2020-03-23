// Code generated mksyscall_windows.exe DO NOT EDIT

package hcs

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
	modvmcompute = windows.NewLazySystemDLL("vmcompute.dll")

	procHcsEnumerateComputeSystems         = modvmcompute.NewProc("HcsEnumerateComputeSystems")
	procHcsCreateComputeSystem             = modvmcompute.NewProc("HcsCreateComputeSystem")
	procHcsOpenComputeSystem               = modvmcompute.NewProc("HcsOpenComputeSystem")
	procHcsCloseComputeSystem              = modvmcompute.NewProc("HcsCloseComputeSystem")
	procHcsStartComputeSystem              = modvmcompute.NewProc("HcsStartComputeSystem")
	procHcsShutdownComputeSystem           = modvmcompute.NewProc("HcsShutdownComputeSystem")
	procHcsTerminateComputeSystem          = modvmcompute.NewProc("HcsTerminateComputeSystem")
	procHcsPauseComputeSystem              = modvmcompute.NewProc("HcsPauseComputeSystem")
	procHcsResumeComputeSystem             = modvmcompute.NewProc("HcsResumeComputeSystem")
	procHcsGetComputeSystemProperties      = modvmcompute.NewProc("HcsGetComputeSystemProperties")
	procHcsModifyComputeSystem             = modvmcompute.NewProc("HcsModifyComputeSystem")
	procHcsRegisterComputeSystemCallback   = modvmcompute.NewProc("HcsRegisterComputeSystemCallback")
	procHcsUnregisterComputeSystemCallback = modvmcompute.NewProc("HcsUnregisterComputeSystemCallback")
	procHcsCreateProcess                   = modvmcompute.NewProc("HcsCreateProcess")
	procHcsOpenProcess                     = modvmcompute.NewProc("HcsOpenProcess")
	procHcsCloseProcess                    = modvmcompute.NewProc("HcsCloseProcess")
	procHcsTerminateProcess                = modvmcompute.NewProc("HcsTerminateProcess")
	procHcsSignalProcess                   = modvmcompute.NewProc("HcsSignalProcess")
	procHcsGetProcessInfo                  = modvmcompute.NewProc("HcsGetProcessInfo")
	procHcsGetProcessProperties            = modvmcompute.NewProc("HcsGetProcessProperties")
	procHcsModifyProcess                   = modvmcompute.NewProc("HcsModifyProcess")
	procHcsGetServiceProperties            = modvmcompute.NewProc("HcsGetServiceProperties")
	procHcsRegisterProcessCallback         = modvmcompute.NewProc("HcsRegisterProcessCallback")
	procHcsUnregisterProcessCallback       = modvmcompute.NewProc("HcsUnregisterProcessCallback")
)

func hcsEnumerateComputeSystems(query string, computeSystems **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(query)
	if hr != nil {
		return
	}
	return _hcsEnumerateComputeSystems(_p0, computeSystems, result)
}

func _hcsEnumerateComputeSystems(query *uint16, computeSystems **uint16, result **uint16) (hr error) {
	if hr = procHcsEnumerateComputeSystems.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsEnumerateComputeSystems.Addr(), 3, uintptr(unsafe.Pointer(query)), uintptr(unsafe.Pointer(computeSystems)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsCreateComputeSystem(id string, configuration string, identity syscall.Handle, computeSystem *hcsSystem, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(configuration)
	if hr != nil {
		return
	}
	return _hcsCreateComputeSystem(_p0, _p1, identity, computeSystem, result)
}

func _hcsCreateComputeSystem(id *uint16, configuration *uint16, identity syscall.Handle, computeSystem *hcsSystem, result **uint16) (hr error) {
	if hr = procHcsCreateComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsCreateComputeSystem.Addr(), 5, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(configuration)), uintptr(identity), uintptr(unsafe.Pointer(computeSystem)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsOpenComputeSystem(id string, computeSystem *hcsSystem, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _hcsOpenComputeSystem(_p0, computeSystem, result)
}

func _hcsOpenComputeSystem(id *uint16, computeSystem *hcsSystem, result **uint16) (hr error) {
	if hr = procHcsOpenComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsOpenComputeSystem.Addr(), 3, uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(computeSystem)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsCloseComputeSystem(computeSystem hcsSystem) (hr error) {
	if hr = procHcsCloseComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsCloseComputeSystem.Addr(), 1, uintptr(computeSystem), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsStartComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsStartComputeSystem(computeSystem, _p0, result)
}

func _hcsStartComputeSystem(computeSystem hcsSystem, options *uint16, result **uint16) (hr error) {
	if hr = procHcsStartComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsStartComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsShutdownComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsShutdownComputeSystem(computeSystem, _p0, result)
}

func _hcsShutdownComputeSystem(computeSystem hcsSystem, options *uint16, result **uint16) (hr error) {
	if hr = procHcsShutdownComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsShutdownComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsTerminateComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsTerminateComputeSystem(computeSystem, _p0, result)
}

func _hcsTerminateComputeSystem(computeSystem hcsSystem, options *uint16, result **uint16) (hr error) {
	if hr = procHcsTerminateComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsTerminateComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsPauseComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsPauseComputeSystem(computeSystem, _p0, result)
}

func _hcsPauseComputeSystem(computeSystem hcsSystem, options *uint16, result **uint16) (hr error) {
	if hr = procHcsPauseComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsPauseComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsResumeComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsResumeComputeSystem(computeSystem, _p0, result)
}

func _hcsResumeComputeSystem(computeSystem hcsSystem, options *uint16, result **uint16) (hr error) {
	if hr = procHcsResumeComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsResumeComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsGetComputeSystemProperties(computeSystem hcsSystem, propertyQuery string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(propertyQuery)
	if hr != nil {
		return
	}
	return _hcsGetComputeSystemProperties(computeSystem, _p0, properties, result)
}

func _hcsGetComputeSystemProperties(computeSystem hcsSystem, propertyQuery *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcsGetComputeSystemProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsGetComputeSystemProperties.Addr(), 4, uintptr(computeSystem), uintptr(unsafe.Pointer(propertyQuery)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsModifyComputeSystem(computeSystem hcsSystem, configuration string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(configuration)
	if hr != nil {
		return
	}
	return _hcsModifyComputeSystem(computeSystem, _p0, result)
}

func _hcsModifyComputeSystem(computeSystem hcsSystem, configuration *uint16, result **uint16) (hr error) {
	if hr = procHcsModifyComputeSystem.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsModifyComputeSystem.Addr(), 3, uintptr(computeSystem), uintptr(unsafe.Pointer(configuration)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsRegisterComputeSystemCallback(computeSystem hcsSystem, callback uintptr, context uintptr, callbackHandle *hcsCallback) (hr error) {
	if hr = procHcsRegisterComputeSystemCallback.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsRegisterComputeSystemCallback.Addr(), 4, uintptr(computeSystem), uintptr(callback), uintptr(context), uintptr(unsafe.Pointer(callbackHandle)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsUnregisterComputeSystemCallback(callbackHandle hcsCallback) (hr error) {
	if hr = procHcsUnregisterComputeSystemCallback.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsUnregisterComputeSystemCallback.Addr(), 1, uintptr(callbackHandle), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsCreateProcess(computeSystem hcsSystem, processParameters string, processInformation *hcsProcessInformation, process *hcsProcess, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(processParameters)
	if hr != nil {
		return
	}
	return _hcsCreateProcess(computeSystem, _p0, processInformation, process, result)
}

func _hcsCreateProcess(computeSystem hcsSystem, processParameters *uint16, processInformation *hcsProcessInformation, process *hcsProcess, result **uint16) (hr error) {
	if hr = procHcsCreateProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsCreateProcess.Addr(), 5, uintptr(computeSystem), uintptr(unsafe.Pointer(processParameters)), uintptr(unsafe.Pointer(processInformation)), uintptr(unsafe.Pointer(process)), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsOpenProcess(computeSystem hcsSystem, pid uint32, process *hcsProcess, result **uint16) (hr error) {
	if hr = procHcsOpenProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsOpenProcess.Addr(), 4, uintptr(computeSystem), uintptr(pid), uintptr(unsafe.Pointer(process)), uintptr(unsafe.Pointer(result)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsCloseProcess(process hcsProcess) (hr error) {
	if hr = procHcsCloseProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsCloseProcess.Addr(), 1, uintptr(process), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsTerminateProcess(process hcsProcess, result **uint16) (hr error) {
	if hr = procHcsTerminateProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsTerminateProcess.Addr(), 2, uintptr(process), uintptr(unsafe.Pointer(result)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsSignalProcess(process hcsProcess, options string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(options)
	if hr != nil {
		return
	}
	return _hcsSignalProcess(process, _p0, result)
}

func _hcsSignalProcess(process hcsProcess, options *uint16, result **uint16) (hr error) {
	if hr = procHcsSignalProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsSignalProcess.Addr(), 3, uintptr(process), uintptr(unsafe.Pointer(options)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsGetProcessInfo(process hcsProcess, processInformation *hcsProcessInformation, result **uint16) (hr error) {
	if hr = procHcsGetProcessInfo.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsGetProcessInfo.Addr(), 3, uintptr(process), uintptr(unsafe.Pointer(processInformation)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsGetProcessProperties(process hcsProcess, processProperties **uint16, result **uint16) (hr error) {
	if hr = procHcsGetProcessProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsGetProcessProperties.Addr(), 3, uintptr(process), uintptr(unsafe.Pointer(processProperties)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsModifyProcess(process hcsProcess, settings string, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(settings)
	if hr != nil {
		return
	}
	return _hcsModifyProcess(process, _p0, result)
}

func _hcsModifyProcess(process hcsProcess, settings *uint16, result **uint16) (hr error) {
	if hr = procHcsModifyProcess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsModifyProcess.Addr(), 3, uintptr(process), uintptr(unsafe.Pointer(settings)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsGetServiceProperties(propertyQuery string, properties **uint16, result **uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(propertyQuery)
	if hr != nil {
		return
	}
	return _hcsGetServiceProperties(_p0, properties, result)
}

func _hcsGetServiceProperties(propertyQuery *uint16, properties **uint16, result **uint16) (hr error) {
	if hr = procHcsGetServiceProperties.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsGetServiceProperties.Addr(), 3, uintptr(unsafe.Pointer(propertyQuery)), uintptr(unsafe.Pointer(properties)), uintptr(unsafe.Pointer(result)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsRegisterProcessCallback(process hcsProcess, callback uintptr, context uintptr, callbackHandle *hcsCallback) (hr error) {
	if hr = procHcsRegisterProcessCallback.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procHcsRegisterProcessCallback.Addr(), 4, uintptr(process), uintptr(callback), uintptr(context), uintptr(unsafe.Pointer(callbackHandle)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func hcsUnregisterProcessCallback(callbackHandle hcsCallback) (hr error) {
	if hr = procHcsUnregisterProcessCallback.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procHcsUnregisterProcessCallback.Addr(), 1, uintptr(callbackHandle), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

// Shim for the Host Compute Service (HCS) to manage Windows Server
// containers and Hyper-V containers.

package hcs

import (
	"syscall"
)

//go:generate go run ../../mksyscall_windows.go -output zsyscall_windows.go hcs.go

//sys hcsEnumerateComputeSystems(query string, computeSystems **uint16, result **uint16) (hr error) = vmcompute.HcsEnumerateComputeSystems?
//sys hcsCreateComputeSystem(id string, configuration string, identity syscall.Handle, computeSystem *hcsSystem, result **uint16) (hr error) = vmcompute.HcsCreateComputeSystem?
//sys hcsOpenComputeSystem(id string, computeSystem *hcsSystem, result **uint16) (hr error) = vmcompute.HcsOpenComputeSystem?
//sys hcsCloseComputeSystem(computeSystem hcsSystem) (hr error) = vmcompute.HcsCloseComputeSystem?
//sys hcsStartComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsStartComputeSystem?
//sys hcsShutdownComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsShutdownComputeSystem?
//sys hcsTerminateComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsTerminateComputeSystem?
//sys hcsPauseComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsPauseComputeSystem?
//sys hcsResumeComputeSystem(computeSystem hcsSystem, options string, result **uint16) (hr error) = vmcompute.HcsResumeComputeSystem?
//sys hcsGetComputeSystemProperties(computeSystem hcsSystem, propertyQuery string, properties **uint16, result **uint16) (hr error) = vmcompute.HcsGetComputeSystemProperties?
//sys hcsModifyComputeSystem(computeSystem hcsSystem, configuration string, result **uint16) (hr error) = vmcompute.HcsModifyComputeSystem?
//sys hcsRegisterComputeSystemCallback(computeSystem hcsSystem, callback uintptr, context uintptr, callbackHandle *hcsCallback) (hr error) = vmcompute.HcsRegisterComputeSystemCallback?
//sys hcsUnregisterComputeSystemCallback(callbackHandle hcsCallback) (hr error) = vmcompute.HcsUnregisterComputeSystemCallback?

//sys hcsCreateProcess(computeSystem hcsSystem, processParameters string, processInformation *hcsProcessInformation, process *hcsProcess, result **uint16) (hr error) = vmcompute.HcsCreateProcess?
//sys hcsOpenProcess(computeSystem hcsSystem, pid uint32, process *hcsProcess, result **uint16) (hr error) = vmcompute.HcsOpenProcess?
//sys hcsCloseProcess(process hcsProcess) (hr error) = vmcompute.HcsCloseProcess?
//sys hcsTerminateProcess(process hcsProcess, result **uint16) (hr error) = vmcompute.HcsTerminateProcess?
//sys hcsSignalProcess(process hcsProcess, options string, result **uint16) (hr error) = vmcompute.HcsTerminateProcess?
//sys hcsGetProcessInfo(process hcsProcess, processInformation *hcsProcessInformation, result **uint16) (hr error) = vmcompute.HcsGetProcessInfo?
//sys hcsGetProcessProperties(process hcsProcess, processProperties **uint16, result **uint16) (hr error) = vmcompute.HcsGetProcessProperties?
//sys hcsModifyProcess(process hcsProcess, settings string, result **uint16) (hr error) = vmcompute.HcsModifyProcess?
//sys hcsGetServiceProperties(propertyQuery string, properties **uint16, result **uint16) (hr error) = vmcompute.HcsGetServiceProperties?
//sys hcsRegisterProcessCallback(process hcsProcess, callback uintptr, context uintptr, callbackHandle *hcsCallback) (hr error) = vmcompute.HcsRegisterProcessCallback?
//sys hcsUnregisterProcessCallback(callbackHandle hcsCallback) (hr error) = vmcompute.HcsUnregisterProcessCallback?

type hcsSystem syscall.Handle
type hcsProcess syscall.Handle
type hcsCallback syscall.Handle

type hcsProcessInformation struct {
	ProcessId uint32
	Reserved  uint32
	StdInput  syscall.Handle
	StdOutput syscall.Handle
	StdError  syscall.Handle
}

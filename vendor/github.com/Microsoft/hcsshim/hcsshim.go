// Shim for the Host Compute Service (HCS) to manage Windows Server
// containers and Hyper-V containers.

package hcsshim

import (
	"fmt"
	"syscall"
	"unsafe"

	"github.com/sirupsen/logrus"
)

//go:generate go run mksyscall_windows.go -output zhcsshim.go hcsshim.go safeopen.go

//sys coTaskMemFree(buffer unsafe.Pointer) = ole32.CoTaskMemFree
//sys SetCurrentThreadCompartmentId(compartmentId uint32) (hr error) = iphlpapi.SetCurrentThreadCompartmentId

//sys activateLayer(info *driverInfo, id string) (hr error) = vmcompute.ActivateLayer?
//sys copyLayer(info *driverInfo, srcId string, dstId string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.CopyLayer?
//sys createLayer(info *driverInfo, id string, parent string) (hr error) = vmcompute.CreateLayer?
//sys createSandboxLayer(info *driverInfo, id string, parent string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.CreateSandboxLayer?
//sys expandSandboxSize(info *driverInfo, id string, size uint64) (hr error) = vmcompute.ExpandSandboxSize?
//sys deactivateLayer(info *driverInfo, id string) (hr error) = vmcompute.DeactivateLayer?
//sys destroyLayer(info *driverInfo, id string) (hr error) = vmcompute.DestroyLayer?
//sys exportLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.ExportLayer?
//sys getLayerMountPath(info *driverInfo, id string, length *uintptr, buffer *uint16) (hr error) = vmcompute.GetLayerMountPath?
//sys getBaseImages(buffer **uint16) (hr error) = vmcompute.GetBaseImages?
//sys importLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.ImportLayer?
//sys layerExists(info *driverInfo, id string, exists *uint32) (hr error) = vmcompute.LayerExists?
//sys nameToGuid(name string, guid *GUID) (hr error) = vmcompute.NameToGuid?
//sys prepareLayer(info *driverInfo, id string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) = vmcompute.PrepareLayer?
//sys unprepareLayer(info *driverInfo, id string) (hr error) = vmcompute.UnprepareLayer?
//sys processBaseImage(path string) (hr error) = vmcompute.ProcessBaseImage?
//sys processUtilityImage(path string) (hr error) = vmcompute.ProcessUtilityImage?

//sys importLayerBegin(info *driverInfo, id string, descriptors []WC_LAYER_DESCRIPTOR, context *uintptr) (hr error) = vmcompute.ImportLayerBegin?
//sys importLayerNext(context uintptr, fileName string, fileInfo *winio.FileBasicInfo) (hr error) = vmcompute.ImportLayerNext?
//sys importLayerWrite(context uintptr, buffer []byte) (hr error) = vmcompute.ImportLayerWrite?
//sys importLayerEnd(context uintptr) (hr error) = vmcompute.ImportLayerEnd?

//sys exportLayerBegin(info *driverInfo, id string, descriptors []WC_LAYER_DESCRIPTOR, context *uintptr) (hr error) = vmcompute.ExportLayerBegin?
//sys exportLayerNext(context uintptr, fileName **uint16, fileInfo *winio.FileBasicInfo, fileSize *int64, deleted *uint32) (hr error) = vmcompute.ExportLayerNext?
//sys exportLayerRead(context uintptr, buffer []byte, bytesRead *uint32) (hr error) = vmcompute.ExportLayerRead?
//sys exportLayerEnd(context uintptr) (hr error) = vmcompute.ExportLayerEnd?

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
//sys hcsGetProcessInfo(process hcsProcess, processInformation *hcsProcessInformation, result **uint16) (hr error) = vmcompute.HcsGetProcessInfo?
//sys hcsGetProcessProperties(process hcsProcess, processProperties **uint16, result **uint16) (hr error) = vmcompute.HcsGetProcessProperties?
//sys hcsModifyProcess(process hcsProcess, settings string, result **uint16) (hr error) = vmcompute.HcsModifyProcess?
//sys hcsGetServiceProperties(propertyQuery string, properties **uint16, result **uint16) (hr error) = vmcompute.HcsGetServiceProperties?
//sys hcsRegisterProcessCallback(process hcsProcess, callback uintptr, context uintptr, callbackHandle *hcsCallback) (hr error) = vmcompute.HcsRegisterProcessCallback?
//sys hcsUnregisterProcessCallback(callbackHandle hcsCallback) (hr error) = vmcompute.HcsUnregisterProcessCallback?

//sys hcsModifyServiceSettings(settings string, result **uint16) (hr error) = vmcompute.HcsModifyServiceSettings?

//sys _hnsCall(method string, path string, object string, response **uint16) (hr error) = vmcompute.HNSCall?

const (
	// Specific user-visible exit codes
	WaitErrExecFailed = 32767

	ERROR_GEN_FAILURE          = syscall.Errno(31)
	ERROR_SHUTDOWN_IN_PROGRESS = syscall.Errno(1115)
	WSAEINVAL                  = syscall.Errno(10022)

	// Timeout on wait calls
	TimeoutInfinite = 0xFFFFFFFF
)

type HcsError struct {
	title string
	rest  string
	Err   error
}

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

func makeError(err error, title, rest string) error {
	// Pass through DLL errors directly since they do not originate from HCS.
	if _, ok := err.(*syscall.DLLError); ok {
		return err
	}
	return &HcsError{title, rest, err}
}

func makeErrorf(err error, title, format string, a ...interface{}) error {
	return makeError(err, title, fmt.Sprintf(format, a...))
}

func win32FromError(err error) uint32 {
	if herr, ok := err.(*HcsError); ok {
		return win32FromError(herr.Err)
	}
	if code, ok := err.(syscall.Errno); ok {
		return uint32(code)
	}
	return uint32(ERROR_GEN_FAILURE)
}

func win32FromHresult(hr uintptr) uintptr {
	if hr&0x1fff0000 == 0x00070000 {
		return hr & 0xffff
	}
	return hr
}

func (e *HcsError) Error() string {
	s := e.title
	if len(s) > 0 && s[len(s)-1] != ' ' {
		s += " "
	}
	s += fmt.Sprintf("failed in Win32: %s (0x%x)", e.Err, win32FromError(e.Err))
	if e.rest != "" {
		if e.rest[0] != ' ' {
			s += " "
		}
		s += e.rest
	}
	return s
}

func convertAndFreeCoTaskMemString(buffer *uint16) string {
	str := syscall.UTF16ToString((*[1 << 30]uint16)(unsafe.Pointer(buffer))[:])
	coTaskMemFree(unsafe.Pointer(buffer))
	return str
}

func convertAndFreeCoTaskMemBytes(buffer *uint16) []byte {
	return []byte(convertAndFreeCoTaskMemString(buffer))
}

func processHcsResult(err error, resultp *uint16) error {
	if resultp != nil {
		result := convertAndFreeCoTaskMemString(resultp)
		logrus.Debugf("Result: %s", result)
	}
	return err
}

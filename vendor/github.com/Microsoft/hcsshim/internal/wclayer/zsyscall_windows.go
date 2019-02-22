// Code generated mksyscall_windows.exe DO NOT EDIT

package wclayer

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

	procActivateLayer       = modvmcompute.NewProc("ActivateLayer")
	procCopyLayer           = modvmcompute.NewProc("CopyLayer")
	procCreateLayer         = modvmcompute.NewProc("CreateLayer")
	procCreateSandboxLayer  = modvmcompute.NewProc("CreateSandboxLayer")
	procExpandSandboxSize   = modvmcompute.NewProc("ExpandSandboxSize")
	procDeactivateLayer     = modvmcompute.NewProc("DeactivateLayer")
	procDestroyLayer        = modvmcompute.NewProc("DestroyLayer")
	procExportLayer         = modvmcompute.NewProc("ExportLayer")
	procGetLayerMountPath   = modvmcompute.NewProc("GetLayerMountPath")
	procGetBaseImages       = modvmcompute.NewProc("GetBaseImages")
	procImportLayer         = modvmcompute.NewProc("ImportLayer")
	procLayerExists         = modvmcompute.NewProc("LayerExists")
	procNameToGuid          = modvmcompute.NewProc("NameToGuid")
	procPrepareLayer        = modvmcompute.NewProc("PrepareLayer")
	procUnprepareLayer      = modvmcompute.NewProc("UnprepareLayer")
	procProcessBaseImage    = modvmcompute.NewProc("ProcessBaseImage")
	procProcessUtilityImage = modvmcompute.NewProc("ProcessUtilityImage")
	procGrantVmAccess       = modvmcompute.NewProc("GrantVmAccess")
)

func activateLayer(info *driverInfo, id string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _activateLayer(info, _p0)
}

func _activateLayer(info *driverInfo, id *uint16) (hr error) {
	if hr = procActivateLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procActivateLayer.Addr(), 2, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func copyLayer(info *driverInfo, srcId string, dstId string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(srcId)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(dstId)
	if hr != nil {
		return
	}
	return _copyLayer(info, _p0, _p1, descriptors)
}

func _copyLayer(info *driverInfo, srcId *uint16, dstId *uint16, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p2 *WC_LAYER_DESCRIPTOR
	if len(descriptors) > 0 {
		_p2 = &descriptors[0]
	}
	if hr = procCopyLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procCopyLayer.Addr(), 5, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(srcId)), uintptr(unsafe.Pointer(dstId)), uintptr(unsafe.Pointer(_p2)), uintptr(len(descriptors)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func createLayer(info *driverInfo, id string, parent string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(parent)
	if hr != nil {
		return
	}
	return _createLayer(info, _p0, _p1)
}

func _createLayer(info *driverInfo, id *uint16, parent *uint16) (hr error) {
	if hr = procCreateLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procCreateLayer.Addr(), 3, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(parent)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func createSandboxLayer(info *driverInfo, id string, parent uintptr, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _createSandboxLayer(info, _p0, parent, descriptors)
}

func _createSandboxLayer(info *driverInfo, id *uint16, parent uintptr, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p1 *WC_LAYER_DESCRIPTOR
	if len(descriptors) > 0 {
		_p1 = &descriptors[0]
	}
	if hr = procCreateSandboxLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procCreateSandboxLayer.Addr(), 5, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(parent), uintptr(unsafe.Pointer(_p1)), uintptr(len(descriptors)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func expandSandboxSize(info *driverInfo, id string, size uint64) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _expandSandboxSize(info, _p0, size)
}

func _expandSandboxSize(info *driverInfo, id *uint16, size uint64) (hr error) {
	if hr = procExpandSandboxSize.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procExpandSandboxSize.Addr(), 3, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(size))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func deactivateLayer(info *driverInfo, id string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _deactivateLayer(info, _p0)
}

func _deactivateLayer(info *driverInfo, id *uint16) (hr error) {
	if hr = procDeactivateLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procDeactivateLayer.Addr(), 2, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func destroyLayer(info *driverInfo, id string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _destroyLayer(info, _p0)
}

func _destroyLayer(info *driverInfo, id *uint16) (hr error) {
	if hr = procDestroyLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procDestroyLayer.Addr(), 2, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func exportLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(path)
	if hr != nil {
		return
	}
	return _exportLayer(info, _p0, _p1, descriptors)
}

func _exportLayer(info *driverInfo, id *uint16, path *uint16, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p2 *WC_LAYER_DESCRIPTOR
	if len(descriptors) > 0 {
		_p2 = &descriptors[0]
	}
	if hr = procExportLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procExportLayer.Addr(), 5, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(path)), uintptr(unsafe.Pointer(_p2)), uintptr(len(descriptors)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func getLayerMountPath(info *driverInfo, id string, length *uintptr, buffer *uint16) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _getLayerMountPath(info, _p0, length, buffer)
}

func _getLayerMountPath(info *driverInfo, id *uint16, length *uintptr, buffer *uint16) (hr error) {
	if hr = procGetLayerMountPath.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procGetLayerMountPath.Addr(), 4, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(length)), uintptr(unsafe.Pointer(buffer)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func getBaseImages(buffer **uint16) (hr error) {
	if hr = procGetBaseImages.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procGetBaseImages.Addr(), 1, uintptr(unsafe.Pointer(buffer)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func importLayer(info *driverInfo, id string, path string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(path)
	if hr != nil {
		return
	}
	return _importLayer(info, _p0, _p1, descriptors)
}

func _importLayer(info *driverInfo, id *uint16, path *uint16, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p2 *WC_LAYER_DESCRIPTOR
	if len(descriptors) > 0 {
		_p2 = &descriptors[0]
	}
	if hr = procImportLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procImportLayer.Addr(), 5, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(path)), uintptr(unsafe.Pointer(_p2)), uintptr(len(descriptors)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func layerExists(info *driverInfo, id string, exists *uint32) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _layerExists(info, _p0, exists)
}

func _layerExists(info *driverInfo, id *uint16, exists *uint32) (hr error) {
	if hr = procLayerExists.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procLayerExists.Addr(), 3, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(exists)))
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func nameToGuid(name string, guid *_guid) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(name)
	if hr != nil {
		return
	}
	return _nameToGuid(_p0, guid)
}

func _nameToGuid(name *uint16, guid *_guid) (hr error) {
	if hr = procNameToGuid.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procNameToGuid.Addr(), 2, uintptr(unsafe.Pointer(name)), uintptr(unsafe.Pointer(guid)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func prepareLayer(info *driverInfo, id string, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _prepareLayer(info, _p0, descriptors)
}

func _prepareLayer(info *driverInfo, id *uint16, descriptors []WC_LAYER_DESCRIPTOR) (hr error) {
	var _p1 *WC_LAYER_DESCRIPTOR
	if len(descriptors) > 0 {
		_p1 = &descriptors[0]
	}
	if hr = procPrepareLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall6(procPrepareLayer.Addr(), 4, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), uintptr(unsafe.Pointer(_p1)), uintptr(len(descriptors)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func unprepareLayer(info *driverInfo, id string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(id)
	if hr != nil {
		return
	}
	return _unprepareLayer(info, _p0)
}

func _unprepareLayer(info *driverInfo, id *uint16) (hr error) {
	if hr = procUnprepareLayer.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procUnprepareLayer.Addr(), 2, uintptr(unsafe.Pointer(info)), uintptr(unsafe.Pointer(id)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func processBaseImage(path string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(path)
	if hr != nil {
		return
	}
	return _processBaseImage(_p0)
}

func _processBaseImage(path *uint16) (hr error) {
	if hr = procProcessBaseImage.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procProcessBaseImage.Addr(), 1, uintptr(unsafe.Pointer(path)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func processUtilityImage(path string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(path)
	if hr != nil {
		return
	}
	return _processUtilityImage(_p0)
}

func _processUtilityImage(path *uint16) (hr error) {
	if hr = procProcessUtilityImage.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procProcessUtilityImage.Addr(), 1, uintptr(unsafe.Pointer(path)), 0, 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

func grantVmAccess(vmid string, filepath string) (hr error) {
	var _p0 *uint16
	_p0, hr = syscall.UTF16PtrFromString(vmid)
	if hr != nil {
		return
	}
	var _p1 *uint16
	_p1, hr = syscall.UTF16PtrFromString(filepath)
	if hr != nil {
		return
	}
	return _grantVmAccess(_p0, _p1)
}

func _grantVmAccess(vmid *uint16, filepath *uint16) (hr error) {
	if hr = procGrantVmAccess.Find(); hr != nil {
		return
	}
	r0, _, _ := syscall.Syscall(procGrantVmAccess.Addr(), 2, uintptr(unsafe.Pointer(vmid)), uintptr(unsafe.Pointer(filepath)), 0)
	if int32(r0) < 0 {
		if r0&0x1fff0000 == 0x00070000 {
			r0 &= 0xffff
		}
		hr = syscall.Errno(r0)
	}
	return
}

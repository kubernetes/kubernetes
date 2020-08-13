// +build windows

package vhd

import "syscall"

//go:generate go run mksyscall_windows.go -output zvhd.go vhd.go

//sys createVirtualDisk(virtualStorageType *virtualStorageType, path string, virtualDiskAccessMask uint32, securityDescriptor *uintptr, flags uint32, providerSpecificFlags uint32, parameters *createVirtualDiskParameters, o *syscall.Overlapped, handle *syscall.Handle) (err error) [failretval != 0] = VirtDisk.CreateVirtualDisk
//sys openVirtualDisk(virtualStorageType *virtualStorageType, path string, virtualDiskAccessMask uint32, flags uint32, parameters *openVirtualDiskParameters, handle *syscall.Handle) (err error) [failretval != 0] = VirtDisk.OpenVirtualDisk
//sys detachVirtualDisk(handle syscall.Handle, flags uint32, providerSpecificFlags uint32) (err error) [failretval != 0] = VirtDisk.DetachVirtualDisk

type virtualStorageType struct {
	DeviceID uint32
	VendorID [16]byte
}

type (
	createVirtualDiskFlag uint32
	VirtualDiskAccessMask uint32
	VirtualDiskFlag       uint32
)

const (
	// Flags for creating a VHD (not exported)
	createVirtualDiskFlagNone                        createVirtualDiskFlag = 0
	createVirtualDiskFlagFullPhysicalAllocation      createVirtualDiskFlag = 1
	createVirtualDiskFlagPreventWritesToSourceDisk   createVirtualDiskFlag = 2
	createVirtualDiskFlagDoNotCopyMetadataFromParent createVirtualDiskFlag = 4

	// Access Mask for opening a VHD
	VirtualDiskAccessNone     VirtualDiskAccessMask = 0
	VirtualDiskAccessAttachRO VirtualDiskAccessMask = 65536
	VirtualDiskAccessAttachRW VirtualDiskAccessMask = 131072
	VirtualDiskAccessDetach   VirtualDiskAccessMask = 262144
	VirtualDiskAccessGetInfo  VirtualDiskAccessMask = 524288
	VirtualDiskAccessCreate   VirtualDiskAccessMask = 1048576
	VirtualDiskAccessMetaOps  VirtualDiskAccessMask = 2097152
	VirtualDiskAccessRead     VirtualDiskAccessMask = 851968
	VirtualDiskAccessAll      VirtualDiskAccessMask = 4128768
	VirtualDiskAccessWritable VirtualDiskAccessMask = 3276800

	// Flags for opening a VHD
	OpenVirtualDiskFlagNone                        VirtualDiskFlag = 0
	OpenVirtualDiskFlagNoParents                   VirtualDiskFlag = 0x1
	OpenVirtualDiskFlagBlankFile                   VirtualDiskFlag = 0x2
	OpenVirtualDiskFlagBootDrive                   VirtualDiskFlag = 0x4
	OpenVirtualDiskFlagCachedIO                    VirtualDiskFlag = 0x8
	OpenVirtualDiskFlagCustomDiffChain             VirtualDiskFlag = 0x10
	OpenVirtualDiskFlagParentCachedIO              VirtualDiskFlag = 0x20
	OpenVirtualDiskFlagVhdSetFileOnly              VirtualDiskFlag = 0x40
	OpenVirtualDiskFlagIgnoreRelativeParentLocator VirtualDiskFlag = 0x80
	OpenVirtualDiskFlagNoWriteHardening            VirtualDiskFlag = 0x100
)

type createVersion2 struct {
	UniqueID                 [16]byte // GUID
	MaximumSize              uint64
	BlockSizeInBytes         uint32
	SectorSizeInBytes        uint32
	ParentPath               *uint16 // string
	SourcePath               *uint16 // string
	OpenFlags                uint32
	ParentVirtualStorageType virtualStorageType
	SourceVirtualStorageType virtualStorageType
	ResiliencyGUID           [16]byte // GUID
}

type createVirtualDiskParameters struct {
	Version  uint32 // Must always be set to 2
	Version2 createVersion2
}

type openVersion2 struct {
	GetInfoOnly    int32    // bool but 4-byte aligned
	ReadOnly       int32    // bool but 4-byte aligned
	ResiliencyGUID [16]byte // GUID
}

type openVirtualDiskParameters struct {
	Version  uint32 // Must always be set to 2
	Version2 openVersion2
}

// CreateVhdx will create a simple vhdx file at the given path using default values.
func CreateVhdx(path string, maxSizeInGb, blockSizeInMb uint32) error {
	var (
		defaultType virtualStorageType
		handle      syscall.Handle
	)

	parameters := createVirtualDiskParameters{
		Version: 2,
		Version2: createVersion2{
			MaximumSize:      uint64(maxSizeInGb) * 1024 * 1024 * 1024,
			BlockSizeInBytes: blockSizeInMb * 1024 * 1024,
		},
	}

	if err := createVirtualDisk(
		&defaultType,
		path,
		uint32(VirtualDiskAccessNone),
		nil,
		uint32(createVirtualDiskFlagNone),
		0,
		&parameters,
		nil,
		&handle); err != nil {
		return err
	}

	if err := syscall.CloseHandle(handle); err != nil {
		return err
	}

	return nil
}

// DetachVhd detaches a mounted container layer vhd found at `path`.
func DetachVhd(path string) error {
	handle, err := OpenVirtualDisk(
		path,
		VirtualDiskAccessNone,
		OpenVirtualDiskFlagCachedIO|OpenVirtualDiskFlagIgnoreRelativeParentLocator)

	if err != nil {
		return err
	}
	defer syscall.CloseHandle(handle)
	return detachVirtualDisk(handle, 0, 0)
}

// OpenVirtualDisk obtains a handle to a VHD opened with supplied access mask and flags.
func OpenVirtualDisk(path string, accessMask VirtualDiskAccessMask, flag VirtualDiskFlag) (syscall.Handle, error) {
	var (
		defaultType virtualStorageType
		handle      syscall.Handle
	)
	parameters := openVirtualDiskParameters{Version: 2}
	if err := openVirtualDisk(
		&defaultType,
		path,
		uint32(accessMask),
		uint32(flag),
		&parameters,
		&handle); err != nil {
		return 0, err
	}
	return handle, nil
}

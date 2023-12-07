//go:build windows
// +build windows

package vhd

import (
	"fmt"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/guid"
	"golang.org/x/sys/windows"
)

//go:generate go run github.com/Microsoft/go-winio/tools/mkwinsyscall -output zvhd_windows.go vhd.go

//sys createVirtualDisk(virtualStorageType *VirtualStorageType, path string, virtualDiskAccessMask uint32, securityDescriptor *uintptr, createVirtualDiskFlags uint32, providerSpecificFlags uint32, parameters *CreateVirtualDiskParameters, overlapped *syscall.Overlapped, handle *syscall.Handle) (win32err error) = virtdisk.CreateVirtualDisk
//sys openVirtualDisk(virtualStorageType *VirtualStorageType, path string, virtualDiskAccessMask uint32, openVirtualDiskFlags uint32, parameters *openVirtualDiskParameters, handle *syscall.Handle) (win32err error) = virtdisk.OpenVirtualDisk
//sys attachVirtualDisk(handle syscall.Handle, securityDescriptor *uintptr, attachVirtualDiskFlag uint32, providerSpecificFlags uint32, parameters *AttachVirtualDiskParameters, overlapped *syscall.Overlapped) (win32err error) = virtdisk.AttachVirtualDisk
//sys detachVirtualDisk(handle syscall.Handle, detachVirtualDiskFlags uint32, providerSpecificFlags uint32) (win32err error) = virtdisk.DetachVirtualDisk
//sys getVirtualDiskPhysicalPath(handle syscall.Handle, diskPathSizeInBytes *uint32, buffer *uint16) (win32err error) = virtdisk.GetVirtualDiskPhysicalPath

type (
	CreateVirtualDiskFlag uint32
	VirtualDiskFlag       uint32
	AttachVirtualDiskFlag uint32
	DetachVirtualDiskFlag uint32
	VirtualDiskAccessMask uint32
)

type VirtualStorageType struct {
	DeviceID uint32
	VendorID guid.GUID
}

type CreateVersion2 struct {
	UniqueID                 guid.GUID
	MaximumSize              uint64
	BlockSizeInBytes         uint32
	SectorSizeInBytes        uint32
	PhysicalSectorSizeInByte uint32
	ParentPath               *uint16 // string
	SourcePath               *uint16 // string
	OpenFlags                uint32
	ParentVirtualStorageType VirtualStorageType
	SourceVirtualStorageType VirtualStorageType
	ResiliencyGUID           guid.GUID
}

type CreateVirtualDiskParameters struct {
	Version  uint32 // Must always be set to 2
	Version2 CreateVersion2
}

type OpenVersion2 struct {
	GetInfoOnly    bool
	ReadOnly       bool
	ResiliencyGUID guid.GUID
}

type OpenVirtualDiskParameters struct {
	Version  uint32 // Must always be set to 2
	Version2 OpenVersion2
}

// The higher level `OpenVersion2` struct uses `bool`s to refer to `GetInfoOnly` and `ReadOnly` for ease of use. However,
// the internal windows structure uses `BOOL`s aka int32s for these types. `openVersion2` is used for translating
// `OpenVersion2` fields to the correct windows internal field types on the `Open____` methods.
type openVersion2 struct {
	getInfoOnly    int32
	readOnly       int32
	resiliencyGUID guid.GUID
}

type openVirtualDiskParameters struct {
	version  uint32
	version2 openVersion2
}

type AttachVersion2 struct {
	RestrictedOffset uint64
	RestrictedLength uint64
}

type AttachVirtualDiskParameters struct {
	Version  uint32
	Version2 AttachVersion2
}

const (
	//revive:disable-next-line:var-naming ALL_CAPS
	VIRTUAL_STORAGE_TYPE_DEVICE_VHDX = 0x3

	// Access Mask for opening a VHD.
	VirtualDiskAccessNone     VirtualDiskAccessMask = 0x00000000
	VirtualDiskAccessAttachRO VirtualDiskAccessMask = 0x00010000
	VirtualDiskAccessAttachRW VirtualDiskAccessMask = 0x00020000
	VirtualDiskAccessDetach   VirtualDiskAccessMask = 0x00040000
	VirtualDiskAccessGetInfo  VirtualDiskAccessMask = 0x00080000
	VirtualDiskAccessCreate   VirtualDiskAccessMask = 0x00100000
	VirtualDiskAccessMetaOps  VirtualDiskAccessMask = 0x00200000
	VirtualDiskAccessRead     VirtualDiskAccessMask = 0x000d0000
	VirtualDiskAccessAll      VirtualDiskAccessMask = 0x003f0000
	VirtualDiskAccessWritable VirtualDiskAccessMask = 0x00320000

	// Flags for creating a VHD.
	CreateVirtualDiskFlagNone                              CreateVirtualDiskFlag = 0x0
	CreateVirtualDiskFlagFullPhysicalAllocation            CreateVirtualDiskFlag = 0x1
	CreateVirtualDiskFlagPreventWritesToSourceDisk         CreateVirtualDiskFlag = 0x2
	CreateVirtualDiskFlagDoNotCopyMetadataFromParent       CreateVirtualDiskFlag = 0x4
	CreateVirtualDiskFlagCreateBackingStorage              CreateVirtualDiskFlag = 0x8
	CreateVirtualDiskFlagUseChangeTrackingSourceLimit      CreateVirtualDiskFlag = 0x10
	CreateVirtualDiskFlagPreserveParentChangeTrackingState CreateVirtualDiskFlag = 0x20
	CreateVirtualDiskFlagVhdSetUseOriginalBackingStorage   CreateVirtualDiskFlag = 0x40 //revive:disable-line:var-naming VHD, not Vhd
	CreateVirtualDiskFlagSparseFile                        CreateVirtualDiskFlag = 0x80
	CreateVirtualDiskFlagPmemCompatible                    CreateVirtualDiskFlag = 0x100 //revive:disable-line:var-naming PMEM, not Pmem
	CreateVirtualDiskFlagSupportCompressedVolumes          CreateVirtualDiskFlag = 0x200

	// Flags for opening a VHD.
	OpenVirtualDiskFlagNone                        VirtualDiskFlag = 0x00000000
	OpenVirtualDiskFlagNoParents                   VirtualDiskFlag = 0x00000001
	OpenVirtualDiskFlagBlankFile                   VirtualDiskFlag = 0x00000002
	OpenVirtualDiskFlagBootDrive                   VirtualDiskFlag = 0x00000004
	OpenVirtualDiskFlagCachedIO                    VirtualDiskFlag = 0x00000008
	OpenVirtualDiskFlagCustomDiffChain             VirtualDiskFlag = 0x00000010
	OpenVirtualDiskFlagParentCachedIO              VirtualDiskFlag = 0x00000020
	OpenVirtualDiskFlagVhdsetFileOnly              VirtualDiskFlag = 0x00000040
	OpenVirtualDiskFlagIgnoreRelativeParentLocator VirtualDiskFlag = 0x00000080
	OpenVirtualDiskFlagNoWriteHardening            VirtualDiskFlag = 0x00000100
	OpenVirtualDiskFlagSupportCompressedVolumes    VirtualDiskFlag = 0x00000200

	// Flags for attaching a VHD.
	AttachVirtualDiskFlagNone                          AttachVirtualDiskFlag = 0x00000000
	AttachVirtualDiskFlagReadOnly                      AttachVirtualDiskFlag = 0x00000001
	AttachVirtualDiskFlagNoDriveLetter                 AttachVirtualDiskFlag = 0x00000002
	AttachVirtualDiskFlagPermanentLifetime             AttachVirtualDiskFlag = 0x00000004
	AttachVirtualDiskFlagNoLocalHost                   AttachVirtualDiskFlag = 0x00000008
	AttachVirtualDiskFlagNoSecurityDescriptor          AttachVirtualDiskFlag = 0x00000010
	AttachVirtualDiskFlagBypassDefaultEncryptionPolicy AttachVirtualDiskFlag = 0x00000020
	AttachVirtualDiskFlagNonPnp                        AttachVirtualDiskFlag = 0x00000040
	AttachVirtualDiskFlagRestrictedRange               AttachVirtualDiskFlag = 0x00000080
	AttachVirtualDiskFlagSinglePartition               AttachVirtualDiskFlag = 0x00000100
	AttachVirtualDiskFlagRegisterVolume                AttachVirtualDiskFlag = 0x00000200

	// Flags for detaching a VHD.
	DetachVirtualDiskFlagNone DetachVirtualDiskFlag = 0x0
)

// CreateVhdx is a helper function to create a simple vhdx file at the given path using
// default values.
//
//revive:disable-next-line:var-naming VHDX, not Vhdx
func CreateVhdx(path string, maxSizeInGb, blockSizeInMb uint32) error {
	params := CreateVirtualDiskParameters{
		Version: 2,
		Version2: CreateVersion2{
			MaximumSize:      uint64(maxSizeInGb) * 1024 * 1024 * 1024,
			BlockSizeInBytes: blockSizeInMb * 1024 * 1024,
		},
	}

	handle, err := CreateVirtualDisk(path, VirtualDiskAccessNone, CreateVirtualDiskFlagNone, &params)
	if err != nil {
		return err
	}

	return syscall.CloseHandle(handle)
}

// DetachVirtualDisk detaches a virtual hard disk by handle.
func DetachVirtualDisk(handle syscall.Handle) (err error) {
	if err := detachVirtualDisk(handle, 0, 0); err != nil {
		return fmt.Errorf("failed to detach virtual disk: %w", err)
	}
	return nil
}

// DetachVhd detaches a vhd found at `path`.
//
//revive:disable-next-line:var-naming VHD, not Vhd
func DetachVhd(path string) error {
	handle, err := OpenVirtualDisk(
		path,
		VirtualDiskAccessNone,
		OpenVirtualDiskFlagCachedIO|OpenVirtualDiskFlagIgnoreRelativeParentLocator,
	)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(handle) //nolint:errcheck
	return DetachVirtualDisk(handle)
}

// AttachVirtualDisk attaches a virtual hard disk for use.
func AttachVirtualDisk(
	handle syscall.Handle,
	attachVirtualDiskFlag AttachVirtualDiskFlag,
	parameters *AttachVirtualDiskParameters,
) (err error) {
	// Supports both version 1 and 2 of the attach parameters as version 2 wasn't present in RS5.
	if err := attachVirtualDisk(
		handle,
		nil,
		uint32(attachVirtualDiskFlag),
		0,
		parameters,
		nil,
	); err != nil {
		return fmt.Errorf("failed to attach virtual disk: %w", err)
	}
	return nil
}

// AttachVhd attaches a virtual hard disk at `path` for use. Attaches using version 2
// of the ATTACH_VIRTUAL_DISK_PARAMETERS.
//
//revive:disable-next-line:var-naming VHD, not Vhd
func AttachVhd(path string) (err error) {
	handle, err := OpenVirtualDisk(
		path,
		VirtualDiskAccessNone,
		OpenVirtualDiskFlagCachedIO|OpenVirtualDiskFlagIgnoreRelativeParentLocator,
	)
	if err != nil {
		return err
	}

	defer syscall.CloseHandle(handle) //nolint:errcheck
	params := AttachVirtualDiskParameters{Version: 2}
	if err := AttachVirtualDisk(
		handle,
		AttachVirtualDiskFlagNone,
		&params,
	); err != nil {
		return fmt.Errorf("failed to attach virtual disk: %w", err)
	}
	return nil
}

// OpenVirtualDisk obtains a handle to a VHD opened with supplied access mask and flags.
func OpenVirtualDisk(
	vhdPath string,
	virtualDiskAccessMask VirtualDiskAccessMask,
	openVirtualDiskFlags VirtualDiskFlag,
) (syscall.Handle, error) {
	parameters := OpenVirtualDiskParameters{Version: 2}
	handle, err := OpenVirtualDiskWithParameters(
		vhdPath,
		virtualDiskAccessMask,
		openVirtualDiskFlags,
		&parameters,
	)
	if err != nil {
		return 0, err
	}
	return handle, nil
}

// OpenVirtualDiskWithParameters obtains a handle to a VHD opened with supplied access mask, flags and parameters.
func OpenVirtualDiskWithParameters(
	vhdPath string,
	virtualDiskAccessMask VirtualDiskAccessMask,
	openVirtualDiskFlags VirtualDiskFlag,
	parameters *OpenVirtualDiskParameters,
) (syscall.Handle, error) {
	var (
		handle      syscall.Handle
		defaultType VirtualStorageType
		getInfoOnly int32
		readOnly    int32
	)
	if parameters.Version != 2 {
		return handle, fmt.Errorf("only version 2 VHDs are supported, found version: %d", parameters.Version)
	}
	if parameters.Version2.GetInfoOnly {
		getInfoOnly = 1
	}
	if parameters.Version2.ReadOnly {
		readOnly = 1
	}
	params := &openVirtualDiskParameters{
		version: parameters.Version,
		version2: openVersion2{
			getInfoOnly,
			readOnly,
			parameters.Version2.ResiliencyGUID,
		},
	}
	if err := openVirtualDisk(
		&defaultType,
		vhdPath,
		uint32(virtualDiskAccessMask),
		uint32(openVirtualDiskFlags),
		params,
		&handle,
	); err != nil {
		return 0, fmt.Errorf("failed to open virtual disk: %w", err)
	}
	return handle, nil
}

// CreateVirtualDisk creates a virtual harddisk and returns a handle to the disk.
func CreateVirtualDisk(
	path string,
	virtualDiskAccessMask VirtualDiskAccessMask,
	createVirtualDiskFlags CreateVirtualDiskFlag,
	parameters *CreateVirtualDiskParameters,
) (syscall.Handle, error) {
	var (
		handle      syscall.Handle
		defaultType VirtualStorageType
	)
	if parameters.Version != 2 {
		return handle, fmt.Errorf("only version 2 VHDs are supported, found version: %d", parameters.Version)
	}

	if err := createVirtualDisk(
		&defaultType,
		path,
		uint32(virtualDiskAccessMask),
		nil,
		uint32(createVirtualDiskFlags),
		0,
		parameters,
		nil,
		&handle,
	); err != nil {
		return handle, fmt.Errorf("failed to create virtual disk: %w", err)
	}
	return handle, nil
}

// GetVirtualDiskPhysicalPath takes a handle to a virtual hard disk and returns the physical
// path of the disk on the machine. This path is in the form \\.\PhysicalDriveX where X is an integer
// that represents the particular enumeration of the physical disk on the caller's system.
func GetVirtualDiskPhysicalPath(handle syscall.Handle) (_ string, err error) {
	var (
		diskPathSizeInBytes uint32 = 256 * 2 // max path length 256 wide chars
		diskPhysicalPathBuf [256]uint16
	)
	if err := getVirtualDiskPhysicalPath(
		handle,
		&diskPathSizeInBytes,
		&diskPhysicalPathBuf[0],
	); err != nil {
		return "", fmt.Errorf("failed to get disk physical path: %w", err)
	}
	return windows.UTF16ToString(diskPhysicalPathBuf[:]), nil
}

// CreateDiffVhd is a helper function to create a differencing virtual disk.
//
//revive:disable-next-line:var-naming VHD, not Vhd
func CreateDiffVhd(diffVhdPath, baseVhdPath string, blockSizeInMB uint32) error {
	// Setting `ParentPath` is how to signal to create a differencing disk.
	createParams := &CreateVirtualDiskParameters{
		Version: 2,
		Version2: CreateVersion2{
			ParentPath:       windows.StringToUTF16Ptr(baseVhdPath),
			BlockSizeInBytes: blockSizeInMB * 1024 * 1024,
			OpenFlags:        uint32(OpenVirtualDiskFlagCachedIO),
		},
	}

	vhdHandle, err := CreateVirtualDisk(
		diffVhdPath,
		VirtualDiskAccessNone,
		CreateVirtualDiskFlagNone,
		createParams,
	)
	if err != nil {
		return fmt.Errorf("failed to create differencing vhd: %w", err)
	}
	if err := syscall.CloseHandle(vhdHandle); err != nil {
		return fmt.Errorf("failed to close differencing vhd handle: %w", err)
	}
	return nil
}

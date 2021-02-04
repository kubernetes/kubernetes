// +build windows

package vhd

import (
	"fmt"
	"syscall"

	"github.com/Microsoft/go-winio/pkg/guid"
	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

//go:generate go run mksyscall_windows.go -output zvhd.go vhd.go

//sys createVirtualDisk(virtualStorageType *VirtualStorageType, path string, virtualDiskAccessMask uint32, securityDescriptor *uintptr, createVirtualDiskFlags uint32, providerSpecificFlags uint32, parameters *CreateVirtualDiskParameters, overlapped *syscall.Overlapped, handle *syscall.Handle) (err error) [failretval != 0] = virtdisk.CreateVirtualDisk
//sys openVirtualDisk(virtualStorageType *VirtualStorageType, path string, virtualDiskAccessMask uint32, openVirtualDiskFlags uint32, parameters *OpenVirtualDiskParameters, handle *syscall.Handle) (err error) [failretval != 0] = virtdisk.OpenVirtualDisk
//sys attachVirtualDisk(handle syscall.Handle, securityDescriptor *uintptr, attachVirtualDiskFlag uint32, providerSpecificFlags uint32, parameters *AttachVirtualDiskParameters, overlapped *syscall.Overlapped) (err error) [failretval != 0] = virtdisk.AttachVirtualDisk
//sys detachVirtualDisk(handle syscall.Handle, detachVirtualDiskFlags uint32, providerSpecificFlags uint32) (err error) [failretval != 0] = virtdisk.DetachVirtualDisk
//sys getVirtualDiskPhysicalPath(handle syscall.Handle, diskPathSizeInBytes *uint32, buffer *uint16) (err error) [failretval != 0] = virtdisk.GetVirtualDiskPhysicalPath

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

type AttachVersion2 struct {
	RestrictedOffset uint64
	RestrictedLength uint64
}

type AttachVirtualDiskParameters struct {
	Version  uint32 // Must always be set to 2
	Version2 AttachVersion2
}

const (
	VIRTUAL_STORAGE_TYPE_DEVICE_VHDX = 0x3

	// Access Mask for opening a VHD
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

	// Flags for creating a VHD
	CreateVirtualDiskFlagNone                              CreateVirtualDiskFlag = 0x0
	CreateVirtualDiskFlagFullPhysicalAllocation            CreateVirtualDiskFlag = 0x1
	CreateVirtualDiskFlagPreventWritesToSourceDisk         CreateVirtualDiskFlag = 0x2
	CreateVirtualDiskFlagDoNotCopyMetadataFromParent       CreateVirtualDiskFlag = 0x4
	CreateVirtualDiskFlagCreateBackingStorage              CreateVirtualDiskFlag = 0x8
	CreateVirtualDiskFlagUseChangeTrackingSourceLimit      CreateVirtualDiskFlag = 0x10
	CreateVirtualDiskFlagPreserveParentChangeTrackingState CreateVirtualDiskFlag = 0x20
	CreateVirtualDiskFlagVhdSetUseOriginalBackingStorage   CreateVirtualDiskFlag = 0x40
	CreateVirtualDiskFlagSparseFile                        CreateVirtualDiskFlag = 0x80
	CreateVirtualDiskFlagPmemCompatible                    CreateVirtualDiskFlag = 0x100
	CreateVirtualDiskFlagSupportCompressedVolumes          CreateVirtualDiskFlag = 0x200

	// Flags for opening a VHD
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

	// Flags for attaching a VHD
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

	// Flags for detaching a VHD
	DetachVirtualDiskFlagNone DetachVirtualDiskFlag = 0x0
)

// CreateVhdx is a helper function to create a simple vhdx file at the given path using
// default values.
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

	if err := syscall.CloseHandle(handle); err != nil {
		return err
	}
	return nil
}

// DetachVirtualDisk detaches a virtual hard disk by handle.
func DetachVirtualDisk(handle syscall.Handle) (err error) {
	if err := detachVirtualDisk(handle, 0, 0); err != nil {
		return errors.Wrap(err, "failed to detach virtual disk")
	}
	return nil
}

// DetachVhd detaches a vhd found at `path`.
func DetachVhd(path string) error {
	handle, err := OpenVirtualDisk(
		path,
		VirtualDiskAccessNone,
		OpenVirtualDiskFlagCachedIO|OpenVirtualDiskFlagIgnoreRelativeParentLocator,
	)
	if err != nil {
		return err
	}
	defer syscall.CloseHandle(handle)
	return DetachVirtualDisk(handle)
}

// AttachVirtualDisk attaches a virtual hard disk for use.
func AttachVirtualDisk(handle syscall.Handle, attachVirtualDiskFlag AttachVirtualDiskFlag, parameters *AttachVirtualDiskParameters) (err error) {
	if parameters.Version != 2 {
		return fmt.Errorf("only version 2 VHDs are supported, found version: %d", parameters.Version)
	}
	if err := attachVirtualDisk(
		handle,
		nil,
		uint32(attachVirtualDiskFlag),
		0,
		parameters,
		nil,
	); err != nil {
		return errors.Wrap(err, "failed to attach virtual disk")
	}
	return nil
}

// AttachVhd attaches a virtual hard disk at `path` for use.
func AttachVhd(path string) (err error) {
	handle, err := OpenVirtualDisk(
		path,
		VirtualDiskAccessNone,
		OpenVirtualDiskFlagCachedIO|OpenVirtualDiskFlagIgnoreRelativeParentLocator,
	)
	if err != nil {
		return err
	}

	defer syscall.CloseHandle(handle)
	params := AttachVirtualDiskParameters{Version: 2}
	if err := AttachVirtualDisk(
		handle,
		AttachVirtualDiskFlagNone,
		&params,
	); err != nil {
		return errors.Wrap(err, "failed to attach virtual disk")
	}
	return nil
}

// OpenVirtualDisk obtains a handle to a VHD opened with supplied access mask and flags.
func OpenVirtualDisk(vhdPath string, virtualDiskAccessMask VirtualDiskAccessMask, openVirtualDiskFlags VirtualDiskFlag) (syscall.Handle, error) {
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
func OpenVirtualDiskWithParameters(vhdPath string, virtualDiskAccessMask VirtualDiskAccessMask, openVirtualDiskFlags VirtualDiskFlag, parameters *OpenVirtualDiskParameters) (syscall.Handle, error) {
	var (
		handle      syscall.Handle
		defaultType VirtualStorageType
	)
	if parameters.Version != 2 {
		return handle, fmt.Errorf("only version 2 VHDs are supported, found version: %d", parameters.Version)
	}
	if err := openVirtualDisk(
		&defaultType,
		vhdPath,
		uint32(virtualDiskAccessMask),
		uint32(openVirtualDiskFlags),
		parameters,
		&handle,
	); err != nil {
		return 0, errors.Wrap(err, "failed to open virtual disk")
	}
	return handle, nil
}

// CreateVirtualDisk creates a virtual harddisk and returns a handle to the disk.
func CreateVirtualDisk(path string, virtualDiskAccessMask VirtualDiskAccessMask, createVirtualDiskFlags CreateVirtualDiskFlag, parameters *CreateVirtualDiskParameters) (syscall.Handle, error) {
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
		return handle, errors.Wrap(err, "failed to create virtual disk")
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
		return "", errors.Wrap(err, "failed to get disk physical path")
	}
	return windows.UTF16ToString(diskPhysicalPathBuf[:]), nil
}

// CreateDiffVhd is a helper function to create a differencing virtual disk.
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
		return fmt.Errorf("failed to create differencing vhd: %s", err)
	}
	if err := syscall.CloseHandle(vhdHandle); err != nil {
		return fmt.Errorf("failed to close differencing vhd handle: %s", err)
	}
	return nil
}

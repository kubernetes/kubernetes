// +build windows

package vhd

import "syscall"

//go:generate go run mksyscall_windows.go -output zvhd.go vhd.go

//sys createVirtualDisk(virtualStorageType *virtualStorageType, path string, virtualDiskAccessMask uint32, securityDescriptor *uintptr, flags uint32, providerSpecificFlags uint32, parameters *createVirtualDiskParameters, o *syscall.Overlapped, handle *syscall.Handle) (err error) [failretval != 0] = VirtDisk.CreateVirtualDisk

type virtualStorageType struct {
	DeviceID uint32
	VendorID [16]byte
}

const virtualDiskAccessNONE uint32 = 0
const virtualDiskAccessATTACHRO uint32 = 65536
const virtualDiskAccessATTACHRW uint32 = 131072
const virtualDiskAccessDETACH uint32 = 262144
const virtualDiskAccessGETINFO uint32 = 524288
const virtualDiskAccessCREATE uint32 = 1048576
const virtualDiskAccessMETAOPS uint32 = 2097152
const virtualDiskAccessREAD uint32 = 851968
const virtualDiskAccessALL uint32 = 4128768
const virtualDiskAccessWRITABLE uint32 = 3276800

const createVirtualDiskFlagNone uint32 = 0
const createVirtualDiskFlagFullPhysicalAllocation uint32 = 1
const createVirtualDiskFlagPreventWritesToSourceDisk uint32 = 2
const createVirtualDiskFlagDoNotCopyMetadataFromParent uint32 = 4

type version2 struct {
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
	Version2 version2
}

// CreateVhdx will create a simple vhdx file at the given path using default values.
func CreateVhdx(path string, maxSizeInGb, blockSizeInMb uint32) error {
	var defaultType virtualStorageType

	parameters := createVirtualDiskParameters{
		Version: 2,
		Version2: version2{
			MaximumSize:      uint64(maxSizeInGb) * 1024 * 1024 * 1024,
			BlockSizeInBytes: blockSizeInMb * 1024 * 1024,
		},
	}

	var handle syscall.Handle

	if err := createVirtualDisk(
		&defaultType,
		path,
		virtualDiskAccessNONE,
		nil,
		createVirtualDiskFlagNone,
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

//go:build windows
// +build windows

package winio

import (
	"os"
	"runtime"
	"unsafe"

	"golang.org/x/sys/windows"
)

// FileBasicInfo contains file access time and file attributes information.
type FileBasicInfo struct {
	CreationTime, LastAccessTime, LastWriteTime, ChangeTime windows.Filetime
	FileAttributes                                          uint32
	_                                                       uint32 // padding
}

// alignedFileBasicInfo is a FileBasicInfo, but aligned to uint64 by containing
// uint64 rather than windows.Filetime. Filetime contains two uint32s. uint64
// alignment is necessary to pass this as FILE_BASIC_INFO.
type alignedFileBasicInfo struct {
	CreationTime, LastAccessTime, LastWriteTime, ChangeTime uint64
	FileAttributes                                          uint32
	_                                                       uint32 // padding
}

// GetFileBasicInfo retrieves times and attributes for a file.
func GetFileBasicInfo(f *os.File) (*FileBasicInfo, error) {
	bi := &alignedFileBasicInfo{}
	if err := windows.GetFileInformationByHandleEx(
		windows.Handle(f.Fd()),
		windows.FileBasicInfo,
		(*byte)(unsafe.Pointer(bi)),
		uint32(unsafe.Sizeof(*bi)),
	); err != nil {
		return nil, &os.PathError{Op: "GetFileInformationByHandleEx", Path: f.Name(), Err: err}
	}
	runtime.KeepAlive(f)
	// Reinterpret the alignedFileBasicInfo as a FileBasicInfo so it matches the
	// public API of this module. The data may be unnecessarily aligned.
	return (*FileBasicInfo)(unsafe.Pointer(bi)), nil
}

// SetFileBasicInfo sets times and attributes for a file.
func SetFileBasicInfo(f *os.File, bi *FileBasicInfo) error {
	// Create an alignedFileBasicInfo based on a FileBasicInfo. The copy is
	// suitable to pass to GetFileInformationByHandleEx.
	biAligned := *(*alignedFileBasicInfo)(unsafe.Pointer(bi))
	if err := windows.SetFileInformationByHandle(
		windows.Handle(f.Fd()),
		windows.FileBasicInfo,
		(*byte)(unsafe.Pointer(&biAligned)),
		uint32(unsafe.Sizeof(biAligned)),
	); err != nil {
		return &os.PathError{Op: "SetFileInformationByHandle", Path: f.Name(), Err: err}
	}
	runtime.KeepAlive(f)
	return nil
}

// FileStandardInfo contains extended information for the file.
// FILE_STANDARD_INFO in WinBase.h
// https://docs.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-file_standard_info
type FileStandardInfo struct {
	AllocationSize, EndOfFile int64
	NumberOfLinks             uint32
	DeletePending, Directory  bool
}

// GetFileStandardInfo retrieves ended information for the file.
func GetFileStandardInfo(f *os.File) (*FileStandardInfo, error) {
	si := &FileStandardInfo{}
	if err := windows.GetFileInformationByHandleEx(windows.Handle(f.Fd()),
		windows.FileStandardInfo,
		(*byte)(unsafe.Pointer(si)),
		uint32(unsafe.Sizeof(*si))); err != nil {
		return nil, &os.PathError{Op: "GetFileInformationByHandleEx", Path: f.Name(), Err: err}
	}
	runtime.KeepAlive(f)
	return si, nil
}

// FileIDInfo contains the volume serial number and file ID for a file. This pair should be
// unique on a system.
type FileIDInfo struct {
	VolumeSerialNumber uint64
	FileID             [16]byte
}

// GetFileID retrieves the unique (volume, file ID) pair for a file.
func GetFileID(f *os.File) (*FileIDInfo, error) {
	fileID := &FileIDInfo{}
	if err := windows.GetFileInformationByHandleEx(
		windows.Handle(f.Fd()),
		windows.FileIdInfo,
		(*byte)(unsafe.Pointer(fileID)),
		uint32(unsafe.Sizeof(*fileID)),
	); err != nil {
		return nil, &os.PathError{Op: "GetFileInformationByHandleEx", Path: f.Name(), Err: err}
	}
	runtime.KeepAlive(f)
	return fileID, nil
}

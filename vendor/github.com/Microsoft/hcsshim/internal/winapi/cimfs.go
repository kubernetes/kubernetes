//go:build windows

package winapi

import (
	"unsafe"

	"github.com/Microsoft/go-winio/pkg/guid"
	"golang.org/x/sys/windows"
)

type g = guid.GUID
type FsHandle uintptr
type StreamHandle uintptr

type CimFsFileMetadata struct {
	Attributes uint32
	FileSize   int64

	CreationTime   windows.Filetime
	LastWriteTime  windows.Filetime
	ChangeTime     windows.Filetime
	LastAccessTime windows.Filetime

	SecurityDescriptorBuffer unsafe.Pointer
	SecurityDescriptorSize   uint32

	ReparseDataBuffer unsafe.Pointer
	ReparseDataSize   uint32

	ExtendedAttributes unsafe.Pointer
	EACount            uint32
}

//sys CimMountImage(imagePath string, fsName string, flags uint32, volumeID *g) (hr error) = cimfs.CimMountImage?
//sys CimDismountImage(volumeID *g) (hr error) = cimfs.CimDismountImage?

//sys CimCreateImage(imagePath string, oldFSName *uint16, newFSName *uint16, cimFSHandle *FsHandle) (hr error) = cimfs.CimCreateImage?
//sys CimCloseImage(cimFSHandle FsHandle) = cimfs.CimCloseImage?
//sys CimCommitImage(cimFSHandle FsHandle) (hr error) = cimfs.CimCommitImage?

//sys CimCreateFile(cimFSHandle FsHandle, path string, file *CimFsFileMetadata, cimStreamHandle *StreamHandle) (hr error) = cimfs.CimCreateFile?
//sys CimCloseStream(cimStreamHandle StreamHandle) (hr error) = cimfs.CimCloseStream?
//sys CimWriteStream(cimStreamHandle StreamHandle, buffer uintptr, bufferSize uint32) (hr error) = cimfs.CimWriteStream?
//sys CimDeletePath(cimFSHandle FsHandle, path string) (hr error) = cimfs.CimDeletePath?
//sys CimCreateHardLink(cimFSHandle FsHandle, newPath string, oldPath string) (hr error) = cimfs.CimCreateHardLink?
//sys CimCreateAlternateStream(cimFSHandle FsHandle, path string, size uint64, cimStreamHandle *StreamHandle) (hr error) = cimfs.CimCreateAlternateStream?

package winio

import (
	"os"
	"syscall"
	"unsafe"
)

//sys getFileInformationByHandleEx(h syscall.Handle, class uint32, buffer *byte, size uint32) (err error) = GetFileInformationByHandleEx
//sys setFileInformationByHandle(h syscall.Handle, class uint32, buffer *byte, size uint32) (err error) = SetFileInformationByHandle

type FileBasicInfo struct {
	CreationTime, LastAccessTime, LastWriteTime, ChangeTime syscall.Filetime
	FileAttributes                                          uintptr // includes padding
}

func GetFileBasicInfo(f *os.File) (*FileBasicInfo, error) {
	bi := &FileBasicInfo{}
	if err := getFileInformationByHandleEx(syscall.Handle(f.Fd()), 0, (*byte)(unsafe.Pointer(bi)), uint32(unsafe.Sizeof(*bi))); err != nil {
		return nil, &os.PathError{"GetFileInformationByHandleEx", f.Name(), err}
	}
	return bi, nil
}

func SetFileBasicInfo(f *os.File, bi *FileBasicInfo) error {
	if err := setFileInformationByHandle(syscall.Handle(f.Fd()), 0, (*byte)(unsafe.Pointer(bi)), uint32(unsafe.Sizeof(*bi))); err != nil {
		return &os.PathError{"SetFileInformationByHandle", f.Name(), err}
	}
	return nil
}

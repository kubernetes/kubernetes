// Copyright (c) 2014 The fileutil Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fileutil

import (
	"io"
	"os"
	"sync"
	"syscall"
	"unsafe"
)

// PunchHole deallocates space inside a file in the byte range starting at
// offset and continuing for len bytes. Not supported on Windows.
func PunchHole(f *os.File, off, len int64) error {
	return puncher(f, off, len)
}

// Fadvise predeclares an access pattern for file data.  See also 'man 2
// posix_fadvise'. Not supported on Windows.
func Fadvise(f *os.File, off, len int64, advice FadviseAdvice) error {
	return nil
}

// IsEOF reports whether err is an EOF condition.
func IsEOF(err error) bool {
	if err == io.EOF {
		return true
	}

	// http://social.technet.microsoft.com/Forums/windowsserver/en-US/1a16311b-c625-46cf-830b-6a26af488435/how-to-solve-error-38-0x26-errorhandleeof-using-fsctlgetretrievalpointers
	x, ok := err.(*os.PathError)
	return ok && x.Op == "read" && x.Err.(syscall.Errno) == 0x26
}

var (
	modkernel32 = syscall.NewLazyDLL("kernel32.dll")

	procDeviceIOControl = modkernel32.NewProc("DeviceIoControl")

	sparseFilesMu sync.Mutex
	sparseFiles   map[uintptr]struct{}
)

func init() {
	// sparseFiles is an fd set for already "sparsed" files - according to
	// msdn.microsoft.com/en-us/library/windows/desktop/aa364225(v=vs.85).aspx
	// the file handles are unique per process.
	sparseFiles = make(map[uintptr]struct{})
}

// puncHoleWindows punches a hole into the given file starting at offset,
// measuring "size" bytes
// (http://msdn.microsoft.com/en-us/library/windows/desktop/aa364597%28v=vs.85%29.aspx)
func puncher(file *os.File, offset, size int64) error {
	if err := ensureFileSparse(file); err != nil {
		return err
	}

	// http://msdn.microsoft.com/en-us/library/windows/desktop/aa364411%28v=vs.85%29.aspx
	// typedef struct _FILE_ZERO_DATA_INFORMATION {
	//  LARGE_INTEGER FileOffset;
	//  LARGE_INTEGER BeyondFinalZero;
	//} FILE_ZERO_DATA_INFORMATION, *PFILE_ZERO_DATA_INFORMATION;
	type fileZeroDataInformation struct {
		FileOffset, BeyondFinalZero int64
	}

	lpInBuffer := fileZeroDataInformation{
		FileOffset:      offset,
		BeyondFinalZero: offset + size}
	return deviceIOControl(false, file.Fd(), uintptr(unsafe.Pointer(&lpInBuffer)), 16)
}

// // http://msdn.microsoft.com/en-us/library/windows/desktop/cc948908%28v=vs.85%29.aspx
// type fileSetSparseBuffer struct {
//	 SetSparse bool
// }

func ensureFileSparse(file *os.File) (err error) {
	fd := file.Fd()
	sparseFilesMu.Lock()
	if _, ok := sparseFiles[fd]; ok {
		sparseFilesMu.Unlock()
		return nil
	}

	if err = deviceIOControl(true, fd, 0, 0); err == nil {
		sparseFiles[fd] = struct{}{}
	}
	sparseFilesMu.Unlock()
	return err
}

func deviceIOControl(setSparse bool, fd, inBuf, inBufLen uintptr) (err error) {
	const (
		//http://source.winehq.org/source/include/winnt.h#L4605
		file_read_data  = 1
		file_write_data = 2

		// METHOD_BUFFERED	0
		method_buffered = 0
		// FILE_ANY_ACCESS   0
		file_any_access = 0
		// FILE_DEVICE_FILE_SYSTEM   0x00000009
		file_device_file_system = 0x00000009
		// FILE_SPECIAL_ACCESS   (FILE_ANY_ACCESS)
		file_special_access = file_any_access
		file_read_access    = file_read_data
		file_write_access   = file_write_data

		// http://source.winehq.org/source/include/winioctl.h
		// #define CTL_CODE 	(  	DeviceType,
		//		Function,
		//		Method,
		//		Access  		 )
		//    ((DeviceType) << 16) | ((Access) << 14) | ((Function) << 2) | (Method)

		// FSCTL_SET_COMPRESSION   CTL_CODE(FILE_DEVICE_FILE_SYSTEM, 16, METHOD_BUFFERED, FILE_READ_DATA | FILE_WRITE_DATA)
		fsctl_set_compression = (file_device_file_system << 16) | ((file_read_access | file_write_access) << 14) | (16 << 2) | method_buffered
		// FSCTL_SET_SPARSE   CTL_CODE(FILE_DEVICE_FILE_SYSTEM, 49, METHOD_BUFFERED, FILE_SPECIAL_ACCESS)
		fsctl_set_sparse = (file_device_file_system << 16) | (file_special_access << 14) | (49 << 2) | method_buffered
		// FSCTL_SET_ZERO_DATA   CTL_CODE(FILE_DEVICE_FILE_SYSTEM, 50, METHOD_BUFFERED, FILE_WRITE_DATA)
		fsctl_set_zero_data = (file_device_file_system << 16) | (file_write_data << 14) | (50 << 2) | method_buffered
	)
	retPtr := uintptr(unsafe.Pointer(&(make([]byte, 8)[0])))
	var r1 uintptr
	var e1 syscall.Errno
	if setSparse {
		// BOOL
		// WINAPI
		// DeviceIoControl( (HANDLE) hDevice,                      // handle to a file
		//                  FSCTL_SET_SPARSE,                      // dwIoControlCode
		//                  (PFILE_SET_SPARSE_BUFFER) lpInBuffer,  // input buffer
		//                  (DWORD) nInBufferSize,                 // size of input buffer
		//                  NULL,                                  // lpOutBuffer
		//                  0,                                     // nOutBufferSize
		//                  (LPDWORD) lpBytesReturned,             // number of bytes returned
		//                  (LPOVERLAPPED) lpOverlapped );         // OVERLAPPED structure
		r1, _, e1 = syscall.Syscall9(procDeviceIOControl.Addr(), 8,
			fd,
			uintptr(fsctl_set_sparse),
			// If the lpInBuffer parameter is NULL, the operation will behave the same as if the SetSparse member of the FILE_SET_SPARSE_BUFFER structure were TRUE. In other words, the operation sets the file to a sparse file.
			0, // uintptr(unsafe.Pointer(&lpInBuffer)),
			0, // 1,
			0,
			0,
			retPtr,
			0,
			0)
	} else {
		// BOOL
		// WINAPI
		// DeviceIoControl( (HANDLE) hDevice,              // handle to a file
		//                  FSCTL_SET_ZERO_DATA,           // dwIoControlCode
		//                  (LPVOID) lpInBuffer,           // input buffer
		//                  (DWORD) nInBufferSize,         // size of input buffer
		//                  NULL,                          // lpOutBuffer
		//                  0,                             // nOutBufferSize
		//                  (LPDWORD) lpBytesReturned,     // number of bytes returned
		//                  (LPOVERLAPPED) lpOverlapped ); // OVERLAPPED structure
		r1, _, e1 = syscall.Syscall9(procDeviceIOControl.Addr(), 8,
			fd,
			uintptr(fsctl_set_zero_data),
			inBuf,
			inBufLen,
			0,
			0,
			retPtr,
			0,
			0)
	}
	if r1 == 0 {
		if e1 != 0 {
			err = error(e1)
		} else {
			err = syscall.EINVAL
		}
	}
	return err
}

package safefile

import (
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unicode/utf16"
	"unsafe"

	"github.com/Microsoft/hcsshim/internal/longpath"

	winio "github.com/Microsoft/go-winio"
)

//go:generate go run $GOROOT\src\syscall\mksyscall_windows.go -output zsyscall_windows.go safeopen.go

//sys ntCreateFile(handle *uintptr, accessMask uint32, oa *objectAttributes, iosb *ioStatusBlock, allocationSize *uint64, fileAttributes uint32, shareAccess uint32, createDisposition uint32, createOptions uint32, eaBuffer *byte, eaLength uint32) (status uint32) = ntdll.NtCreateFile
//sys ntSetInformationFile(handle uintptr, iosb *ioStatusBlock, information uintptr, length uint32, class uint32) (status uint32) = ntdll.NtSetInformationFile
//sys rtlNtStatusToDosError(status uint32) (winerr error) = ntdll.RtlNtStatusToDosErrorNoTeb
//sys localAlloc(flags uint32, size int) (ptr uintptr) = kernel32.LocalAlloc
//sys localFree(ptr uintptr) = kernel32.LocalFree

type ioStatusBlock struct {
	Status, Information uintptr
}

type objectAttributes struct {
	Length             uintptr
	RootDirectory      uintptr
	ObjectName         uintptr
	Attributes         uintptr
	SecurityDescriptor uintptr
	SecurityQoS        uintptr
}

type unicodeString struct {
	Length        uint16
	MaximumLength uint16
	Buffer        uintptr
}

type fileLinkInformation struct {
	ReplaceIfExists bool
	RootDirectory   uintptr
	FileNameLength  uint32
	FileName        [1]uint16
}

type fileDispositionInformationEx struct {
	Flags uintptr
}

const (
	_FileLinkInformation          = 11
	_FileDispositionInformationEx = 64

	FILE_READ_ATTRIBUTES  = 0x0080
	FILE_WRITE_ATTRIBUTES = 0x0100
	DELETE                = 0x10000

	FILE_OPEN   = 1
	FILE_CREATE = 2

	FILE_DIRECTORY_FILE          = 0x00000001
	FILE_SYNCHRONOUS_IO_NONALERT = 0x00000020
	FILE_DELETE_ON_CLOSE         = 0x00001000
	FILE_OPEN_FOR_BACKUP_INTENT  = 0x00004000
	FILE_OPEN_REPARSE_POINT      = 0x00200000

	FILE_DISPOSITION_DELETE = 0x00000001

	_OBJ_DONT_REPARSE = 0x1000

	_STATUS_REPARSE_POINT_ENCOUNTERED = 0xC000050B
)

func OpenRoot(path string) (*os.File, error) {
	longpath, err := longpath.LongAbs(path)
	if err != nil {
		return nil, err
	}
	return winio.OpenForBackup(longpath, syscall.GENERIC_READ, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, syscall.OPEN_EXISTING)
}

func ntRelativePath(path string) ([]uint16, error) {
	path = filepath.Clean(path)
	if strings.Contains(path, ":") {
		// Since alternate data streams must follow the file they
		// are attached to, finding one here (out of order) is invalid.
		return nil, errors.New("path contains invalid character `:`")
	}
	fspath := filepath.FromSlash(path)
	if len(fspath) > 0 && fspath[0] == '\\' {
		return nil, errors.New("expected relative path")
	}

	path16 := utf16.Encode(([]rune)(fspath))
	if len(path16) > 32767 {
		return nil, syscall.ENAMETOOLONG
	}

	return path16, nil
}

// openRelativeInternal opens a relative path from the given root, failing if
// any of the intermediate path components are reparse points.
func openRelativeInternal(path string, root *os.File, accessMask uint32, shareFlags uint32, createDisposition uint32, flags uint32) (*os.File, error) {
	var (
		h    uintptr
		iosb ioStatusBlock
		oa   objectAttributes
	)

	path16, err := ntRelativePath(path)
	if err != nil {
		return nil, err
	}

	if root == nil || root.Fd() == 0 {
		return nil, errors.New("missing root directory")
	}

	upathBuffer := localAlloc(0, int(unsafe.Sizeof(unicodeString{}))+len(path16)*2)
	defer localFree(upathBuffer)

	upath := (*unicodeString)(unsafe.Pointer(upathBuffer))
	upath.Length = uint16(len(path16) * 2)
	upath.MaximumLength = upath.Length
	upath.Buffer = upathBuffer + unsafe.Sizeof(*upath)
	copy((*[32768]uint16)(unsafe.Pointer(upath.Buffer))[:], path16)

	oa.Length = unsafe.Sizeof(oa)
	oa.ObjectName = upathBuffer
	oa.RootDirectory = uintptr(root.Fd())
	oa.Attributes = _OBJ_DONT_REPARSE
	status := ntCreateFile(
		&h,
		accessMask|syscall.SYNCHRONIZE,
		&oa,
		&iosb,
		nil,
		0,
		shareFlags,
		createDisposition,
		FILE_OPEN_FOR_BACKUP_INTENT|FILE_SYNCHRONOUS_IO_NONALERT|flags,
		nil,
		0,
	)
	if status != 0 {
		return nil, rtlNtStatusToDosError(status)
	}

	fullPath, err := longpath.LongAbs(filepath.Join(root.Name(), path))
	if err != nil {
		syscall.Close(syscall.Handle(h))
		return nil, err
	}

	return os.NewFile(h, fullPath), nil
}

// OpenRelative opens a relative path from the given root, failing if
// any of the intermediate path components are reparse points.
func OpenRelative(path string, root *os.File, accessMask uint32, shareFlags uint32, createDisposition uint32, flags uint32) (*os.File, error) {
	f, err := openRelativeInternal(path, root, accessMask, shareFlags, createDisposition, flags)
	if err != nil {
		err = &os.PathError{Op: "open", Path: filepath.Join(root.Name(), path), Err: err}
	}
	return f, err
}

// LinkRelative creates a hard link from oldname to newname (relative to oldroot
// and newroot), failing if any of the intermediate path components are reparse
// points.
func LinkRelative(oldname string, oldroot *os.File, newname string, newroot *os.File) error {
	// Open the old file.
	oldf, err := openRelativeInternal(
		oldname,
		oldroot,
		syscall.FILE_WRITE_ATTRIBUTES,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_OPEN,
		0,
	)
	if err != nil {
		return &os.LinkError{Op: "link", Old: filepath.Join(oldroot.Name(), oldname), New: filepath.Join(newroot.Name(), newname), Err: err}
	}
	defer oldf.Close()

	// Open the parent of the new file.
	var parent *os.File
	parentPath := filepath.Dir(newname)
	if parentPath != "." {
		parent, err = openRelativeInternal(
			parentPath,
			newroot,
			syscall.GENERIC_READ,
			syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
			FILE_OPEN,
			FILE_DIRECTORY_FILE)
		if err != nil {
			return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(newroot.Name(), newname), Err: err}
		}
		defer parent.Close()

		fi, err := winio.GetFileBasicInfo(parent)
		if err != nil {
			return err
		}
		if (fi.FileAttributes & syscall.FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
			return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(newroot.Name(), newname), Err: rtlNtStatusToDosError(_STATUS_REPARSE_POINT_ENCOUNTERED)}
		}

	} else {
		parent = newroot
	}

	// Issue an NT call to create the link. This will be safe because NT will
	// not open any more directories to create the link, so it cannot walk any
	// more reparse points.
	newbase := filepath.Base(newname)
	newbase16, err := ntRelativePath(newbase)
	if err != nil {
		return err
	}

	size := int(unsafe.Offsetof(fileLinkInformation{}.FileName)) + len(newbase16)*2
	linkinfoBuffer := localAlloc(0, size)
	defer localFree(linkinfoBuffer)
	linkinfo := (*fileLinkInformation)(unsafe.Pointer(linkinfoBuffer))
	linkinfo.RootDirectory = parent.Fd()
	linkinfo.FileNameLength = uint32(len(newbase16) * 2)
	copy((*[32768]uint16)(unsafe.Pointer(&linkinfo.FileName[0]))[:], newbase16)

	var iosb ioStatusBlock
	status := ntSetInformationFile(
		oldf.Fd(),
		&iosb,
		linkinfoBuffer,
		uint32(size),
		_FileLinkInformation,
	)
	if status != 0 {
		return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(parent.Name(), newbase), Err: rtlNtStatusToDosError(status)}
	}

	return nil
}

// deleteOnClose marks a file to be deleted when the handle is closed.
func deleteOnClose(f *os.File) error {
	disposition := fileDispositionInformationEx{Flags: FILE_DISPOSITION_DELETE}
	var iosb ioStatusBlock
	status := ntSetInformationFile(
		f.Fd(),
		&iosb,
		uintptr(unsafe.Pointer(&disposition)),
		uint32(unsafe.Sizeof(disposition)),
		_FileDispositionInformationEx,
	)
	if status != 0 {
		return rtlNtStatusToDosError(status)
	}
	return nil
}

// clearReadOnly clears the readonly attribute on a file.
func clearReadOnly(f *os.File) error {
	bi, err := winio.GetFileBasicInfo(f)
	if err != nil {
		return err
	}
	if bi.FileAttributes&syscall.FILE_ATTRIBUTE_READONLY == 0 {
		return nil
	}
	sbi := winio.FileBasicInfo{
		FileAttributes: bi.FileAttributes &^ syscall.FILE_ATTRIBUTE_READONLY,
	}
	if sbi.FileAttributes == 0 {
		sbi.FileAttributes = syscall.FILE_ATTRIBUTE_NORMAL
	}
	return winio.SetFileBasicInfo(f, &sbi)
}

// RemoveRelative removes a file or directory relative to a root, failing if any
// intermediate path components are reparse points.
func RemoveRelative(path string, root *os.File) error {
	f, err := openRelativeInternal(
		path,
		root,
		FILE_READ_ATTRIBUTES|FILE_WRITE_ATTRIBUTES|DELETE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_OPEN,
		FILE_OPEN_REPARSE_POINT)
	if err == nil {
		defer f.Close()
		err = deleteOnClose(f)
		if err == syscall.ERROR_ACCESS_DENIED {
			// Maybe the file is marked readonly. Clear the bit and retry.
			clearReadOnly(f)
			err = deleteOnClose(f)
		}
	}
	if err != nil {
		return &os.PathError{Op: "remove", Path: filepath.Join(root.Name(), path), Err: err}
	}
	return nil
}

// RemoveAllRelative removes a directory tree relative to a root, failing if any
// intermediate path components are reparse points.
func RemoveAllRelative(path string, root *os.File) error {
	fi, err := LstatRelative(path, root)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}
	fileAttributes := fi.Sys().(*syscall.Win32FileAttributeData).FileAttributes
	if fileAttributes&syscall.FILE_ATTRIBUTE_DIRECTORY == 0 || fileAttributes&syscall.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		// If this is a reparse point, it can't have children. Simple remove will do.
		err := RemoveRelative(path, root)
		if err == nil || os.IsNotExist(err) {
			return nil
		}
		return err
	}

	// It is necessary to use os.Open as Readdirnames does not work with
	// OpenRelative. This is safe because the above lstatrelative fails
	// if the target is outside the root, and we know this is not a
	// symlink from the above FILE_ATTRIBUTE_REPARSE_POINT check.
	fd, err := os.Open(filepath.Join(root.Name(), path))
	if err != nil {
		if os.IsNotExist(err) {
			// Race. It was deleted between the Lstat and Open.
			// Return nil per RemoveAll's docs.
			return nil
		}
		return err
	}

	// Remove contents & return first error.
	for {
		names, err1 := fd.Readdirnames(100)
		for _, name := range names {
			err1 := RemoveAllRelative(path+string(os.PathSeparator)+name, root)
			if err == nil {
				err = err1
			}
		}
		if err1 == io.EOF {
			break
		}
		// If Readdirnames returned an error, use it.
		if err == nil {
			err = err1
		}
		if len(names) == 0 {
			break
		}
	}
	fd.Close()

	// Remove directory.
	err1 := RemoveRelative(path, root)
	if err1 == nil || os.IsNotExist(err1) {
		return nil
	}
	if err == nil {
		err = err1
	}
	return err
}

// MkdirRelative creates a directory relative to a root, failing if any
// intermediate path components are reparse points.
func MkdirRelative(path string, root *os.File) error {
	f, err := openRelativeInternal(
		path,
		root,
		0,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_CREATE,
		FILE_DIRECTORY_FILE)
	if err == nil {
		f.Close()
	} else {
		err = &os.PathError{Op: "mkdir", Path: filepath.Join(root.Name(), path), Err: err}
	}
	return err
}

// LstatRelative performs a stat operation on a file relative to a root, failing
// if any intermediate path components are reparse points.
func LstatRelative(path string, root *os.File) (os.FileInfo, error) {
	f, err := openRelativeInternal(
		path,
		root,
		FILE_READ_ATTRIBUTES,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_OPEN,
		FILE_OPEN_REPARSE_POINT)
	if err != nil {
		return nil, &os.PathError{Op: "stat", Path: filepath.Join(root.Name(), path), Err: err}
	}
	defer f.Close()
	return f.Stat()
}

// EnsureNotReparsePointRelative validates that a given file (relative to a
// root) and all intermediate path components are not a reparse points.
func EnsureNotReparsePointRelative(path string, root *os.File) error {
	// Perform an open with OBJ_DONT_REPARSE but without specifying FILE_OPEN_REPARSE_POINT.
	f, err := OpenRelative(
		path,
		root,
		0,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		FILE_OPEN,
		0)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

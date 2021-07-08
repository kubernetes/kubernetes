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
	"github.com/Microsoft/hcsshim/internal/winapi"

	winio "github.com/Microsoft/go-winio"
)

func OpenRoot(path string) (*os.File, error) {
	longpath, err := longpath.LongAbs(path)
	if err != nil {
		return nil, err
	}
	return winio.OpenForBackup(longpath, syscall.GENERIC_READ, syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE, syscall.OPEN_EXISTING)
}

func cleanGoStringRelativePath(path string) (string, error) {
	path = filepath.Clean(path)
	if strings.Contains(path, ":") {
		// Since alternate data streams must follow the file they
		// are attached to, finding one here (out of order) is invalid.
		return "", errors.New("path contains invalid character `:`")
	}
	fspath := filepath.FromSlash(path)
	if len(fspath) > 0 && fspath[0] == '\\' {
		return "", errors.New("expected relative path")
	}
	return fspath, nil
}

func ntRelativePath(path string) ([]uint16, error) {
	fspath, err := cleanGoStringRelativePath(path)
	if err != nil {
		return nil, err
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
		iosb winapi.IOStatusBlock
		oa   winapi.ObjectAttributes
	)

	cleanRelativePath, err := cleanGoStringRelativePath(path)
	if err != nil {
		return nil, err
	}

	if root == nil || root.Fd() == 0 {
		return nil, errors.New("missing root directory")
	}

	pathUnicode, err := winapi.NewUnicodeString(cleanRelativePath)
	if err != nil {
		return nil, err
	}

	oa.Length = unsafe.Sizeof(oa)
	oa.ObjectName = pathUnicode
	oa.RootDirectory = uintptr(root.Fd())
	oa.Attributes = winapi.OBJ_DONT_REPARSE
	status := winapi.NtCreateFile(
		&h,
		accessMask|syscall.SYNCHRONIZE,
		&oa,
		&iosb,
		nil,
		0,
		shareFlags,
		createDisposition,
		winapi.FILE_OPEN_FOR_BACKUP_INTENT|winapi.FILE_SYNCHRONOUS_IO_NONALERT|flags,
		nil,
		0,
	)
	if status != 0 {
		return nil, winapi.RtlNtStatusToDosError(status)
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
		winapi.FILE_OPEN,
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
			winapi.FILE_OPEN,
			winapi.FILE_DIRECTORY_FILE)
		if err != nil {
			return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(newroot.Name(), newname), Err: err}
		}
		defer parent.Close()

		fi, err := winio.GetFileBasicInfo(parent)
		if err != nil {
			return err
		}
		if (fi.FileAttributes & syscall.FILE_ATTRIBUTE_REPARSE_POINT) != 0 {
			return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(newroot.Name(), newname), Err: winapi.RtlNtStatusToDosError(winapi.STATUS_REPARSE_POINT_ENCOUNTERED)}
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

	size := int(unsafe.Offsetof(winapi.FileLinkInformation{}.FileName)) + len(newbase16)*2
	linkinfoBuffer := winapi.LocalAlloc(0, size)
	defer winapi.LocalFree(linkinfoBuffer)

	linkinfo := (*winapi.FileLinkInformation)(unsafe.Pointer(linkinfoBuffer))
	linkinfo.RootDirectory = parent.Fd()
	linkinfo.FileNameLength = uint32(len(newbase16) * 2)
	copy(winapi.Uint16BufferToSlice(&linkinfo.FileName[0], len(newbase16)), newbase16)

	var iosb winapi.IOStatusBlock
	status := winapi.NtSetInformationFile(
		oldf.Fd(),
		&iosb,
		linkinfoBuffer,
		uint32(size),
		winapi.FileLinkInformationClass,
	)
	if status != 0 {
		return &os.LinkError{Op: "link", Old: oldf.Name(), New: filepath.Join(parent.Name(), newbase), Err: winapi.RtlNtStatusToDosError(status)}
	}

	return nil
}

// deleteOnClose marks a file to be deleted when the handle is closed.
func deleteOnClose(f *os.File) error {
	disposition := winapi.FileDispositionInformationEx{Flags: winapi.FILE_DISPOSITION_DELETE}
	var iosb winapi.IOStatusBlock
	status := winapi.NtSetInformationFile(
		f.Fd(),
		&iosb,
		uintptr(unsafe.Pointer(&disposition)),
		uint32(unsafe.Sizeof(disposition)),
		winapi.FileDispositionInformationExClass,
	)
	if status != 0 {
		return winapi.RtlNtStatusToDosError(status)
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
		winapi.FILE_READ_ATTRIBUTES|winapi.FILE_WRITE_ATTRIBUTES|winapi.DELETE,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		winapi.FILE_OPEN,
		winapi.FILE_OPEN_REPARSE_POINT)
	if err == nil {
		defer f.Close()
		err = deleteOnClose(f)
		if err == syscall.ERROR_ACCESS_DENIED {
			// Maybe the file is marked readonly. Clear the bit and retry.
			_ = clearReadOnly(f)
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
		winapi.FILE_CREATE,
		winapi.FILE_DIRECTORY_FILE)
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
		winapi.FILE_READ_ATTRIBUTES,
		syscall.FILE_SHARE_READ|syscall.FILE_SHARE_WRITE|syscall.FILE_SHARE_DELETE,
		winapi.FILE_OPEN,
		winapi.FILE_OPEN_REPARSE_POINT)
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
		winapi.FILE_OPEN,
		0)
	if err != nil {
		return err
	}
	f.Close()
	return nil
}

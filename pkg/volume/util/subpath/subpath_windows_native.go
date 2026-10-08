//go:build windows

/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package subpath

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"golang.org/x/sys/windows"
)

type windowsLinkType uint8

const (
	windowsNotLink windowsLinkType = iota
	windowsSymbolicLink
	windowsJunction
	windowsHardLink
)

// Reparse points include objects other than the two link types PowerShell
// follows. Inspect the tag rather than relying on Lstat's Go-version-dependent
// classification of junctions.
func nativeLinkType(path string) (windowsLinkType, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return windowsNotLink, &os.PathError{Op: "linktype", Path: path, Err: err}
	}
	attributes, err := windows.GetFileAttributes(p)
	if err != nil {
		return windowsNotLink, &os.PathError{Op: "linktype", Path: path, Err: err}
	}
	if attributes&windows.FILE_ATTRIBUTE_REPARSE_POINT == 0 && attributes&windows.FILE_ATTRIBUTE_DIRECTORY != 0 {
		return windowsNotLink, nil
	}
	handle, err := windows.CreateFile(p, 0, windows.FILE_SHARE_READ|windows.FILE_SHARE_WRITE|windows.FILE_SHARE_DELETE, nil, windows.OPEN_EXISTING, windows.FILE_FLAG_BACKUP_SEMANTICS|windows.FILE_FLAG_OPEN_REPARSE_POINT, 0)
	if err != nil {
		return windowsNotLink, &os.PathError{Op: "linktype", Path: path, Err: err}
	}
	defer windows.CloseHandle(handle)
	if attributes&windows.FILE_ATTRIBUTE_REPARSE_POINT != 0 {
		var info struct{ attributes, tag uint32 }
		err = windows.GetFileInformationByHandleEx(handle, windows.FileAttributeTagInfo, (*byte)(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)))
		if err != nil {
			return windowsNotLink, &os.PathError{Op: "linktype", Path: path, Err: err}
		}
		switch info.tag {
		case windows.IO_REPARSE_TAG_SYMLINK:
			return windowsSymbolicLink, nil
		case windows.IO_REPARSE_TAG_MOUNT_POINT:
			return windowsJunction, nil
		}
		return windowsNotLink, nil
	}
	var info windows.ByHandleFileInformation
	if err := windows.GetFileInformationByHandle(handle, &info); err != nil {
		return windowsNotLink, &os.PathError{Op: "linktype", Path: path, Err: err}
	}
	if info.NumberOfLinks > 1 {
		return windowsHardLink, nil
	}
	return windowsNotLink, nil
}

var (
	linkKernel32       = windows.NewLazySystemDLL("kernel32.dll")
	findFirstFileNameW = linkKernel32.NewProc("FindFirstFileNameW")
	findNextFileNameW  = linkKernel32.NewProc("FindNextFileNameW")
)

// Hardlinks are names of the same file, not redirects. PowerShell's Target
// lists the other names, which may point back to the input and cause resolver
// recursion. Enumerate the names for containment checks without following them.
func hardLinkTargets(path string) ([]string, error) {
	absolute, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	// Enumeration returns long names even when the input uses an 8.3 alias.
	// Compare names in the same form so the input is not its own target.
	absolute, err = longPathName(absolute)
	if err != nil {
		return nil, err
	}
	p, err := windows.UTF16PtrFromString(absolute)
	if err != nil {
		return nil, err
	}
	volume := make([]uint16, MaxPathLength+1)
	if err := windows.GetVolumePathName(p, &volume[0], uint32(len(volume))); err != nil {
		return nil, &os.PathError{Op: "hardlink targets", Path: path, Err: err}
	}
	volumePath := windows.UTF16ToString(volume)
	buffer := make([]uint16, MaxPathLength+1)
	length := uint32(len(buffer))
	handle, _, err := findFirstFileNameW.Call(uintptr(unsafe.Pointer(p)), 0, uintptr(unsafe.Pointer(&length)), uintptr(unsafe.Pointer(&buffer[0])))
	if handle == uintptr(windows.InvalidHandle) {
		return nil, &os.PathError{Op: "hardlink targets", Path: path, Err: err}
	}
	defer windows.FindClose(windows.Handle(handle))
	var targets []string
	for {
		target := filepath.Join(volumePath, strings.TrimPrefix(windows.UTF16ToString(buffer), `\`))
		if !strings.EqualFold(filepath.Clean(target), filepath.Clean(absolute)) {
			targets = append(targets, target)
		}
		length = uint32(len(buffer))
		ok, _, err := findNextFileNameW.Call(handle, uintptr(unsafe.Pointer(&length)), uintptr(unsafe.Pointer(&buffer[0])))
		if ok == 0 {
			if err != windows.ERROR_HANDLE_EOF {
				return nil, &os.PathError{Op: "hardlink targets", Path: path, Err: err}
			}
			break
		}
	}
	return targets, nil
}

func longPathName(path string) (string, error) {
	p, err := windows.UTF16PtrFromString(path)
	if err != nil {
		return "", err
	}
	buffer := make([]uint16, MaxPathLength+1)
	n, err := windows.GetLongPathName(p, &buffer[0], uint32(len(buffer)))
	if err == nil && n >= uint32(len(buffer)) {
		err = windows.ERROR_INSUFFICIENT_BUFFER
	}
	if err != nil {
		return "", &os.PathError{Op: "long path name", Path: path, Err: err}
	}
	return windows.UTF16ToString(buffer[:n]), nil
}

// Called while the caller retains the file's lockPath handle. A hardlink with
// a name outside the volume must not turn native resolution into a bypass of
// the subPath containment check.
func checkHardLinkTargets(path, volumePath string) error {
	// Otherwise a valid long-name alias appears outside an 8.3 volume path.
	volumePath, err := longPathName(volumePath)
	if err != nil {
		return err
	}
	targets, err := hardLinkTargets(path)
	if err != nil {
		return err
	}
	for _, target := range targets {
		resolved, err := evalSymlink(target)
		if err != nil {
			return err
		}
		relative, err := filepath.Rel(volumePath, resolved)
		if err != nil || relative == ".." || strings.HasPrefix(relative, ".."+string(os.PathSeparator)) {
			return fmt.Errorf("hardlink %q has a target %q outside volume %q", path, target, volumePath)
		}
	}
	return nil
}

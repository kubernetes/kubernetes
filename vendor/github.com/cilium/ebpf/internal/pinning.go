package internal

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"

	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

func Pin(currentPath, newPath string, fd *sys.FD) error {
	if newPath == "" {
		return errors.New("given pinning path cannot be empty")
	}
	if currentPath == newPath {
		return nil
	}

	fsType, err := FSType(filepath.Dir(newPath))
	if err != nil {
		return err
	}
	if fsType != unix.BPF_FS_MAGIC {
		return fmt.Errorf("%s is not on a bpf filesystem", newPath)
	}

	defer runtime.KeepAlive(fd)

	if currentPath == "" {
		return sys.ObjPin(&sys.ObjPinAttr{
			Pathname: sys.NewStringPointer(newPath),
			BpfFd:    fd.Uint(),
		})
	}

	// Renameat2 is used instead of os.Rename to disallow the new path replacing
	// an existing path.
	err = unix.Renameat2(unix.AT_FDCWD, currentPath, unix.AT_FDCWD, newPath, unix.RENAME_NOREPLACE)
	if err == nil {
		// Object is now moved to the new pinning path.
		return nil
	}
	if !os.IsNotExist(err) {
		return fmt.Errorf("unable to move pinned object to new path %v: %w", newPath, err)
	}
	// Internal state not in sync with the file system so let's fix it.
	return sys.ObjPin(&sys.ObjPinAttr{
		Pathname: sys.NewStringPointer(newPath),
		BpfFd:    fd.Uint(),
	})
}

func Unpin(pinnedPath string) error {
	if pinnedPath == "" {
		return nil
	}
	err := os.Remove(pinnedPath)
	if err == nil || os.IsNotExist(err) {
		return nil
	}
	return err
}

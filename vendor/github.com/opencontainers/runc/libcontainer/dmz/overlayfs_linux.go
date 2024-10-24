package dmz

import (
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"golang.org/x/sys/unix"

	"github.com/opencontainers/runc/libcontainer/utils"
)

func fsopen(fsName string, flags int) (*os.File, error) {
	// Make sure we always set O_CLOEXEC.
	flags |= unix.FSOPEN_CLOEXEC
	fd, err := unix.Fsopen(fsName, flags)
	if err != nil {
		return nil, os.NewSyscallError("fsopen "+fsName, err)
	}
	return os.NewFile(uintptr(fd), "fscontext:"+fsName), nil
}

func fsmount(ctx *os.File, flags, mountAttrs int) (*os.File, error) {
	// Make sure we always set O_CLOEXEC.
	flags |= unix.FSMOUNT_CLOEXEC
	fd, err := unix.Fsmount(int(ctx.Fd()), flags, mountAttrs)
	if err != nil {
		return nil, os.NewSyscallError("fsmount "+ctx.Name(), err)
	}
	runtime.KeepAlive(ctx) // make sure fd is kept alive while it's used
	return os.NewFile(uintptr(fd), "fsmount:"+ctx.Name()), nil
}

func escapeOverlayLowerDir(path string) string {
	// If the lowerdir path contains ":" we need to escape them, and if there
	// were any escape characters already (\) we need to escape those first.
	return strings.ReplaceAll(strings.ReplaceAll(path, `\`, `\\`), `:`, `\:`)
}

// sealedOverlayfs will create an internal overlayfs mount using fsopen() that
// uses the directory containing the binary as a lowerdir and a temporary tmpfs
// as an upperdir. There is no way to "unwrap" this (unlike MS_BIND+MS_RDONLY)
// and so we can create a safe zero-copy sealed version of /proc/self/exe.
// This only works for privileged users and on kernels with overlayfs and
// fsopen() enabled.
//
// TODO: Since Linux 5.11, overlayfs can be created inside user namespaces so
// it is technically possible to create an overlayfs even for rootless
// containers. Unfortunately, this would require some ugly manual CGo+fork
// magic so we can do this later if we feel it's really needed.
func sealedOverlayfs(binPath, tmpDir string) (_ *os.File, Err error) {
	// Try to do the superblock creation first to bail out early if we can't
	// use this method.
	overlayCtx, err := fsopen("overlay", unix.FSOPEN_CLOEXEC)
	if err != nil {
		return nil, err
	}
	defer overlayCtx.Close()

	// binPath is going to be /proc/self/exe, so do a readlink to get the real
	// path. overlayfs needs the real underlying directory for this protection
	// mode to work properly.
	if realPath, err := os.Readlink(binPath); err == nil {
		binPath = realPath
	}
	binLowerDirPath, binName := filepath.Split(binPath)
	// Escape any ":"s or "\"s in the path.
	binLowerDirPath = escapeOverlayLowerDir(binLowerDirPath)

	// Overlayfs requires two lowerdirs in order to run in "lower-only" mode,
	// where writes are completely blocked. Ideally we would create a dummy
	// tmpfs for this, but it turns out that overlayfs doesn't allow for
	// anonymous mountns paths.
	// NOTE: I'm working on a patch to fix this but it won't be backported.
	dummyLowerDirPath := escapeOverlayLowerDir(tmpDir)

	// Configure the lowerdirs. The binary lowerdir needs to be on the top to
	// ensure that a file called "runc" (binName) in the dummy lowerdir doesn't
	// mask the binary.
	lowerDirStr := binLowerDirPath + ":" + dummyLowerDirPath
	if err := unix.FsconfigSetString(int(overlayCtx.Fd()), "lowerdir", lowerDirStr); err != nil {
		return nil, fmt.Errorf("fsconfig set overlayfs lowerdir=%s: %w", lowerDirStr, err)
	}

	// Get an actual handle to the overlayfs.
	if err := unix.FsconfigCreate(int(overlayCtx.Fd())); err != nil {
		return nil, os.NewSyscallError("fsconfig create overlayfs", err)
	}
	overlayFd, err := fsmount(overlayCtx, unix.FSMOUNT_CLOEXEC, unix.MS_RDONLY|unix.MS_NODEV|unix.MS_NOSUID)
	if err != nil {
		return nil, err
	}
	defer overlayFd.Close()

	// Grab a handle to the binary through overlayfs.
	exeFile, err := utils.Openat(overlayFd, binName, unix.O_PATH|unix.O_NOFOLLOW|unix.O_CLOEXEC, 0)
	if err != nil {
		return nil, fmt.Errorf("open %s from overlayfs (lowerdir=%s): %w", binName, lowerDirStr, err)
	}
	// NOTE: We would like to check that exeFile is the same as /proc/self/exe,
	// except this is a little difficult. Depending on what filesystems the
	// layers are on, overlayfs can remap the inode numbers (and it always
	// creates its own device numbers -- see ovl_map_dev_ino) so we can't do a
	// basic stat-based check. The only reasonable option would be to hash both
	// files and compare them, but this would require fully reading both files
	// which would produce a similar performance overhead to memfd cloning.
	//
	// Ultimately, there isn't a real attack to be worried about here. An
	// attacker would need to be able to modify files in /usr/sbin (or wherever
	// runc lives), at which point they could just replace the runc binary with
	// something malicious anyway.
	return exeFile, nil
}

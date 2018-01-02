// +build linux

package graphdriver

import (
	"path/filepath"

	"github.com/docker/docker/pkg/mount"
	"golang.org/x/sys/unix"
)

const (
	// FsMagicAufs filesystem id for Aufs
	FsMagicAufs = FsMagic(0x61756673)
	// FsMagicBtrfs filesystem id for Btrfs
	FsMagicBtrfs = FsMagic(0x9123683E)
	// FsMagicCramfs filesystem id for Cramfs
	FsMagicCramfs = FsMagic(0x28cd3d45)
	// FsMagicEcryptfs filesystem id for eCryptfs
	FsMagicEcryptfs = FsMagic(0xf15f)
	// FsMagicExtfs filesystem id for Extfs
	FsMagicExtfs = FsMagic(0x0000EF53)
	// FsMagicF2fs filesystem id for F2fs
	FsMagicF2fs = FsMagic(0xF2F52010)
	// FsMagicGPFS filesystem id for GPFS
	FsMagicGPFS = FsMagic(0x47504653)
	// FsMagicJffs2Fs filesystem if for Jffs2Fs
	FsMagicJffs2Fs = FsMagic(0x000072b6)
	// FsMagicJfs filesystem id for Jfs
	FsMagicJfs = FsMagic(0x3153464a)
	// FsMagicNfsFs filesystem id for NfsFs
	FsMagicNfsFs = FsMagic(0x00006969)
	// FsMagicRAMFs filesystem id for RamFs
	FsMagicRAMFs = FsMagic(0x858458f6)
	// FsMagicReiserFs filesystem id for ReiserFs
	FsMagicReiserFs = FsMagic(0x52654973)
	// FsMagicSmbFs filesystem id for SmbFs
	FsMagicSmbFs = FsMagic(0x0000517B)
	// FsMagicSquashFs filesystem id for SquashFs
	FsMagicSquashFs = FsMagic(0x73717368)
	// FsMagicTmpFs filesystem id for TmpFs
	FsMagicTmpFs = FsMagic(0x01021994)
	// FsMagicVxFS filesystem id for VxFs
	FsMagicVxFS = FsMagic(0xa501fcf5)
	// FsMagicXfs filesystem id for Xfs
	FsMagicXfs = FsMagic(0x58465342)
	// FsMagicZfs filesystem id for Zfs
	FsMagicZfs = FsMagic(0x2fc12fc1)
	// FsMagicOverlay filesystem id for overlay
	FsMagicOverlay = FsMagic(0x794C7630)
)

var (
	// Slice of drivers that should be used in an order
	priority = []string{
		"aufs",
		"btrfs",
		"zfs",
		"overlay2",
		"overlay",
		"devicemapper",
		"vfs",
	}

	// FsNames maps filesystem id to name of the filesystem.
	FsNames = map[FsMagic]string{
		FsMagicAufs:        "aufs",
		FsMagicBtrfs:       "btrfs",
		FsMagicCramfs:      "cramfs",
		FsMagicExtfs:       "extfs",
		FsMagicF2fs:        "f2fs",
		FsMagicGPFS:        "gpfs",
		FsMagicJffs2Fs:     "jffs2",
		FsMagicJfs:         "jfs",
		FsMagicNfsFs:       "nfs",
		FsMagicOverlay:     "overlayfs",
		FsMagicRAMFs:       "ramfs",
		FsMagicReiserFs:    "reiserfs",
		FsMagicSmbFs:       "smb",
		FsMagicSquashFs:    "squashfs",
		FsMagicTmpFs:       "tmpfs",
		FsMagicUnsupported: "unsupported",
		FsMagicVxFS:        "vxfs",
		FsMagicXfs:         "xfs",
		FsMagicZfs:         "zfs",
	}
)

// GetFSMagic returns the filesystem id given the path.
func GetFSMagic(rootpath string) (FsMagic, error) {
	var buf unix.Statfs_t
	if err := unix.Statfs(filepath.Dir(rootpath), &buf); err != nil {
		return 0, err
	}
	return FsMagic(buf.Type), nil
}

// NewFsChecker returns a checker configured for the provided FsMagic
func NewFsChecker(t FsMagic) Checker {
	return &fsChecker{
		t: t,
	}
}

type fsChecker struct {
	t FsMagic
}

func (c *fsChecker) IsMounted(path string) bool {
	m, _ := Mounted(c.t, path)
	return m
}

// NewDefaultChecker returns a check that parses /proc/mountinfo to check
// if the specified path is mounted.
func NewDefaultChecker() Checker {
	return &defaultChecker{}
}

type defaultChecker struct {
}

func (c *defaultChecker) IsMounted(path string) bool {
	m, _ := mount.Mounted(path)
	return m
}

// Mounted checks if the given path is mounted as the fs type
func Mounted(fsType FsMagic, mountPath string) (bool, error) {
	var buf unix.Statfs_t
	if err := unix.Statfs(mountPath, &buf); err != nil {
		return false, err
	}
	return FsMagic(buf.Type) == fsType, nil
}

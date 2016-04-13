// +build linux

package graphdriver

import (
	"path/filepath"
	"syscall"
)

const (
	FsMagicAufs     = FsMagic(0x61756673)
	FsMagicBtrfs    = FsMagic(0x9123683E)
	FsMagicCramfs   = FsMagic(0x28cd3d45)
	FsMagicExtfs    = FsMagic(0x0000EF53)
	FsMagicF2fs     = FsMagic(0xF2F52010)
	FsMagicJffs2Fs  = FsMagic(0x000072b6)
	FsMagicJfs      = FsMagic(0x3153464a)
	FsMagicNfsFs    = FsMagic(0x00006969)
	FsMagicRamFs    = FsMagic(0x858458f6)
	FsMagicReiserFs = FsMagic(0x52654973)
	FsMagicSmbFs    = FsMagic(0x0000517B)
	FsMagicSquashFs = FsMagic(0x73717368)
	FsMagicTmpFs    = FsMagic(0x01021994)
	FsMagicXfs      = FsMagic(0x58465342)
	FsMagicZfs      = FsMagic(0x2fc12fc1)
)

var (
	// Slice of drivers that should be used in an order
	priority = []string{
		"aufs",
		"btrfs",
		"zfs",
		"devicemapper",
		"overlay",
		"vfs",
	}

	FsNames = map[FsMagic]string{
		FsMagicAufs:        "aufs",
		FsMagicBtrfs:       "btrfs",
		FsMagicCramfs:      "cramfs",
		FsMagicExtfs:       "extfs",
		FsMagicF2fs:        "f2fs",
		FsMagicJffs2Fs:     "jffs2",
		FsMagicJfs:         "jfs",
		FsMagicNfsFs:       "nfs",
		FsMagicRamFs:       "ramfs",
		FsMagicReiserFs:    "reiserfs",
		FsMagicSmbFs:       "smb",
		FsMagicSquashFs:    "squashfs",
		FsMagicTmpFs:       "tmpfs",
		FsMagicUnsupported: "unsupported",
		FsMagicXfs:         "xfs",
		FsMagicZfs:         "zfs",
	}
)

func GetFSMagic(rootpath string) (FsMagic, error) {
	var buf syscall.Statfs_t
	if err := syscall.Statfs(filepath.Dir(rootpath), &buf); err != nil {
		return 0, err
	}
	return FsMagic(buf.Type), nil
}

// +build !windows

package fs

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/containerd/continuity/sysx"
	"github.com/pkg/errors"
)

// whiteouts are files with a special meaning for the layered filesystem.
// Docker uses AUFS whiteout files inside exported archives. In other
// filesystems these files are generated/handled on tar creation/extraction.

// whiteoutPrefix prefix means file is a whiteout. If this is followed by a
// filename this means that file has been removed from the base layer.
const whiteoutPrefix = ".wh."

// whiteoutMetaPrefix prefix means whiteout has a special meaning and is not
// for removing an actual file. Normally these files are excluded from exported
// archives.
const whiteoutMetaPrefix = whiteoutPrefix + whiteoutPrefix

// whiteoutLinkDir is a directory AUFS uses for storing hardlink links to other
// layers. Normally these should not go into exported archives and all changed
// hardlinks should be copied to the top layer.
const whiteoutLinkDir = whiteoutMetaPrefix + "plnk"

// whiteoutOpaqueDir file means directory has been made opaque - meaning
// readdir calls to this directory do not follow to lower layers.
const whiteoutOpaqueDir = whiteoutMetaPrefix + ".opq"

// detectDirDiff returns diff dir options if a directory could
// be found in the mount info for upper which is the direct
// diff with the provided lower directory
func detectDirDiff(upper, lower string) *diffDirOptions {
	// TODO: get mount options for upper
	// TODO: detect AUFS
	// TODO: detect overlay
	return nil
}

func aufsMetadataSkip(path string) (skip bool, err error) {
	skip, err = filepath.Match(string(os.PathSeparator)+whiteoutMetaPrefix+"*", path)
	if err != nil {
		skip = true
	}
	return
}

func aufsDeletedFile(root, path string, fi os.FileInfo) (string, error) {
	f := filepath.Base(path)

	// If there is a whiteout, then the file was removed
	if strings.HasPrefix(f, whiteoutPrefix) {
		originalFile := f[len(whiteoutPrefix):]
		return filepath.Join(filepath.Dir(path), originalFile), nil
	}

	return "", nil
}

// compareSysStat returns whether the stats are equivalent,
// whether the files are considered the same file, and
// an error
func compareSysStat(s1, s2 interface{}) (bool, error) {
	ls1, ok := s1.(*syscall.Stat_t)
	if !ok {
		return false, nil
	}
	ls2, ok := s2.(*syscall.Stat_t)
	if !ok {
		return false, nil
	}

	return ls1.Mode == ls2.Mode && ls1.Uid == ls2.Uid && ls1.Gid == ls2.Gid && ls1.Rdev == ls2.Rdev, nil
}

func compareCapabilities(p1, p2 string) (bool, error) {
	c1, err := sysx.LGetxattr(p1, "security.capability")
	if err != nil && err != sysx.ENODATA {
		return false, errors.Wrapf(err, "failed to get xattr for %s", p1)
	}
	c2, err := sysx.LGetxattr(p2, "security.capability")
	if err != nil && err != sysx.ENODATA {
		return false, errors.Wrapf(err, "failed to get xattr for %s", p2)
	}
	return bytes.Equal(c1, c2), nil
}

func isLinked(f os.FileInfo) bool {
	s, ok := f.Sys().(*syscall.Stat_t)
	if !ok {
		return false
	}
	return !f.IsDir() && s.Nlink > 1
}

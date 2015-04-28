// Copyright 2014 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package tar contains helper functions for working with tar files
package tar

import (
	"archive/tar"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"time"
)

const DEFAULT_DIR_MODE os.FileMode = 0755

type insecureLinkError error

var ErrNotSupportedPlatform = errors.New("platform and architecture is not supported")

// Map of paths that should be whitelisted. The paths should be relative to the
// root of the tar file and should be cleaned (for example using filepath.Clean)
type PathWhitelistMap map[string]struct{}

// ExtractTar extracts a tarball (from a tar.Reader) into the given directory
// if pwl is not nil, only the paths in the map are extracted.
// If overwrite is true, existing files will be overwritten.
func ExtractTar(tr *tar.Reader, dir string, overwrite bool, pwl PathWhitelistMap) error {
	um := syscall.Umask(0)
	defer syscall.Umask(um)

	dirhdrs := []*tar.Header{}
Tar:
	for {
		hdr, err := tr.Next()
		switch err {
		case io.EOF:
			break Tar
		case nil:
			if pwl != nil {
				relpath := filepath.Clean(hdr.Name)
				if _, ok := pwl[relpath]; !ok {
					continue
				}
			}
			err = ExtractFile(tr, hdr, dir, overwrite)
			if err != nil {
				return fmt.Errorf("error extracting tarball: %v", err)
			}
			if hdr.Typeflag == tar.TypeDir {
				dirhdrs = append(dirhdrs, hdr)
			}
		default:
			return fmt.Errorf("error extracting tarball: %v", err)
		}
	}

	// Restore dirs atime and mtime. This has to be done after extracting
	// as a file extraction will change its parent directory's times.
	for _, hdr := range dirhdrs {
		p := filepath.Join(dir, hdr.Name)
		if err := syscall.UtimesNano(p, HdrToTimespec(hdr)); err != nil {
			return err
		}
	}
	return nil
}

// ExtractFile extracts the file described by hdr from the given tarball into
// the provided directory.
// If overwrite is true, existing files will be overwritten.
func ExtractFile(tr *tar.Reader, hdr *tar.Header, dir string, overwrite bool) error {
	p := filepath.Join(dir, hdr.Name)
	fi := hdr.FileInfo()
	typ := hdr.Typeflag
	if overwrite {
		info, err := os.Lstat(p)
		switch {
		case os.IsNotExist(err):
		case err == nil:
			// If the old and new paths are both dirs do nothing or
			// RemoveAll will remove all dir's contents
			if !info.IsDir() || typ != tar.TypeDir {
				err := os.RemoveAll(p)
				if err != nil {
					return err
				}
			}
		default:
			return err
		}
	}

	// Create parent dir if it doesn't exist
	if err := os.MkdirAll(filepath.Dir(p), DEFAULT_DIR_MODE); err != nil {
		return err
	}
	switch {
	case typ == tar.TypeReg || typ == tar.TypeRegA:
		f, err := os.OpenFile(p, os.O_CREATE|os.O_RDWR, fi.Mode())
		if err != nil {
			return err
		}
		_, err = io.Copy(f, tr)
		if err != nil {
			f.Close()
			return err
		}
		f.Close()
	case typ == tar.TypeDir:
		if err := os.MkdirAll(p, fi.Mode()); err != nil {
			return err
		}
		dir, err := os.Open(p)
		if err != nil {
			return err
		}
		if err := dir.Chmod(fi.Mode()); err != nil {
			dir.Close()
			return err
		}
		dir.Close()
	case typ == tar.TypeLink:
		dest := filepath.Join(dir, hdr.Linkname)
		if !strings.HasPrefix(dest, dir) {
			return insecureLinkError(fmt.Errorf("insecure link %q -> %q", p, hdr.Linkname))
		}
		if err := os.Link(dest, p); err != nil {
			return err
		}
	case typ == tar.TypeSymlink:
		dest := filepath.Join(filepath.Dir(p), hdr.Linkname)
		if !strings.HasPrefix(dest, dir) {
			return insecureLinkError(fmt.Errorf("insecure symlink %q -> %q", p, hdr.Linkname))
		}
		if err := os.Symlink(hdr.Linkname, p); err != nil {
			return err
		}
	case typ == tar.TypeChar:
		dev := makedev(int(hdr.Devmajor), int(hdr.Devminor))
		mode := uint32(fi.Mode()) | syscall.S_IFCHR
		if err := syscall.Mknod(p, mode, dev); err != nil {
			return err
		}
	case typ == tar.TypeBlock:
		dev := makedev(int(hdr.Devmajor), int(hdr.Devminor))
		mode := uint32(fi.Mode()) | syscall.S_IFBLK
		if err := syscall.Mknod(p, mode, dev); err != nil {
			return err
		}
	case typ == tar.TypeFifo:
		if err := syscall.Mkfifo(p, uint32(fi.Mode())); err != nil {
			return err
		}
	// TODO(jonboulle): implement other modes
	default:
		return fmt.Errorf("unsupported type: %v", typ)
	}

	if err := os.Lchown(p, hdr.Uid, hdr.Gid); err != nil {
		return err
	}

	// lchown(2) says that, depending on the linux kernel version, it
	// can change the file's mode also if executed as root. So call
	// os.Chmod after it.
	if typ != tar.TypeSymlink {
		if err := os.Chmod(p, fi.Mode()); err != nil {
			return err
		}
	}

	// Restore entry atime and mtime.
	// Use special function LUtimesNano not available on go's syscall package because we
	// have to restore symlink's times and not the referenced file times.
	ts := HdrToTimespec(hdr)
	if hdr.Typeflag != tar.TypeSymlink {
		if err := syscall.UtimesNano(p, ts); err != nil {
			return err
		}
	} else {
		if err := LUtimesNano(p, ts); err != nil && err != ErrNotSupportedPlatform {
			return err
		}
	}

	return nil
}

// ExtractFileFromTar extracts a regular file from the given tar, returning its
// contents as a byte slice
func ExtractFileFromTar(tr *tar.Reader, file string) ([]byte, error) {
	for {
		hdr, err := tr.Next()
		switch err {
		case io.EOF:
			return nil, fmt.Errorf("file not found")
		case nil:
			if filepath.Clean(hdr.Name) != filepath.Clean(file) {
				continue
			}
			switch hdr.Typeflag {
			case tar.TypeReg:
			case tar.TypeRegA:
			default:
				return nil, fmt.Errorf("requested file not a regular file")
			}
			buf, err := ioutil.ReadAll(tr)
			if err != nil {
				return nil, fmt.Errorf("error extracting tarball: %v", err)
			}
			return buf, nil
		default:
			return nil, fmt.Errorf("error extracting tarball: %v", err)
		}
	}
}

// makedev mimics glib's gnu_dev_makedev
func makedev(major, minor int) int {
	return (minor & 0xff) | (major & 0xfff << 8) | int((uint64(minor & ^0xff) << 12)) | int(uint64(major & ^0xfff)<<32)
}

func HdrToTimespec(hdr *tar.Header) []syscall.Timespec {
	return []syscall.Timespec{timeToTimespec(hdr.AccessTime), timeToTimespec(hdr.ModTime)}
}

// TODO(sgotti) use UTIMES_OMIT on linux if Time.IsZero ?
func timeToTimespec(time time.Time) (ts syscall.Timespec) {
	nsec := int64(0)
	if !time.IsZero() {
		nsec = time.UnixNano()
	}
	return syscall.NsecToTimespec(nsec)
}

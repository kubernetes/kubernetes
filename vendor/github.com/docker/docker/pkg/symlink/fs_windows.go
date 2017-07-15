package symlink

import (
	"bytes"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"syscall"

	"github.com/docker/docker/pkg/longpath"
)

func toShort(path string) (string, error) {
	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return "", err
	}
	b := p // GetShortPathName says we can reuse buffer
	n, err := syscall.GetShortPathName(&p[0], &b[0], uint32(len(b)))
	if err != nil {
		return "", err
	}
	if n > uint32(len(b)) {
		b = make([]uint16, n)
		if _, err = syscall.GetShortPathName(&p[0], &b[0], uint32(len(b))); err != nil {
			return "", err
		}
	}
	return syscall.UTF16ToString(b), nil
}

func toLong(path string) (string, error) {
	p, err := syscall.UTF16FromString(path)
	if err != nil {
		return "", err
	}
	b := p // GetLongPathName says we can reuse buffer
	n, err := syscall.GetLongPathName(&p[0], &b[0], uint32(len(b)))
	if err != nil {
		return "", err
	}
	if n > uint32(len(b)) {
		b = make([]uint16, n)
		n, err = syscall.GetLongPathName(&p[0], &b[0], uint32(len(b)))
		if err != nil {
			return "", err
		}
	}
	b = b[:n]
	return syscall.UTF16ToString(b), nil
}

func evalSymlinks(path string) (string, error) {
	path, err := walkSymlinks(path)
	if err != nil {
		return "", err
	}

	p, err := toShort(path)
	if err != nil {
		return "", err
	}
	p, err = toLong(p)
	if err != nil {
		return "", err
	}
	// syscall.GetLongPathName does not change the case of the drive letter,
	// but the result of EvalSymlinks must be unique, so we have
	// EvalSymlinks(`c:\a`) == EvalSymlinks(`C:\a`).
	// Make drive letter upper case.
	if len(p) >= 2 && p[1] == ':' && 'a' <= p[0] && p[0] <= 'z' {
		p = string(p[0]+'A'-'a') + p[1:]
	} else if len(p) >= 6 && p[5] == ':' && 'a' <= p[4] && p[4] <= 'z' {
		p = p[:3] + string(p[4]+'A'-'a') + p[5:]
	}
	return filepath.Clean(p), nil
}

const utf8RuneSelf = 0x80

func walkSymlinks(path string) (string, error) {
	const maxIter = 255
	originalPath := path
	// consume path by taking each frontmost path element,
	// expanding it if it's a symlink, and appending it to b
	var b bytes.Buffer
	for n := 0; path != ""; n++ {
		if n > maxIter {
			return "", errors.New("EvalSymlinks: too many links in " + originalPath)
		}

		// A path beginning with `\\?\` represents the root, so automatically
		// skip that part and begin processing the next segment.
		if strings.HasPrefix(path, longpath.Prefix) {
			b.WriteString(longpath.Prefix)
			path = path[4:]
			continue
		}

		// find next path component, p
		var i = -1
		for j, c := range path {
			if c < utf8RuneSelf && os.IsPathSeparator(uint8(c)) {
				i = j
				break
			}
		}
		var p string
		if i == -1 {
			p, path = path, ""
		} else {
			p, path = path[:i], path[i+1:]
		}

		if p == "" {
			if b.Len() == 0 {
				// must be absolute path
				b.WriteRune(filepath.Separator)
			}
			continue
		}

		// If this is the first segment after the long path prefix, accept the
		// current segment as a volume root or UNC share and move on to the next.
		if b.String() == longpath.Prefix {
			b.WriteString(p)
			b.WriteRune(filepath.Separator)
			continue
		}

		fi, err := os.Lstat(b.String() + p)
		if err != nil {
			return "", err
		}
		if fi.Mode()&os.ModeSymlink == 0 {
			b.WriteString(p)
			if path != "" || (b.Len() == 2 && len(p) == 2 && p[1] == ':') {
				b.WriteRune(filepath.Separator)
			}
			continue
		}

		// it's a symlink, put it at the front of path
		dest, err := os.Readlink(b.String() + p)
		if err != nil {
			return "", err
		}
		if filepath.IsAbs(dest) || os.IsPathSeparator(dest[0]) {
			b.Reset()
		}
		path = dest + string(filepath.Separator) + path
	}
	return filepath.Clean(b.String()), nil
}

func isDriveOrRoot(p string) bool {
	if p == string(filepath.Separator) {
		return true
	}

	length := len(p)
	if length >= 2 {
		if p[length-1] == ':' && (('a' <= p[length-2] && p[length-2] <= 'z') || ('A' <= p[length-2] && p[length-2] <= 'Z')) {
			return true
		}
	}
	return false
}

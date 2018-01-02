package fs

import (
	"bytes"
	"context"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

var (
	errTooManyLinks = errors.New("too many links")
)

type currentPath struct {
	path     string
	f        os.FileInfo
	fullPath string
}

func pathChange(lower, upper *currentPath) (ChangeKind, string) {
	if lower == nil {
		if upper == nil {
			panic("cannot compare nil paths")
		}
		return ChangeKindAdd, upper.path
	}
	if upper == nil {
		return ChangeKindDelete, lower.path
	}
	// TODO: compare by directory

	switch i := strings.Compare(lower.path, upper.path); {
	case i < 0:
		// File in lower that is not in upper
		return ChangeKindDelete, lower.path
	case i > 0:
		// File in upper that is not in lower
		return ChangeKindAdd, upper.path
	default:
		return ChangeKindModify, upper.path
	}
}

func sameFile(f1, f2 *currentPath) (bool, error) {
	if os.SameFile(f1.f, f2.f) {
		return true, nil
	}

	equalStat, err := compareSysStat(f1.f.Sys(), f2.f.Sys())
	if err != nil || !equalStat {
		return equalStat, err
	}

	if eq, err := compareCapabilities(f1.fullPath, f2.fullPath); err != nil || !eq {
		return eq, err
	}

	// If not a directory also check size, modtime, and content
	if !f1.f.IsDir() {
		if f1.f.Size() != f2.f.Size() {
			return false, nil
		}
		t1 := f1.f.ModTime()
		t2 := f2.f.ModTime()

		if t1.Unix() != t2.Unix() {
			return false, nil
		}

		// If the timestamp may have been truncated in one of the
		// files, check content of file to determine difference
		if t1.Nanosecond() == 0 || t2.Nanosecond() == 0 {
			if f1.f.Size() > 0 {
				eq, err := compareFileContent(f1.fullPath, f2.fullPath)
				if err != nil || !eq {
					return eq, err
				}
			}
		} else if t1.Nanosecond() != t2.Nanosecond() {
			return false, nil
		}
	}

	return true, nil
}

const compareChuckSize = 32 * 1024

// compareFileContent compares the content of 2 same sized files
// by comparing each byte.
func compareFileContent(p1, p2 string) (bool, error) {
	f1, err := os.Open(p1)
	if err != nil {
		return false, err
	}
	defer f1.Close()
	f2, err := os.Open(p2)
	if err != nil {
		return false, err
	}
	defer f2.Close()

	b1 := make([]byte, compareChuckSize)
	b2 := make([]byte, compareChuckSize)
	for {
		n1, err1 := f1.Read(b1)
		if err1 != nil && err1 != io.EOF {
			return false, err1
		}
		n2, err2 := f2.Read(b2)
		if err2 != nil && err2 != io.EOF {
			return false, err2
		}
		if n1 != n2 || !bytes.Equal(b1[:n1], b2[:n2]) {
			return false, nil
		}
		if err1 == io.EOF && err2 == io.EOF {
			return true, nil
		}
	}
}

func pathWalk(ctx context.Context, root string, pathC chan<- *currentPath) error {
	return filepath.Walk(root, func(path string, f os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		// Rebase path
		path, err = filepath.Rel(root, path)
		if err != nil {
			return err
		}

		path = filepath.Join(string(os.PathSeparator), path)

		// Skip root
		if path == string(os.PathSeparator) {
			return nil
		}

		p := &currentPath{
			path:     path,
			f:        f,
			fullPath: filepath.Join(root, path),
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case pathC <- p:
			return nil
		}
	})
}

func nextPath(ctx context.Context, pathC <-chan *currentPath) (*currentPath, error) {
	select {
	case <-ctx.Done():
		return nil, ctx.Err()
	case p := <-pathC:
		return p, nil
	}
}

// RootPath joins a path with a root, evaluating and bounding any
// symlink to the root directory.
func RootPath(root, path string) (string, error) {
	if path == "" {
		return root, nil
	}
	var linksWalked int // to protect against cycles
	for {
		i := linksWalked
		newpath, err := walkLinks(root, path, &linksWalked)
		if err != nil {
			return "", err
		}
		path = newpath
		if i == linksWalked {
			newpath = filepath.Join("/", newpath)
			if path == newpath {
				return filepath.Join(root, newpath), nil
			}
			path = newpath
		}
	}
}

func walkLink(root, path string, linksWalked *int) (newpath string, islink bool, err error) {
	if *linksWalked > 255 {
		return "", false, errTooManyLinks
	}

	path = filepath.Join("/", path)
	if path == "/" {
		return path, false, nil
	}
	realPath := filepath.Join(root, path)

	fi, err := os.Lstat(realPath)
	if err != nil {
		// If path does not yet exist, treat as non-symlink
		if os.IsNotExist(err) {
			return path, false, nil
		}
		return "", false, err
	}
	if fi.Mode()&os.ModeSymlink == 0 {
		return path, false, nil
	}
	newpath, err = os.Readlink(realPath)
	if err != nil {
		return "", false, err
	}
	if filepath.IsAbs(newpath) && strings.HasPrefix(newpath, root) {
		newpath = newpath[:len(root)]
		if !strings.HasPrefix(newpath, "/") {
			newpath = "/" + newpath
		}
	}
	*linksWalked++
	return newpath, true, nil
}

func walkLinks(root, path string, linksWalked *int) (string, error) {
	switch dir, file := filepath.Split(path); {
	case dir == "":
		newpath, _, err := walkLink(root, file, linksWalked)
		return newpath, err
	case file == "":
		if os.IsPathSeparator(dir[len(dir)-1]) {
			if dir == "/" {
				return dir, nil
			}
			return walkLinks(root, dir[:len(dir)-1], linksWalked)
		}
		newpath, _, err := walkLink(root, dir, linksWalked)
		return newpath, err
	default:
		newdir, err := walkLinks(root, dir, linksWalked)
		if err != nil {
			return "", err
		}
		newpath, islink, err := walkLink(root, filepath.Join(newdir, file), linksWalked)
		if err != nil {
			return "", err
		}
		if !islink {
			return newpath, nil
		}
		if filepath.IsAbs(newpath) {
			return newpath, nil
		}
		return filepath.Join(newdir, newpath), nil
	}
}

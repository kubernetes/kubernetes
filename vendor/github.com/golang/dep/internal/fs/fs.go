// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fs

import (
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"syscall"
	"unicode"

	"github.com/pkg/errors"
)

// HasFilepathPrefix will determine if "path" starts with "prefix" from
// the point of view of a filesystem.
//
// Unlike filepath.HasPrefix, this function is path-aware, meaning that
// it knows that two directories /foo and /foobar are not the same
// thing, and therefore HasFilepathPrefix("/foobar", "/foo") will return
// false.
//
// This function also handles the case where the involved filesystems
// are case-insensitive, meaning /foo/bar and /Foo/Bar correspond to the
// same file. In that situation HasFilepathPrefix("/Foo/Bar", "/foo")
// will return true. The implementation is *not* OS-specific, so a FAT32
// filesystem mounted on Linux will be handled correctly.
func HasFilepathPrefix(path, prefix string) (bool, error) {
	// this function is more convoluted then ideal due to need for special
	// handling of volume name/drive letter on Windows. vnPath and vnPrefix
	// are first compared, and then used to initialize initial values of p and
	// d which will be appended to for incremental checks using
	// IsCaseSensitiveFilesystem and then equality.

	// no need to check IsCaseSensitiveFilesystem because VolumeName return
	// empty string on all non-Windows machines
	vnPath := strings.ToLower(filepath.VolumeName(path))
	vnPrefix := strings.ToLower(filepath.VolumeName(prefix))
	if vnPath != vnPrefix {
		return false, nil
	}

	// Because filepath.Join("c:","dir") returns "c:dir", we have to manually
	// add path separator to drive letters. Also, we need to set the path root
	// on *nix systems, since filepath.Join("", "dir") returns a relative path.
	vnPath += string(os.PathSeparator)
	vnPrefix += string(os.PathSeparator)

	var dn string

	if isDir, err := IsDir(path); err != nil {
		return false, errors.Wrap(err, "failed to check filepath prefix")
	} else if isDir {
		dn = path
	} else {
		dn = filepath.Dir(path)
	}

	dn = strings.TrimSuffix(dn, string(os.PathSeparator))
	prefix = strings.TrimSuffix(prefix, string(os.PathSeparator))

	// [1:] in the lines below eliminates empty string on *nix and volume name on Windows
	dirs := strings.Split(dn, string(os.PathSeparator))[1:]
	prefixes := strings.Split(prefix, string(os.PathSeparator))[1:]

	if len(prefixes) > len(dirs) {
		return false, nil
	}

	// d,p are initialized with "/" on *nix and volume name on Windows
	d := vnPath
	p := vnPrefix

	for i := range prefixes {
		// need to test each component of the path for
		// case-sensitiveness because on Unix we could have
		// something like ext4 filesystem mounted on FAT
		// mountpoint, mounted on ext4 filesystem, i.e. the
		// problematic filesystem is not the last one.
		caseSensitive, err := IsCaseSensitiveFilesystem(filepath.Join(d, dirs[i]))
		if err != nil {
			return false, errors.Wrap(err, "failed to check filepath prefix")
		}
		if caseSensitive {
			d = filepath.Join(d, dirs[i])
			p = filepath.Join(p, prefixes[i])
		} else {
			d = filepath.Join(d, strings.ToLower(dirs[i]))
			p = filepath.Join(p, strings.ToLower(prefixes[i]))
		}

		if p != d {
			return false, nil
		}
	}

	return true, nil
}

// EquivalentPaths compares the paths passed to check if they are equivalent.
// It respects the case-sensitivity of the underlying filesysyems.
func EquivalentPaths(p1, p2 string) (bool, error) {
	p1 = filepath.Clean(p1)
	p2 = filepath.Clean(p2)

	fi1, err := os.Stat(p1)
	if err != nil {
		return false, errors.Wrapf(err, "could not check for path equivalence")
	}
	fi2, err := os.Stat(p2)
	if err != nil {
		return false, errors.Wrapf(err, "could not check for path equivalence")
	}

	p1Filename, p2Filename := "", ""

	if !fi1.IsDir() {
		p1, p1Filename = filepath.Split(p1)
	}
	if !fi2.IsDir() {
		p2, p2Filename = filepath.Split(p2)
	}

	if isPrefix1, err := HasFilepathPrefix(p1, p2); err != nil {
		return false, errors.Wrap(err, "failed to check for path equivalence")
	} else if isPrefix2, err := HasFilepathPrefix(p2, p1); err != nil {
		return false, errors.Wrap(err, "failed to check for path equivalence")
	} else if !isPrefix1 || !isPrefix2 {
		return false, nil
	}

	if p1Filename != "" || p2Filename != "" {
		caseSensitive, err := IsCaseSensitiveFilesystem(filepath.Join(p1, p1Filename))
		if err != nil {
			return false, errors.Wrap(err, "could not check for filesystem case-sensitivity")
		}
		if caseSensitive {
			if p1Filename != p2Filename {
				return false, nil
			}
		} else {
			if strings.ToLower(p1Filename) != strings.ToLower(p2Filename) {
				return false, nil
			}
		}
	}

	return true, nil
}

// RenameWithFallback attempts to rename a file or directory, but falls back to
// copying in the event of a cross-device link error. If the fallback copy
// succeeds, src is still removed, emulating normal rename behavior.
func RenameWithFallback(src, dst string) error {
	_, err := os.Stat(src)
	if err != nil {
		return errors.Wrapf(err, "cannot stat %s", src)
	}

	err = os.Rename(src, dst)
	if err == nil {
		return nil
	}

	return renameFallback(err, src, dst)
}

// renameByCopy attempts to rename a file or directory by copying it to the
// destination and then removing the src thus emulating the rename behavior.
func renameByCopy(src, dst string) error {
	var cerr error
	if dir, _ := IsDir(src); dir {
		cerr = CopyDir(src, dst)
		if cerr != nil {
			cerr = errors.Wrap(cerr, "copying directory failed")
		}
	} else {
		cerr = copyFile(src, dst)
		if cerr != nil {
			cerr = errors.Wrap(cerr, "copying file failed")
		}
	}

	if cerr != nil {
		return errors.Wrapf(cerr, "rename fallback failed: cannot rename %s to %s", src, dst)
	}

	return errors.Wrapf(os.RemoveAll(src), "cannot delete %s", src)
}

// IsCaseSensitiveFilesystem determines if the filesystem where dir
// exists is case sensitive or not.
//
// CAVEAT: this function works by taking the last component of the given
// path and flipping the case of the first letter for which case
// flipping is a reversible operation (/foo/Bar → /foo/bar), then
// testing for the existence of the new filename. There are two
// possibilities:
//
// 1. The alternate filename does not exist. We can conclude that the
// filesystem is case sensitive.
//
// 2. The filename happens to exist. We have to test if the two files
// are the same file (case insensitive file system) or different ones
// (case sensitive filesystem).
//
// If the input directory is such that the last component is composed
// exclusively of case-less codepoints (e.g.  numbers), this function will
// return false.
func IsCaseSensitiveFilesystem(dir string) (bool, error) {
	alt := filepath.Join(filepath.Dir(dir), genTestFilename(filepath.Base(dir)))

	dInfo, err := os.Stat(dir)
	if err != nil {
		return false, errors.Wrap(err, "could not determine the case-sensitivity of the filesystem")
	}

	aInfo, err := os.Stat(alt)
	if err != nil {
		// If the file doesn't exists, assume we are on a case-sensitive filesystem.
		if os.IsNotExist(err) {
			return true, nil
		}

		return false, errors.Wrap(err, "could not determine the case-sensitivity of the filesystem")
	}

	return !os.SameFile(dInfo, aInfo), nil
}

// genTestFilename returns a string with at most one rune case-flipped.
//
// The transformation is applied only to the first rune that can be
// reversibly case-flipped, meaning:
//
// * A lowercase rune for which it's true that lower(upper(r)) == r
// * An uppercase rune for which it's true that upper(lower(r)) == r
//
// All the other runes are left intact.
func genTestFilename(str string) string {
	flip := true
	return strings.Map(func(r rune) rune {
		if flip {
			if unicode.IsLower(r) {
				u := unicode.ToUpper(r)
				if unicode.ToLower(u) == r {
					r = u
					flip = false
				}
			} else if unicode.IsUpper(r) {
				l := unicode.ToLower(r)
				if unicode.ToUpper(l) == r {
					r = l
					flip = false
				}
			}
		}
		return r
	}, str)
}

var errPathNotDir = errors.New("given path is not a directory")

// ReadActualFilenames is used to determine the actual file names in given directory.
//
// On case sensitive file systems like ext4, it will check if those files exist using
// `os.Stat` and return a map with key and value as filenames which exist in the folder.
//
// Otherwise, it reads the contents of the directory and returns a map which has the
// given file name as the key and actual filename as the value(if it was found).
func ReadActualFilenames(dirPath string, names []string) (map[string]string, error) {
	actualFilenames := make(map[string]string, len(names))
	if len(names) == 0 {
		// This isn't expected to happen for current usage. Adding edge case handling,
		// as it may be useful in future.
		return actualFilenames, nil
	}
	// First, check that the given path is valid and it is a directory
	dirStat, err := os.Stat(dirPath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read actual filenames")
	}

	if !dirStat.IsDir() {
		return nil, errPathNotDir
	}

	// Ideally, we would use `os.Stat` for getting the actual file names but that returns
	// the name we passed in as an argument and not the actual filename. So we are forced
	// to list the directory contents and check against that. Since this check is costly,
	// we do it only if absolutely necessary.
	caseSensitive, err := IsCaseSensitiveFilesystem(dirPath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read actual filenames")
	}
	if caseSensitive {
		// There will be no difference between actual filename and given filename. So
		// just check if those files exist.
		for _, name := range names {
			_, err := os.Stat(filepath.Join(dirPath, name))
			if err == nil {
				actualFilenames[name] = name
			} else if !os.IsNotExist(err) {
				// Some unexpected err, wrap and return it.
				return nil, errors.Wrap(err, "failed to read actual filenames")
			}
		}
		return actualFilenames, nil
	}

	dir, err := os.Open(dirPath)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read actual filenames")
	}
	defer dir.Close()

	// Pass -1 to read all filenames in directory
	filenames, err := dir.Readdirnames(-1)
	if err != nil {
		return nil, errors.Wrap(err, "failed to read actual filenames")
	}

	// namesMap holds the mapping from lowercase name to search name. Using this, we can
	// avoid repeatedly looping through names.
	namesMap := make(map[string]string, len(names))
	for _, name := range names {
		namesMap[strings.ToLower(name)] = name
	}

	for _, filename := range filenames {
		searchName, ok := namesMap[strings.ToLower(filename)]
		if ok {
			// We are interested in this file, case insensitive match successful.
			actualFilenames[searchName] = filename
			if len(actualFilenames) == len(names) {
				// We found all that we were looking for.
				return actualFilenames, nil
			}
		}
	}
	return actualFilenames, nil
}

var (
	errSrcNotDir = errors.New("source is not a directory")
	errDstExist  = errors.New("destination already exists")
)

// CopyDir recursively copies a directory tree, attempting to preserve permissions.
// Source directory must exist, destination directory must *not* exist.
func CopyDir(src, dst string) error {
	src = filepath.Clean(src)
	dst = filepath.Clean(dst)

	// We use os.Lstat() here to ensure we don't fall in a loop where a symlink
	// actually links to a one of its parent directories.
	fi, err := os.Lstat(src)
	if err != nil {
		return err
	}
	if !fi.IsDir() {
		return errSrcNotDir
	}

	_, err = os.Stat(dst)
	if err != nil && !os.IsNotExist(err) {
		return err
	}
	if err == nil {
		return errDstExist
	}

	if err = os.MkdirAll(dst, fi.Mode()); err != nil {
		return errors.Wrapf(err, "cannot mkdir %s", dst)
	}

	entries, err := ioutil.ReadDir(src)
	if err != nil {
		return errors.Wrapf(err, "cannot read directory %s", dst)
	}

	for _, entry := range entries {
		srcPath := filepath.Join(src, entry.Name())
		dstPath := filepath.Join(dst, entry.Name())

		if entry.IsDir() {
			if err = CopyDir(srcPath, dstPath); err != nil {
				return errors.Wrap(err, "copying directory failed")
			}
		} else {
			// This will include symlinks, which is what we want when
			// copying things.
			if err = copyFile(srcPath, dstPath); err != nil {
				return errors.Wrap(err, "copying file failed")
			}
		}
	}

	return nil
}

// copyFile copies the contents of the file named src to the file named
// by dst. The file will be created if it does not already exist. If the
// destination file exists, all its contents will be replaced by the contents
// of the source file. The file mode will be copied from the source.
func copyFile(src, dst string) (err error) {
	if sym, err := IsSymlink(src); err != nil {
		return errors.Wrap(err, "symlink check failed")
	} else if sym {
		if err := cloneSymlink(src, dst); err != nil {
			if runtime.GOOS == "windows" {
				// If cloning the symlink fails on Windows because the user
				// does not have the required privileges, ignore the error and
				// fall back to copying the file contents.
				//
				// ERROR_PRIVILEGE_NOT_HELD is 1314 (0x522):
				// https://msdn.microsoft.com/en-us/library/windows/desktop/ms681385(v=vs.85).aspx
				if lerr, ok := err.(*os.LinkError); ok && lerr.Err != syscall.Errno(1314) {
					return err
				}
			} else {
				return err
			}
		} else {
			return nil
		}
	}

	in, err := os.Open(src)
	if err != nil {
		return
	}
	defer in.Close()

	out, err := os.Create(dst)
	if err != nil {
		return
	}

	if _, err = io.Copy(out, in); err != nil {
		out.Close()
		return
	}

	// Check for write errors on Close
	if err = out.Close(); err != nil {
		return
	}

	si, err := os.Stat(src)
	if err != nil {
		return
	}

	// Temporary fix for Go < 1.9
	//
	// See: https://github.com/golang/dep/issues/774
	// and https://github.com/golang/go/issues/20829
	if runtime.GOOS == "windows" {
		dst = fixLongPath(dst)
	}
	err = os.Chmod(dst, si.Mode())

	return
}

// cloneSymlink will create a new symlink that points to the resolved path of sl.
// If sl is a relative symlink, dst will also be a relative symlink.
func cloneSymlink(sl, dst string) error {
	resolved, err := os.Readlink(sl)
	if err != nil {
		return err
	}

	return os.Symlink(resolved, dst)
}

// EnsureDir tries to ensure that a directory is present at the given path. It first
// checks if the directory already exists at the given path. If there isn't one, it tries
// to create it with the given permissions. However, it does not try to create the
// directory recursively.
func EnsureDir(path string, perm os.FileMode) error {
	_, err := IsDir(path)

	if os.IsNotExist(err) {
		err = os.Mkdir(path, perm)
		if err != nil {
			return errors.Wrapf(err, "failed to ensure directory at %q", path)
		}
	}

	return err
}

// IsDir determines is the path given is a directory or not.
func IsDir(name string) (bool, error) {
	fi, err := os.Stat(name)
	if err != nil {
		return false, err
	}
	if !fi.IsDir() {
		return false, errors.Errorf("%q is not a directory", name)
	}
	return true, nil
}

// IsNonEmptyDir determines if the path given is a non-empty directory or not.
func IsNonEmptyDir(name string) (bool, error) {
	isDir, err := IsDir(name)
	if err != nil && !os.IsNotExist(err) {
		return false, err
	} else if !isDir {
		return false, nil
	}

	// Get file descriptor
	f, err := os.Open(name)
	if err != nil {
		return false, err
	}
	defer f.Close()

	// Query only 1 child. EOF if no children.
	_, err = f.Readdirnames(1)
	switch err {
	case io.EOF:
		return false, nil
	case nil:
		return true, nil
	default:
		return false, err
	}
}

// IsRegular determines if the path given is a regular file or not.
func IsRegular(name string) (bool, error) {
	fi, err := os.Stat(name)
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	mode := fi.Mode()
	if mode&os.ModeType != 0 {
		return false, errors.Errorf("%q is a %v, expected a file", name, mode)
	}
	return true, nil
}

// IsSymlink determines if the given path is a symbolic link.
func IsSymlink(path string) (bool, error) {
	l, err := os.Lstat(path)
	if err != nil {
		return false, err
	}

	return l.Mode()&os.ModeSymlink == os.ModeSymlink, nil
}

// fixLongPath returns the extended-length (\\?\-prefixed) form of
// path when needed, in order to avoid the default 260 character file
// path limit imposed by Windows. If path is not easily converted to
// the extended-length form (for example, if path is a relative path
// or contains .. elements), or is short enough, fixLongPath returns
// path unmodified.
//
// See https://msdn.microsoft.com/en-us/library/windows/desktop/aa365247(v=vs.85).aspx#maxpath
func fixLongPath(path string) string {
	// Do nothing (and don't allocate) if the path is "short".
	// Empirically (at least on the Windows Server 2013 builder),
	// the kernel is arbitrarily okay with < 248 bytes. That
	// matches what the docs above say:
	// "When using an API to create a directory, the specified
	// path cannot be so long that you cannot append an 8.3 file
	// name (that is, the directory name cannot exceed MAX_PATH
	// minus 12)." Since MAX_PATH is 260, 260 - 12 = 248.
	//
	// The MSDN docs appear to say that a normal path that is 248 bytes long
	// will work; empirically the path must be less then 248 bytes long.
	if len(path) < 248 {
		// Don't fix. (This is how Go 1.7 and earlier worked,
		// not automatically generating the \\?\ form)
		return path
	}

	// The extended form begins with \\?\, as in
	// \\?\c:\windows\foo.txt or \\?\UNC\server\share\foo.txt.
	// The extended form disables evaluation of . and .. path
	// elements and disables the interpretation of / as equivalent
	// to \. The conversion here rewrites / to \ and elides
	// . elements as well as trailing or duplicate separators. For
	// simplicity it avoids the conversion entirely for relative
	// paths or paths containing .. elements. For now,
	// \\server\share paths are not converted to
	// \\?\UNC\server\share paths because the rules for doing so
	// are less well-specified.
	if len(path) >= 2 && path[:2] == `\\` {
		// Don't canonicalize UNC paths.
		return path
	}
	if !isAbs(path) {
		// Relative path
		return path
	}

	const prefix = `\\?`

	pathbuf := make([]byte, len(prefix)+len(path)+len(`\`))
	copy(pathbuf, prefix)
	n := len(path)
	r, w := 0, len(prefix)
	for r < n {
		switch {
		case os.IsPathSeparator(path[r]):
			// empty block
			r++
		case path[r] == '.' && (r+1 == n || os.IsPathSeparator(path[r+1])):
			// /./
			r++
		case r+1 < n && path[r] == '.' && path[r+1] == '.' && (r+2 == n || os.IsPathSeparator(path[r+2])):
			// /../ is currently unhandled
			return path
		default:
			pathbuf[w] = '\\'
			w++
			for ; r < n && !os.IsPathSeparator(path[r]); r++ {
				pathbuf[w] = path[r]
				w++
			}
		}
	}
	// A drive's root directory needs a trailing \
	if w == len(`\\?\c:`) {
		pathbuf[w] = '\\'
		w++
	}
	return string(pathbuf[:w])
}

func isAbs(path string) (b bool) {
	v := volumeName(path)
	if v == "" {
		return false
	}
	path = path[len(v):]
	if path == "" {
		return false
	}
	return os.IsPathSeparator(path[0])
}

func volumeName(path string) (v string) {
	if len(path) < 2 {
		return ""
	}
	// with drive letter
	c := path[0]
	if path[1] == ':' &&
		('0' <= c && c <= '9' || 'a' <= c && c <= 'z' ||
			'A' <= c && c <= 'Z') {
		return path[:2]
	}
	// is it UNC
	if l := len(path); l >= 5 && os.IsPathSeparator(path[0]) && os.IsPathSeparator(path[1]) &&
		!os.IsPathSeparator(path[2]) && path[2] != '.' {
		// first, leading `\\` and next shouldn't be `\`. its server name.
		for n := 3; n < l-1; n++ {
			// second, next '\' shouldn't be repeated.
			if os.IsPathSeparator(path[n]) {
				n++
				// third, following something characters. its share name.
				if !os.IsPathSeparator(path[n]) {
					if path[n] == '.' {
						break
					}
					for ; n < l; n++ {
						if os.IsPathSeparator(path[n]) {
							break
						}
					}
					return path[:n]
				}
				break
			}
		}
	}
	return ""
}

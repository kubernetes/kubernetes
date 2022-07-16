package filepath

import (
	"regexp"
	"strings"
)

var windowsAbs = regexp.MustCompile(`^[a-zA-Z]:\\.*$`)

// Abs is a version of path/filepath's Abs with an explicit operating
// system and current working directory.
func Abs(os, path, cwd string) (_ string, err error) {
	if IsAbs(os, path) {
		return Clean(os, path), nil
	}
	return Clean(os, Join(os, cwd, path)), nil
}

// IsAbs is a version of path/filepath's IsAbs with an explicit
// operating system.
func IsAbs(os, path string) bool {
	if os == "windows" {
		// FIXME: copy hideous logic from Go's
		// src/path/filepath/path_windows.go into somewhere where we can
		// put 3-clause BSD licensed code.
		return windowsAbs.MatchString(path)
	}
	sep := Separator(os)

	// POSIX has [1]:
	//
	// > If a pathname begins with two successive <slash> characters,
	// > the first component following the leading <slash> characters
	// > may be interpreted in an implementation-defined manner,
	// > although more than two leading <slash> characters shall be
	// > treated as a single <slash> character.
	//
	// And Boost treats // as non-absolute [2], but Linux [3,4], Python
	// [5] and Go [6] all treat // as absolute.
	//
	// [1]: http://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap04.html#tag_04_13
	// [2]: https://github.com/boostorg/filesystem/blob/boost-1.64.0/test/path_test.cpp#L861
	// [3]: http://man7.org/linux/man-pages/man7/path_resolution.7.html
	// [4]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/tree/Documentation/filesystems/path-lookup.md?h=v4.12#n41
	// [5]: https://github.com/python/cpython/blob/v3.6.1/Lib/posixpath.py#L64-L66
	// [6]: https://go.googlesource.com/go/+/go1.8.3/src/path/path.go#199
	return strings.HasPrefix(path, string(sep))
}

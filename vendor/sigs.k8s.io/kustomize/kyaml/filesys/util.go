// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package filesys

import (
	"os"
	"path/filepath"
	"strings"
)

// RootedPath returns a rooted path, e.g. "/foo/bar" as
// opposed to "foo/bar".
func RootedPath(elem ...string) string {
	return Separator + filepath.Join(elem...)
}

// StripTrailingSeps trims trailing filepath separators from input.
func StripTrailingSeps(s string) string {
	k := len(s)
	for k > 0 && s[k-1] == filepath.Separator {
		k--
	}
	return s[:k]
}

// StripLeadingSeps trims leading filepath separators from input.
func StripLeadingSeps(s string) string {
	k := 0
	for k < len(s) && s[k] == filepath.Separator {
		k++
	}
	return s[k:]
}

// PathSplit converts a file path to a slice of string.
// If the path is absolute (if the path has a leading slash),
// then the first entry in the result is an empty string.
// Desired:  path == PathJoin(PathSplit(path))
func PathSplit(incoming string) []string {
	if incoming == "" {
		return []string{}
	}
	dir, path := filepath.Split(incoming)
	if dir == string(os.PathSeparator) {
		if path == "" {
			return []string{""}
		}
		return []string{"", path}
	}
	dir = strings.TrimSuffix(dir, string(os.PathSeparator))
	if dir == "" {
		return []string{path}
	}
	return append(PathSplit(dir), path)
}

// PathJoin converts a slice of string to a file path.
// If the first entry is an empty string, then the returned
// path is absolute (it has a leading slash).
// Desired:  path == PathJoin(PathSplit(path))
func PathJoin(incoming []string) string {
	if len(incoming) == 0 {
		return ""
	}
	if incoming[0] == "" {
		return string(os.PathSeparator) + filepath.Join(incoming[1:]...)
	}
	return filepath.Join(incoming...)
}

// InsertPathPart inserts 'part' at position 'pos' in the given filepath.
// The first position is 0.
//
// E.g. if part == 'PEACH'
//
//                  OLD : NEW                    : POS
//      --------------------------------------------------------
//              {empty} : PEACH                  : irrelevant
//                    / : /PEACH                 : irrelevant
//                  pie : PEACH/pie              : 0 (or negative)
//                 /pie : /PEACH/pie             : 0 (or negative)
//                  raw : raw/PEACH              : 1 (or larger)
//                 /raw : /raw/PEACH             : 1 (or larger)
//      a/nice/warm/pie : a/nice/warm/PEACH/pie  : 3
//     /a/nice/warm/pie : /a/nice/warm/PEACH/pie : 3
//
// * An empty part results in no change.
//
// * Absolute paths get their leading '/' stripped, treated like
//   relative paths, and the leading '/' is re-added on output.
//   The meaning of pos is intentionally the same in either absolute or
//   relative paths; if it weren't, this function could convert absolute
//   paths to relative paths, which is not desirable.
//
// * For robustness (liberal input, conservative output) Pos values that
//   that are too small (large) to index the split filepath result in a
//   prefix (postfix) rather than an error.  Use extreme position values
//   to assure a prefix or postfix (e.g. 0 will always prefix, and
//   9999 will presumably always postfix).
func InsertPathPart(path string, pos int, part string) string {
	if part == "" {
		return path
	}
	parts := PathSplit(path)
	if pos < 0 {
		pos = 0
	} else if pos > len(parts) {
		pos = len(parts)
	}
	if len(parts) > 0 && parts[0] == "" && pos < len(parts) {
		// An empty string at 0 indicates an absolute path, and means
		// we must increment pos.  This change means that a position
		// specification has the same meaning in relative and absolute paths.
		// E.g. in either the path 'a/b/c' or the path '/a/b/c',
		// 'a' is at 0, 'b' is at 1 and 'c' is at 2, and inserting at
		// zero means a new first field _without_ changing an absolute
		// path to a relative path.
		pos++
	}
	result := make([]string, len(parts)+1)
	copy(result, parts[0:pos])
	result[pos] = part
	return PathJoin(append(result, parts[pos:]...)) // nolint: makezero
}

func IsHiddenFilePath(pattern string) bool {
	return strings.HasPrefix(filepath.Base(pattern), ".")
}

// Removes paths containing hidden files/folders from a list of paths
func RemoveHiddenFiles(paths []string) []string {
	if len(paths) == 0 {
		return paths
	}
	var result []string
	for _, path := range paths {
		if !IsHiddenFilePath(path) {
			result = append(result, path)
		}
	}
	return result
}

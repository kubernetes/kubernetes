package getter

import (
	"fmt"
	"path/filepath"
	"strings"
)

// SourceDirSubdir takes a source and returns a tuple of the URL without
// the subdir and the URL with the subdir.
func SourceDirSubdir(src string) (string, string) {
	// Calcaulate an offset to avoid accidentally marking the scheme
	// as the dir.
	var offset int
	if idx := strings.Index(src, "://"); idx > -1 {
		offset = idx + 3
	}

	// First see if we even have an explicit subdir
	idx := strings.Index(src[offset:], "//")
	if idx == -1 {
		return src, ""
	}

	idx += offset
	subdir := src[idx+2:]
	src = src[:idx]

	// Next, check if we have query parameters and push them onto the
	// URL.
	if idx = strings.Index(subdir, "?"); idx > -1 {
		query := subdir[idx:]
		subdir = subdir[:idx]
		src += query
	}

	return src, subdir
}

// SubdirGlob returns the actual subdir with globbing processed.
//
// dst should be a destination directory that is already populated (the
// download is complete) and subDir should be the set subDir. If subDir
// is an empty string, this returns an empty string.
//
// The returned path is the full absolute path.
func SubdirGlob(dst, subDir string) (string, error) {
	matches, err := filepath.Glob(filepath.Join(dst, subDir))
	if err != nil {
		return "", err
	}

	if len(matches) == 0 {
		return "", fmt.Errorf("subdir %q not found", subDir)
	}

	if len(matches) > 1 {
		return "", fmt.Errorf("subdir %q matches multiple paths", subDir)
	}

	return matches[0], nil
}

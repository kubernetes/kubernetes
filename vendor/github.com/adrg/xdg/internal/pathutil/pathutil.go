package pathutil

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// Unique eliminates the duplicate paths from the provided slice and returns
// the result. The items in the output slice are in the order in which they
// occur in the input slice. If a `home` location is provided, the paths are
// expanded using the `ExpandHome` function.
func Unique(paths []string, home string) []string {
	var (
		uniq     []string
		registry = map[string]struct{}{}
	)

	for _, p := range paths {
		p = ExpandHome(p, home)
		if p != "" && filepath.IsAbs(p) {
			if _, ok := registry[p]; ok {
				continue
			}

			registry[p] = struct{}{}
			uniq = append(uniq, p)
		}
	}

	return uniq
}

// Create returns a suitable location relative to which the file with the
// specified `name` can be written. The first path from the provided `paths`
// slice which is successfully created (or already exists) is used as a base
// path for the file. The `name` parameter should contain the name of the file
// which is going to be written in the location returned by this function, but
// it can also contain a set of parent directories, which will be created
// relative to the selected parent path.
func Create(name string, paths []string) (string, error) {
	var searchedPaths []string
	for _, p := range paths {
		p = filepath.Join(p, name)

		dir := filepath.Dir(p)
		if Exists(dir) {
			return p, nil
		}
		if err := os.MkdirAll(dir, os.ModeDir|0700); err == nil {
			return p, nil
		}

		searchedPaths = append(searchedPaths, dir)
	}

	return "", fmt.Errorf("could not create any of the following paths: %s",
		strings.Join(searchedPaths, ", "))
}

// Search searches for the file with the specified `name` in the provided
// slice of `paths`. The `name` parameter must contain the name of the file,
// but it can also contain a set of parent directories.
func Search(name string, paths []string) (string, error) {
	var searchedPaths []string
	for _, p := range paths {
		p = filepath.Join(p, name)
		if Exists(p) {
			return p, nil
		}

		searchedPaths = append(searchedPaths, filepath.Dir(p))
	}

	return "", fmt.Errorf("could not locate `%s` in any of the following paths: %s",
		filepath.Base(name), strings.Join(searchedPaths, ", "))
}

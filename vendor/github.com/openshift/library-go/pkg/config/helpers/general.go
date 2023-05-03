package helpers

import (
	"fmt"
	"path/filepath"
	"strings"
)

// ResolvePaths updates the given refs to be absolute paths, relative to the given base directory.
// Empty and "-" paths are never resolved.
func ResolvePaths(refs []*string, base string) error {
	for _, ref := range refs {
		// Don't resolve empty paths, or "-"
		if len(*ref) > 0 && *ref != "-" {
			// Don't resolve absolute paths
			if !filepath.IsAbs(*ref) {
				*ref = filepath.Join(base, *ref)
			}
		}
	}
	return nil
}

func makeRelative(path, base string) (string, error) {
	if len(path) > 0 && path != "-" {
		rel, err := filepath.Rel(base, path)
		if err != nil {
			return path, err
		}
		return rel, nil
	}
	return path, nil
}

// RelativizePathWithNoBacksteps updates the given refs to be relative paths, relative to the given base directory as long as they do not require backsteps.
// Any path requiring a backstep is left as-is as long it is absolute.  Any non-absolute path that can't be relativized produces an error
// Empty and "-" paths are never relativized.
func RelativizePathWithNoBacksteps(refs []*string, base string) error {
	for _, ref := range refs {
		// Don't relativize empty paths, or "-"
		if len(*ref) > 0 && *ref != "-" {
			rel, err := makeRelative(*ref, base)
			if err != nil {
				return err
			}

			if rel == "-" {
				rel = "./-"
			}

			// if we have a backstep, don't mess with the path
			if strings.HasPrefix(rel, "../") {
				if filepath.IsAbs(*ref) {
					continue
				}

				return fmt.Errorf("%v requires backsteps and is not absolute", *ref)
			}

			*ref = rel
		}
	}
	return nil
}

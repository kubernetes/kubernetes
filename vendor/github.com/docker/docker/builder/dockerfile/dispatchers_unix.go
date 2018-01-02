// +build !windows

package dockerfile

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// normaliseWorkdir normalises a user requested working directory in a
// platform semantically consistent way.
func normaliseWorkdir(current string, requested string) (string, error) {
	if requested == "" {
		return "", errors.New("cannot normalise nothing")
	}
	current = filepath.FromSlash(current)
	requested = filepath.FromSlash(requested)
	if !filepath.IsAbs(requested) {
		return filepath.Join(string(os.PathSeparator), current, requested), nil
	}
	return requested, nil
}

func errNotJSON(command, _ string) error {
	return fmt.Errorf("%s requires the arguments to be in JSON form", command)
}

// equalEnvKeys compare two strings and returns true if they are equal. On
// Windows this comparison is case insensitive.
func equalEnvKeys(from, to string) bool {
	return from == to
}

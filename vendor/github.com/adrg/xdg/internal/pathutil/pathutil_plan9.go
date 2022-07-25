package pathutil

import (
	"os"
	"path/filepath"
	"strings"
)

// Exists returns true if the specified path exists.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil || os.IsExist(err)
}

// ExpandHome substitutes `~` and `$home` at the start of the specified
// `path` using the provided `home` location.
func ExpandHome(path, home string) string {
	if path == "" || home == "" {
		return path
	}
	if path[0] == '~' {
		return filepath.Join(home, path[1:])
	}
	if strings.HasPrefix(path, "$home") {
		return filepath.Join(home, path[5:])
	}

	return path
}

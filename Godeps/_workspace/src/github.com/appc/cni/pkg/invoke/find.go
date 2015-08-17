package invoke

import (
	"os"
	"path/filepath"
	"strings"
)

func FindInPath(plugin string, path []string) string {
	for _, p := range path {
		fullname := filepath.Join(p, plugin)
		if fi, err := os.Stat(fullname); err == nil && fi.Mode().IsRegular() {
			return fullname
		}
	}
	return ""
}

// Find returns the full path of the plugin by searching in CNI_PATH
func Find(plugin string) string {
	paths := strings.Split(os.Getenv("CNI_PATH"), ":")
	return FindInPath(plugin, paths)
}

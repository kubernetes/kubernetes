// +build go1.8,!windows,amd64

package plugin

import (
	"fmt"
	"path/filepath"
	"plugin"
	"runtime"
)

// loadPlugins loads all plugins for the OS and Arch
// that containerd is built for inside the provided path
func loadPlugins(path string) error {
	abs, err := filepath.Abs(path)
	if err != nil {
		return err
	}
	pattern := filepath.Join(abs, fmt.Sprintf(
		"*-%s-%s.%s",
		runtime.GOOS,
		runtime.GOARCH,
		getLibExt(),
	))
	libs, err := filepath.Glob(pattern)
	if err != nil {
		return err
	}
	for _, lib := range libs {
		if _, err := plugin.Open(lib); err != nil {
			return err
		}
	}
	return nil
}

// getLibExt returns a platform specific lib extension for
// the platform that containerd is running on
func getLibExt() string {
	switch runtime.GOOS {
	case "windows":
		return "dll"
	default:
		return "so"
	}
}

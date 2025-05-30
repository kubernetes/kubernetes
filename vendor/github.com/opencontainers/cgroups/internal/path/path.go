package path

import (
	"errors"
	"os"
	"path/filepath"

	"github.com/opencontainers/cgroups"
)

// Inner returns a path to cgroup relative to a cgroup mount point, based
// on cgroup configuration, or an error, if cgroup configuration is invalid.
// To be used only by fs cgroup managers (systemd has different path rules).
func Inner(c *cgroups.Cgroup) (string, error) {
	if (c.Name != "" || c.Parent != "") && c.Path != "" {
		return "", errors.New("cgroup: either Path or Name and Parent should be used")
	}

	// XXX: Do not remove cleanPath. Path safety is important! -- cyphar
	innerPath := cleanPath(c.Path)
	if innerPath == "" {
		cgParent := cleanPath(c.Parent)
		cgName := cleanPath(c.Name)
		innerPath = filepath.Join(cgParent, cgName)
	}

	return innerPath, nil
}

// cleanPath is a copy of github.com/opencontainers/runc/libcontainer/utils.CleanPath.
func cleanPath(path string) string {
	// Deal with empty strings nicely.
	if path == "" {
		return ""
	}

	// Ensure that all paths are cleaned (especially problematic ones like
	// "/../../../../../" which can cause lots of issues).

	if filepath.IsAbs(path) {
		return filepath.Clean(path)
	}

	// If the path isn't absolute, we need to do more processing to fix paths
	// such as "../../../../<etc>/some/path". We also shouldn't convert absolute
	// paths to relative ones.
	path = filepath.Clean(string(os.PathSeparator) + path)
	// This can't fail, as (by definition) all paths are relative to root.
	path, _ = filepath.Rel(string(os.PathSeparator), path)

	return path
}

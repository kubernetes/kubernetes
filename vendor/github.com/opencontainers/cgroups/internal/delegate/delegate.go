// Package delegate implements cgroup v2 delegation to a non-root user.
package delegate

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path/filepath"
)

// Chown delegates the cgroup v2 directory at path to the given uid,
// by changing the owner of the directory itself, and of the files
// within it which the kernel considers delegatable.
func Chown(path string, uid int) error {
	if err := os.Chown(path, uid, -1); err != nil {
		return err
	}

	files, err := filesToChown()
	if err != nil {
		return err
	}

	for _, f := range files {
		err := os.Chown(filepath.Join(path, f), uid, -1)
		// Some files might not be present.
		if err != nil && !errors.Is(err, os.ErrNotExist) {
			return err
		}
	}

	return nil
}

// The kernel exposes a list of files that should be chowned to the delegate
// uid in /sys/kernel/cgroup/delegate.  If the file is not present
// (Linux < 4.15), use the initial values mentioned in cgroups(7).
func filesToChown() ([]string, error) {
	const cgroupDelegateFile = "/sys/kernel/cgroup/delegate"

	f, err := os.Open(cgroupDelegateFile)
	if err != nil {
		return []string{"cgroup.procs", "cgroup.subtree_control", "cgroup.threads"}, nil
	}
	defer f.Close()

	files := []string{}
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		files = append(files, scanner.Text())
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("error reading %s: %w", cgroupDelegateFile, err)
	}

	return files, nil
}

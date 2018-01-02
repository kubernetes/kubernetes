// +build linux freebsd darwin solaris

package volume

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"

	mounttypes "github.com/docker/docker/api/types/mount"
)

var platformRawValidationOpts = []func(o *validateOpts){
	// need to make sure to not error out if the bind source does not exist on unix
	// this is supported for historical reasons, the path will be automatically
	// created later.
	func(o *validateOpts) { o.skipBindSourceCheck = true },
}

// read-write modes
var rwModes = map[string]bool{
	"rw": true,
	"ro": true,
}

// label modes
var labelModes = map[string]bool{
	"Z": true,
	"z": true,
}

// consistency modes
var consistencyModes = map[mounttypes.Consistency]bool{
	mounttypes.ConsistencyFull:      true,
	mounttypes.ConsistencyCached:    true,
	mounttypes.ConsistencyDelegated: true,
}

// BackwardsCompatible decides whether this mount point can be
// used in old versions of Docker or not.
// Only bind mounts and local volumes can be used in old versions of Docker.
func (m *MountPoint) BackwardsCompatible() bool {
	return len(m.Source) > 0 || m.Driver == DefaultDriverName
}

// HasResource checks whether the given absolute path for a container is in
// this mount point. If the relative path starts with `../` then the resource
// is outside of this mount point, but we can't simply check for this prefix
// because it misses `..` which is also outside of the mount, so check both.
func (m *MountPoint) HasResource(absolutePath string) bool {
	relPath, err := filepath.Rel(m.Destination, absolutePath)
	return err == nil && relPath != ".." && !strings.HasPrefix(relPath, fmt.Sprintf("..%c", filepath.Separator))
}

// IsVolumeNameValid checks a volume name in a platform specific manner.
func IsVolumeNameValid(name string) (bool, error) {
	return true, nil
}

// ValidMountMode will make sure the mount mode is valid.
// returns if it's a valid mount mode or not.
func ValidMountMode(mode string) bool {
	if mode == "" {
		return true
	}

	rwModeCount := 0
	labelModeCount := 0
	propagationModeCount := 0
	copyModeCount := 0
	consistencyModeCount := 0

	for _, o := range strings.Split(mode, ",") {
		switch {
		case rwModes[o]:
			rwModeCount++
		case labelModes[o]:
			labelModeCount++
		case propagationModes[mounttypes.Propagation(o)]:
			propagationModeCount++
		case copyModeExists(o):
			copyModeCount++
		case consistencyModes[mounttypes.Consistency(o)]:
			consistencyModeCount++
		default:
			return false
		}
	}

	// Only one string for each mode is allowed.
	if rwModeCount > 1 || labelModeCount > 1 || propagationModeCount > 1 || copyModeCount > 1 || consistencyModeCount > 1 {
		return false
	}
	return true
}

// ReadWrite tells you if a mode string is a valid read-write mode or not.
// If there are no specifications w.r.t read write mode, then by default
// it returns true.
func ReadWrite(mode string) bool {
	if !ValidMountMode(mode) {
		return false
	}

	for _, o := range strings.Split(mode, ",") {
		if o == "ro" {
			return false
		}
	}
	return true
}

func validateNotRoot(p string) error {
	p = filepath.Clean(convertSlash(p))
	if p == "/" {
		return fmt.Errorf("invalid specification: destination can't be '/'")
	}
	return nil
}

func validateCopyMode(mode bool) error {
	return nil
}

func convertSlash(p string) string {
	return filepath.ToSlash(p)
}

func splitRawSpec(raw string) ([]string, error) {
	if strings.Count(raw, ":") > 2 {
		return nil, errInvalidSpec(raw)
	}

	arr := strings.SplitN(raw, ":", 3)
	if arr[0] == "" {
		return nil, errInvalidSpec(raw)
	}
	return arr, nil
}

func clean(p string) string {
	return filepath.Clean(p)
}

func validateStat(fi os.FileInfo) error {
	return nil
}

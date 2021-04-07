package mountinfo

import (
	"os"
)

// GetMounts retrieves a list of mounts for the current running process,
// with an optional filter applied (use nil for no filter).
func GetMounts(f FilterFunc) ([]*Info, error) {
	return parseMountTable(f)
}

// Mounted determines if a specified path is a mount point.
//
// The argument must be an absolute path, with all symlinks resolved, and clean.
// One way to ensure it is to process the path using filepath.Abs followed by
// filepath.EvalSymlinks before calling this function.
func Mounted(path string) (bool, error) {
	// root is always mounted
	if path == string(os.PathSeparator) {
		return true, nil
	}
	return mounted(path)
}

// Info reveals information about a particular mounted filesystem. This
// struct is populated from the content in the /proc/<pid>/mountinfo file.
type Info struct {
	// ID is a unique identifier of the mount (may be reused after umount).
	ID int

	// Parent indicates the ID of the mount parent (or of self for the top of the
	// mount tree).
	Parent int

	// Major indicates one half of the device ID which identifies the device class.
	Major int

	// Minor indicates one half of the device ID which identifies a specific
	// instance of device.
	Minor int

	// Root of the mount within the filesystem.
	Root string

	// Mountpoint indicates the mount point relative to the process's root.
	Mountpoint string

	// Options represents mount-specific options.
	Options string

	// Optional represents optional fields.
	Optional string

	// FSType indicates the type of filesystem, such as EXT3.
	FSType string

	// Source indicates filesystem specific information or "none".
	Source string

	// VFSOptions represents per super block options.
	VFSOptions string
}

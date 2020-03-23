package mount // import "github.com/docker/docker/pkg/mount"

import (
	"sort"
	"strconv"
	"strings"

	"github.com/sirupsen/logrus"
)

// mountError records an error from mount or unmount operation
type mountError struct {
	op             string
	source, target string
	flags          uintptr
	data           string
	err            error
}

func (e *mountError) Error() string {
	out := e.op + " "

	if e.source != "" {
		out += e.source + ":" + e.target
	} else {
		out += e.target
	}

	if e.flags != uintptr(0) {
		out += ", flags: 0x" + strconv.FormatUint(uint64(e.flags), 16)
	}
	if e.data != "" {
		out += ", data: " + e.data
	}

	out += ": " + e.err.Error()
	return out
}

// Cause returns the underlying cause of the error
func (e *mountError) Cause() error {
	return e.err
}

// FilterFunc is a type defining a callback function
// to filter out unwanted entries. It takes a pointer
// to an Info struct (not fully populated, currently
// only Mountpoint is filled in), and returns two booleans:
//  - skip: true if the entry should be skipped
//  - stop: true if parsing should be stopped after the entry
type FilterFunc func(*Info) (skip, stop bool)

// PrefixFilter discards all entries whose mount points
// do not start with a prefix specified
func PrefixFilter(prefix string) FilterFunc {
	return func(m *Info) (bool, bool) {
		skip := !strings.HasPrefix(m.Mountpoint, prefix)
		return skip, false
	}
}

// SingleEntryFilter looks for a specific entry
func SingleEntryFilter(mp string) FilterFunc {
	return func(m *Info) (bool, bool) {
		if m.Mountpoint == mp {
			return false, true // don't skip, stop now
		}
		return true, false // skip, keep going
	}
}

// ParentsFilter returns all entries whose mount points
// can be parents of a path specified, discarding others.
// For example, given `/var/lib/docker/something`, entries
// like `/var/lib/docker`, `/var` and `/` are returned.
func ParentsFilter(path string) FilterFunc {
	return func(m *Info) (bool, bool) {
		skip := !strings.HasPrefix(path, m.Mountpoint)
		return skip, false
	}
}

// GetMounts retrieves a list of mounts for the current running process,
// with an optional filter applied (use nil for no filter).
func GetMounts(f FilterFunc) ([]*Info, error) {
	return parseMountTable(f)
}

// Mounted determines if a specified mountpoint has been mounted.
// On Linux it looks at /proc/self/mountinfo.
func Mounted(mountpoint string) (bool, error) {
	entries, err := GetMounts(SingleEntryFilter(mountpoint))
	if err != nil {
		return false, err
	}

	return len(entries) > 0, nil
}

// Mount will mount filesystem according to the specified configuration, on the
// condition that the target path is *not* already mounted. Options must be
// specified like the mount or fstab unix commands: "opt1=val1,opt2=val2". See
// flags.go for supported option flags.
func Mount(device, target, mType, options string) error {
	flag, _ := parseOptions(options)
	if flag&REMOUNT != REMOUNT {
		if mounted, err := Mounted(target); err != nil || mounted {
			return err
		}
	}
	return ForceMount(device, target, mType, options)
}

// ForceMount will mount a filesystem according to the specified configuration,
// *regardless* if the target path is not already mounted. Options must be
// specified like the mount or fstab unix commands: "opt1=val1,opt2=val2". See
// flags.go for supported option flags.
func ForceMount(device, target, mType, options string) error {
	flag, data := parseOptions(options)
	return mount(device, target, mType, uintptr(flag), data)
}

// Unmount lazily unmounts a filesystem on supported platforms, otherwise
// does a normal unmount.
func Unmount(target string) error {
	return unmount(target, mntDetach)
}

// RecursiveUnmount unmounts the target and all mounts underneath, starting with
// the deepsest mount first.
func RecursiveUnmount(target string) error {
	mounts, err := parseMountTable(PrefixFilter(target))
	if err != nil {
		return err
	}

	// Make the deepest mount be first
	sort.Slice(mounts, func(i, j int) bool {
		return len(mounts[i].Mountpoint) > len(mounts[j].Mountpoint)
	})

	for i, m := range mounts {
		logrus.Debugf("Trying to unmount %s", m.Mountpoint)
		err = unmount(m.Mountpoint, mntDetach)
		if err != nil {
			if i == len(mounts)-1 { // last mount
				if mounted, e := Mounted(m.Mountpoint); e != nil || mounted {
					return err
				}
			} else {
				// This is some submount, we can ignore this error for now, the final unmount will fail if this is a real problem
				logrus.WithError(err).Warnf("Failed to unmount submount %s", m.Mountpoint)
			}
		}

		logrus.Debugf("Unmounted %s", m.Mountpoint)
	}
	return nil
}

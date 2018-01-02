package mount

import (
	"strings"

	"golang.org/x/sys/unix"
)

// Mount to the provided target path
func (m *Mount) Mount(target string) error {
	flags, data := parseMountOptions(m.Options)

	// propagation types.
	const ptypes = unix.MS_SHARED | unix.MS_PRIVATE | unix.MS_SLAVE | unix.MS_UNBINDABLE

	// Ensure propagation type change flags aren't included in other calls.
	oflags := flags &^ ptypes

	// In the case of remounting with changed data (data != ""), need to call mount (moby/moby#34077).
	if flags&unix.MS_REMOUNT == 0 || data != "" {
		// Initial call applying all non-propagation flags for mount
		// or remount with changed data
		if err := unix.Mount(m.Source, target, m.Type, uintptr(oflags), data); err != nil {
			return err
		}
	}

	if flags&ptypes != 0 {
		// Change the propagation type.
		const pflags = ptypes | unix.MS_REC | unix.MS_SILENT
		if err := unix.Mount("", target, "", uintptr(flags&pflags), ""); err != nil {
			return err
		}
	}

	const broflags = unix.MS_BIND | unix.MS_RDONLY
	if oflags&broflags == broflags {
		// Remount the bind to apply read only.
		return unix.Mount("", target, "", uintptr(oflags|unix.MS_REMOUNT), "")
	}
	return nil
}

// Unmount the provided mount path with the flags
func Unmount(mount string, flags int) error {
	return unix.Unmount(mount, flags)
}

// UnmountAll repeatedly unmounts the given mount point until there
// are no mounts remaining (EINVAL is returned by mount), which is
// useful for undoing a stack of mounts on the same mount point.
func UnmountAll(mount string, flags int) error {
	for {
		if err := Unmount(mount, flags); err != nil {
			// EINVAL is returned if the target is not a
			// mount point, indicating that we are
			// done. It can also indicate a few other
			// things (such as invalid flags) which we
			// unfortunately end up squelching here too.
			if err == unix.EINVAL {
				return nil
			}
			return err
		}
	}
}

// parseMountOptions takes fstab style mount options and parses them for
// use with a standard mount() syscall
func parseMountOptions(options []string) (int, string) {
	var (
		flag int
		data []string
	)
	flags := map[string]struct {
		clear bool
		flag  int
	}{
		"async":         {true, unix.MS_SYNCHRONOUS},
		"atime":         {true, unix.MS_NOATIME},
		"bind":          {false, unix.MS_BIND},
		"defaults":      {false, 0},
		"dev":           {true, unix.MS_NODEV},
		"diratime":      {true, unix.MS_NODIRATIME},
		"dirsync":       {false, unix.MS_DIRSYNC},
		"exec":          {true, unix.MS_NOEXEC},
		"mand":          {false, unix.MS_MANDLOCK},
		"noatime":       {false, unix.MS_NOATIME},
		"nodev":         {false, unix.MS_NODEV},
		"nodiratime":    {false, unix.MS_NODIRATIME},
		"noexec":        {false, unix.MS_NOEXEC},
		"nomand":        {true, unix.MS_MANDLOCK},
		"norelatime":    {true, unix.MS_RELATIME},
		"nostrictatime": {true, unix.MS_STRICTATIME},
		"nosuid":        {false, unix.MS_NOSUID},
		"rbind":         {false, unix.MS_BIND | unix.MS_REC},
		"relatime":      {false, unix.MS_RELATIME},
		"remount":       {false, unix.MS_REMOUNT},
		"ro":            {false, unix.MS_RDONLY},
		"rw":            {true, unix.MS_RDONLY},
		"strictatime":   {false, unix.MS_STRICTATIME},
		"suid":          {true, unix.MS_NOSUID},
		"sync":          {false, unix.MS_SYNCHRONOUS},
	}
	for _, o := range options {
		// If the option does not exist in the flags table or the flag
		// is not supported on the platform,
		// then it is a data value for a specific fs type
		if f, exists := flags[o]; exists && f.flag != 0 {
			if f.clear {
				flag &^= f.flag
			} else {
				flag |= f.flag
			}
		} else {
			data = append(data, o)
		}
	}
	return flag, strings.Join(data, ",")
}

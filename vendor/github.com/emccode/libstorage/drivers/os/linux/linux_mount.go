// +build linux

package linux

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strings"
	"syscall"
	"time"

	"github.com/emccode/libstorage/api/types"
)

const (
	/* 36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
	   (1)(2)(3)   (4)   (5)      (6)      (7)   (8) (9)   (10)         (11)
	   (1) mount ID:  unique identifier of the mount (may be reused after umount)
	   (2) parent ID:  ID of parent (or of self for the top of the mount tree)
	   (3) major:minor:  value of st_dev for files on filesystem
	   (4) root:  root of the mount within the filesystem
	   (5) mount point:  mount point relative to the process's root
	   (6) mount options:  per mount options
	   (7) optional fields:  zero or more fields of the form "tag[:value]"
	   (8) separator:  marks the end of the optional fields
	   (9) filesystem type:  name of filesystem of the form "type[.subtype]"
	   (10) mount source:  filesystem specific information or "none"
	   (11) super options:  per super block options*/
	mountinfoFormat = "%d %d %d:%d %s %s %s %s"
)

const (
	// RDONLY will mount the file system read-only.
	RDONLY = syscall.MS_RDONLY

	// NOSUID will not allow set-user-identifier or set-group-identifier bits to
	// take effect.
	NOSUID = syscall.MS_NOSUID

	// NODEV will not interpret character or block special devices on the file
	// system.
	NODEV = syscall.MS_NODEV

	// NOEXEC will not allow execution of any binaries on the mounted file system.
	NOEXEC = syscall.MS_NOEXEC

	// SYNCHRONOUS will allow I/O to the file system to be done synchronously.
	SYNCHRONOUS = syscall.MS_SYNCHRONOUS

	// DIRSYNC will force all directory updates within the file system to be done
	// synchronously. This affects the following system calls: create, link,
	// unlink, symlink, mkdir, rmdir, mknod and rename.
	DIRSYNC = syscall.MS_DIRSYNC

	// REMOUNT will attempt to remount an already-mounted file system. This is
	// commonly used to change the mount flags for a file system, especially to
	// make a readonly file system writeable. It does not change device or mount
	// point.
	REMOUNT = syscall.MS_REMOUNT

	// MANDLOCK will force mandatory locks on a filesystem.
	MANDLOCK = syscall.MS_MANDLOCK

	// NOATIME will not update the file access time when reading from a file.
	NOATIME = syscall.MS_NOATIME

	// NODIRATIME will not update the directory access time.
	NODIRATIME = syscall.MS_NODIRATIME

	// BIND remounts a subtree somewhere else.
	BIND = syscall.MS_BIND

	// RBIND remounts a subtree and all possible submounts somewhere else.
	RBIND = syscall.MS_BIND | syscall.MS_REC

	// UNBINDABLE creates a mount which cannot be cloned through a bind operation.
	UNBINDABLE = syscall.MS_UNBINDABLE

	// RUNBINDABLE marks the entire mount tree as UNBINDABLE.
	RUNBINDABLE = syscall.MS_UNBINDABLE | syscall.MS_REC

	// PRIVATE creates a mount which carries no propagation abilities.
	PRIVATE = syscall.MS_PRIVATE

	// RPRIVATE marks the entire mount tree as PRIVATE.
	RPRIVATE = syscall.MS_PRIVATE | syscall.MS_REC

	// SLAVE creates a mount which receives propagation from its master, but not
	// vice versa.
	SLAVE = syscall.MS_SLAVE

	// RSLAVE marks the entire mount tree as SLAVE.
	RSLAVE = syscall.MS_SLAVE | syscall.MS_REC

	// SHARED creates a mount which provides the ability to create mirrors of
	// that mount such that mounts and unmounts within any of the mirrors
	// propagate to the other mirrors.
	SHARED = syscall.MS_SHARED

	// RSHARED marks the entire mount tree as SHARED.
	RSHARED = syscall.MS_SHARED | syscall.MS_REC

	// RELATIME updates inode access times relative to modify or change time.
	RELATIME = syscall.MS_RELATIME

	// STRICTATIME allows to explicitly request full atime updates.  This makes
	// it possible for the kernel to default to relatime or noatime but still
	// allow userspace to override it.
	STRICTATIME = syscall.MS_STRICTATIME
)

// Parse /proc/self/mountinfo because comparing Dev and ino does not work from
// bind mounts
func parseMountTable() ([]*types.MountInfo, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseInfoFile(f)
}

func parseInfoFile(r io.Reader) ([]*types.MountInfo, error) {
	var (
		s   = bufio.NewScanner(r)
		out = []*types.MountInfo{}
	)

	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}

		var (
			p              = &types.MountInfo{}
			text           = s.Text()
			optionalFields string
		)

		if _, err := fmt.Sscanf(text, mountinfoFormat,
			&p.ID, &p.Parent, &p.Major, &p.Minor,
			&p.Root, &p.MountPoint, &p.Opts, &optionalFields); err != nil {
			return nil, fmt.Errorf("Scanning '%s' failed: %s", text, err)
		}
		// Safe as mountinfo encodes mountpoints with spaces as \040.
		index := strings.Index(text, " - ")
		postSeparatorFields := strings.Fields(text[index+3:])
		if len(postSeparatorFields) < 3 {
			return nil, fmt.Errorf("Error found less than 3 fields post '-' in %q", text)
		}

		if optionalFields != "-" {
			p.Optional = optionalFields
		}

		p.FSType = postSeparatorFields[0]
		p.Source = postSeparatorFields[1]
		p.VFSOpts = strings.Join(postSeparatorFields[2:], " ")
		out = append(out, p)
	}
	return out, nil
}

// pidMountInfo collects the mounts for a specific process ID. If the process
// ID is unknown, it is better to use `GetMounts` which will inspect
// "/proc/self/mountinfo" instead.
func pidMountInfo(pid int) ([]*types.MountInfo, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/mountinfo", pid))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseInfoFile(f)
}

// parseOptions parses fstab type mount options into mount() flags
// and device specific data
func parseOptions(options string) (int, string) {
	var (
		flag int
		data []string
	)

	flags := map[string]struct {
		clear bool
		flag  int
	}{
		"defaults":      {false, 0},
		"ro":            {false, RDONLY},
		"rw":            {true, RDONLY},
		"suid":          {true, NOSUID},
		"nosuid":        {false, NOSUID},
		"dev":           {true, NODEV},
		"nodev":         {false, NODEV},
		"exec":          {true, NOEXEC},
		"noexec":        {false, NOEXEC},
		"sync":          {false, SYNCHRONOUS},
		"async":         {true, SYNCHRONOUS},
		"dirsync":       {false, DIRSYNC},
		"remount":       {false, REMOUNT},
		"mand":          {false, MANDLOCK},
		"nomand":        {true, MANDLOCK},
		"atime":         {true, NOATIME},
		"noatime":       {false, NOATIME},
		"diratime":      {true, NODIRATIME},
		"nodiratime":    {false, NODIRATIME},
		"bind":          {false, BIND},
		"rbind":         {false, RBIND},
		"unbindable":    {false, UNBINDABLE},
		"runbindable":   {false, RUNBINDABLE},
		"private":       {false, PRIVATE},
		"rprivate":      {false, RPRIVATE},
		"shared":        {false, SHARED},
		"rshared":       {false, RSHARED},
		"slave":         {false, SLAVE},
		"rslave":        {false, RSLAVE},
		"relatime":      {false, RELATIME},
		"norelatime":    {true, RELATIME},
		"strictatime":   {false, STRICTATIME},
		"nostrictatime": {true, STRICTATIME},
	}

	for _, o := range strings.Split(options, ",") {
		// If the option does not exist in the flags table or the flag
		// is not supported on the platform,
		// then it is a data value for a specific fs type
		if f, exists := flags[o]; exists && f.flag != 0 {
			if f.clear {
				flag &= ^f.flag
			} else {
				flag |= f.flag
			}
		} else {
			data = append(data, o)
		}
	}
	return flag, strings.Join(data, ",")
}

// parseTmpfsOptions parse fstab type mount options into flags and data
func parseTmpfsOptions(options string) (int, string, error) {
	flags, data := parseOptions(options)
	validFlags := map[string]bool{
		"":          true,
		"size":      true,
		"mode":      true,
		"uid":       true,
		"gid":       true,
		"nr_inodes": true,
		"nr_blocks": true,
		"mpol":      true,
	}
	for _, o := range strings.Split(data, ",") {
		opt := strings.SplitN(o, "=", 2)
		if !validFlags[opt[0]] {
			return 0, "", fmt.Errorf("Invalid tmpfs option %q", opt)
		}
	}
	return flags, data, nil
}

// getMounts retrieves a list of mounts for the current running process.
func getMounts() ([]*types.MountInfo, error) {
	return parseMountTable()
}

// Mounted looks at /proc/self/mountinfo to determine of the specified
// mountpoint has been mounted
func mounted(mountpoint string) (bool, error) {
	entries, err := parseMountTable()
	if err != nil {
		return false, err
	}

	// Search the table for the mountpoint
	for _, e := range entries {
		if e.MountPoint == mountpoint {
			return true, nil
		}
	}
	return false, nil
}

// mount will mount filesystem according to the specified configuration, on the
// condition that the target path is *not* already mounted. Options must be
// specified like the mount or fstab unix commands: "opt1=val1,opt2=val2". See
// flags.go for supported option flags.
func mount(device, target, mType, options string) error {
	flag, _ := parseOptions(options)
	if flag&REMOUNT != REMOUNT {
		if mounted, err := mounted(target); err != nil || mounted {
			return err
		}
	}
	return forceMount(device, target, mType, options)
}

// sysMount will mount filesystem according to the specified configuration, on
// the condition that the target path is *not* already mounted. Options must be
// specified like the mount or fstab unix commands: "opt1=val1,opt2=val2". See
// flags.go for supported option flags.
func sysMount(device, target, mType string, flag uintptr, data string) error {
	if err := syscall.Mount(device, target, mType, flag, data); err != nil {
		return err
	}

	// If we have a bind mount or remount, remount...
	if flag&syscall.MS_BIND == syscall.MS_BIND &&
		flag&syscall.MS_RDONLY == syscall.MS_RDONLY {
		return syscall.Mount(
			device, target, mType, flag|syscall.MS_REMOUNT, data)
	}
	return nil
}

// forceMount will mount a filesystem according to the specified configuration,
// *regardless* if the target path is not already mounted. Options must be
// specified like the mount or fstab unix commands: "opt1=val1,opt2=val2". See
// flags.go for supported option flags.
func forceMount(device, target, mType, options string) error {
	flag, data := parseOptions(options)
	err := sysMount(device, target, mType, uintptr(flag), data)
	if err != nil {
		return err
	}
	return nil
}

// unmount will unmount the target filesystem, so long as it is mounted.
func unmount(target string) error {
	if mounted, err := mounted(target); err != nil || !mounted {
		return err
	}
	return forceUnmount(target)
}

func sysUnmount(target string, flag int) error {
	return syscall.Unmount(target, flag)
}

// forceUnmount will force an unmount of the target filesystem, regardless if
// it is mounted or not.
func forceUnmount(target string) (err error) {
	// Simple retry logic for unmount
	for i := 0; i < 10; i++ {
		if err = sysUnmount(target, 0); err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return
}

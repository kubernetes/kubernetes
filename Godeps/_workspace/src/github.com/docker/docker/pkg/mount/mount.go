package mount

import (
	"time"
)

func GetMounts() ([]*MountInfo, error) {
	return parseMountTable()
}

// Looks at /proc/self/mountinfo to determine of the specified
// mountpoint has been mounted
func Mounted(mountpoint string) (bool, error) {
	entries, err := parseMountTable()
	if err != nil {
		return false, err
	}

	// Search the table for the mountpoint
	for _, e := range entries {
		if e.Mountpoint == mountpoint {
			return true, nil
		}
	}
	return false, nil
}

// Mount the specified options at the target path only if
// the target is not mounted
// Options must be specified as fstab style
func Mount(device, target, mType, options string) error {
	flag, _ := parseOptions(options)
	if flag&REMOUNT != REMOUNT {
		if mounted, err := Mounted(target); err != nil || mounted {
			return err
		}
	}
	return ForceMount(device, target, mType, options)
}

// Mount the specified options at the target path
// reguardless if the target is mounted or not
// Options must be specified as fstab style
func ForceMount(device, target, mType, options string) error {
	flag, data := parseOptions(options)
	if err := mount(device, target, mType, uintptr(flag), data); err != nil {
		return err
	}
	return nil
}

// Unmount the target only if it is mounted
func Unmount(target string) error {
	if mounted, err := Mounted(target); err != nil || !mounted {
		return err
	}
	return ForceUnmount(target)
}

// Unmount the target reguardless if it is mounted or not
func ForceUnmount(target string) (err error) {
	// Simple retry logic for unmount
	for i := 0; i < 10; i++ {
		if err = unmount(target, 0); err == nil {
			return nil
		}
		time.Sleep(100 * time.Millisecond)
	}
	return
}

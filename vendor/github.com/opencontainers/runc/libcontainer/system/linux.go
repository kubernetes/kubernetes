// +build linux

package system

import (
	"os"
	"os/exec"
	"sync"
	"unsafe"

	"github.com/opencontainers/runc/libcontainer/user"
	"golang.org/x/sys/unix"
)

type ParentDeathSignal int

func (p ParentDeathSignal) Restore() error {
	if p == 0 {
		return nil
	}
	current, err := GetParentDeathSignal()
	if err != nil {
		return err
	}
	if p == current {
		return nil
	}
	return p.Set()
}

func (p ParentDeathSignal) Set() error {
	return SetParentDeathSignal(uintptr(p))
}

func Execv(cmd string, args []string, env []string) error {
	name, err := exec.LookPath(cmd)
	if err != nil {
		return err
	}

	return unix.Exec(name, args, env)
}

func Prlimit(pid, resource int, limit unix.Rlimit) error {
	_, _, err := unix.RawSyscall6(unix.SYS_PRLIMIT64, uintptr(pid), uintptr(resource), uintptr(unsafe.Pointer(&limit)), uintptr(unsafe.Pointer(&limit)), 0, 0)
	if err != 0 {
		return err
	}
	return nil
}

func SetParentDeathSignal(sig uintptr) error {
	if err := unix.Prctl(unix.PR_SET_PDEATHSIG, sig, 0, 0, 0); err != nil {
		return err
	}
	return nil
}

func GetParentDeathSignal() (ParentDeathSignal, error) {
	var sig int
	if err := unix.Prctl(unix.PR_GET_PDEATHSIG, uintptr(unsafe.Pointer(&sig)), 0, 0, 0); err != nil {
		return -1, err
	}
	return ParentDeathSignal(sig), nil
}

func SetKeepCaps() error {
	if err := unix.Prctl(unix.PR_SET_KEEPCAPS, 1, 0, 0, 0); err != nil {
		return err
	}

	return nil
}

func ClearKeepCaps() error {
	if err := unix.Prctl(unix.PR_SET_KEEPCAPS, 0, 0, 0, 0); err != nil {
		return err
	}

	return nil
}

func Setctty() error {
	if err := unix.IoctlSetInt(0, unix.TIOCSCTTY, 0); err != nil {
		return err
	}
	return nil
}

var (
	inUserNS bool
	nsOnce   sync.Once
)

// RunningInUserNS detects whether we are currently running in a user namespace.
// Originally copied from github.com/lxc/lxd/shared/util.go
func RunningInUserNS() bool {
	nsOnce.Do(func() {
		uidmap, err := user.CurrentProcessUIDMap()
		if err != nil {
			// This kernel-provided file only exists if user namespaces are supported
			return
		}
		inUserNS = UIDMapInUserNS(uidmap)
	})
	return inUserNS
}

func UIDMapInUserNS(uidmap []user.IDMap) bool {
	/*
	 * We assume we are in the initial user namespace if we have a full
	 * range - 4294967295 uids starting at uid 0.
	 */
	if len(uidmap) == 1 && uidmap[0].ID == 0 && uidmap[0].ParentID == 0 && uidmap[0].Count == 4294967295 {
		return false
	}
	return true
}

// GetParentNSeuid returns the euid within the parent user namespace
func GetParentNSeuid() int64 {
	euid := int64(os.Geteuid())
	uidmap, err := user.CurrentProcessUIDMap()
	if err != nil {
		// This kernel-provided file only exists if user namespaces are supported
		return euid
	}
	for _, um := range uidmap {
		if um.ID <= euid && euid <= um.ID+um.Count-1 {
			return um.ParentID + euid - um.ID
		}
	}
	return euid
}

// SetSubreaper sets the value i as the subreaper setting for the calling process
func SetSubreaper(i int) error {
	return unix.Prctl(unix.PR_SET_CHILD_SUBREAPER, uintptr(i), 0, 0, 0)
}

// GetSubreaper returns the subreaper setting for the calling process
func GetSubreaper() (int, error) {
	var i uintptr

	if err := unix.Prctl(unix.PR_GET_CHILD_SUBREAPER, uintptr(unsafe.Pointer(&i)), 0, 0, 0); err != nil {
		return -1, err
	}

	return int(i), nil
}

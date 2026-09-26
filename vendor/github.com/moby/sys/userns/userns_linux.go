package userns

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"sync"
	"syscall"
)

// See PROC_USER_INIT_INO in https://github.com/torvalds/linux/blob/v7.1/include/uapi/linux/nsfs.h#L50.
const procUserInitIno = 0xEFFFFFFD

var inUserNS = sync.OnceValue(runningInUserNS)

// runningInUserNS detects whether we are currently running in a user namespace.
//
// This code was migrated from [libcontainer/runc] and based on an implementation
// from [lcx/incus].
//
// [libcontainer/runc]: https://github.com/opencontainers/runc/blob/3778ae603c706494fd1e2c2faf83b406e38d687d/libcontainer/userns/userns_linux.go#L12-L49
// [lcx/incus]: https://github.com/lxc/incus/blob/e45085dd42f826b3c8c3228e9733c0b6f998eafe/shared/util.go#L678-L700
func runningInUserNS() bool {
	var st syscall.Stat_t
	if err := syscall.Stat("/proc/self/ns/user", &st); err == nil {
		// The kernel's initial user namespace inode is definitive when it
		// matches. OpenVZ virtualizes namespace inode numbers, so a mismatch
		// must fall back to uid_map-based detection there.
		if st.Ino == procUserInitIno {
			return false
		}
		if runningInOpenVZ() {
			return runningInUserNSFromUIDMap()
		}
		return true
	} else if !os.IsNotExist(err) {
		// As long as /proc/self/ns/user exists, we are on a modern kernel.
		// Other errors indicate an unexpected procfs state, where assuming the
		// init namespace would be unsafe.
		return false
	}

	// Only fall back for older kernels that do not expose the user namespace
	// through procfs at /proc/self/ns/user.
	// TODO: Remove this fallback once Linux kernels older than 3.8 are no
	// longer supported.
	return runningInUserNSFromUIDMap()
}

func runningInUserNSFromUIDMap() bool {
	file, err := os.Open("/proc/self/uid_map")
	if err != nil {
		// This kernel-provided file only exists if user namespaces are supported.
		return false
	}
	defer file.Close()

	buf := bufio.NewReader(file)
	l, _, err := buf.ReadLine()
	if err != nil {
		return false
	}

	return uidMapInUserNS(string(l))
}

func uidMapInUserNS(uidMap string) bool {
	if uidMap == "" {
		// File exist but empty (the initial state when userns is created,
		// see user_namespaces(7)).
		return true
	}

	var a, b, c int64
	if _, err := fmt.Sscanf(uidMap, "%d %d %d", &a, &b, &c); err != nil {
		// Assume we are in a regular, non user namespace.
		return false
	}

	// As per user_namespaces(7), /proc/self/uid_map of
	// the initial user namespace shows 0 0 4294967295.
	initNS := a == 0 && b == 0 && c == 4294967295
	return !initNS
}

// runningInOpenVZ reports whether the process is running inside an OpenVZ
// container.
//
// OpenVZ exposes /proc/vz both on the host and inside containers, while
// /proc/bc is only exposed on the host. This follows systemd's OpenVZ
// detection:
// https://github.com/systemd/systemd/blob/v261.2/src/basic/virt.c#L642-L653
func runningInOpenVZ() bool {
	var st syscall.Stat_t
	if err := syscall.Stat("/proc/vz", &st); err != nil {
		return false
	}
	err := syscall.Stat("/proc/bc", &st)
	return errors.Is(err, syscall.ENOENT)
}

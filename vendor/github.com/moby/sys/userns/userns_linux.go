package userns

import (
	"bufio"
	"fmt"
	"os"
	"sync"
)

var inUserNS = sync.OnceValue(runningInUserNS)

// runningInUserNS detects whether we are currently running in a user namespace.
//
// This code was migrated from [libcontainer/runc] and based on an implementation
// from [lcx/incus].
//
// [libcontainer/runc]: https://github.com/opencontainers/runc/blob/3778ae603c706494fd1e2c2faf83b406e38d687d/libcontainer/userns/userns_linux.go#L12-L49
// [lcx/incus]: https://github.com/lxc/incus/blob/e45085dd42f826b3c8c3228e9733c0b6f998eafe/shared/util.go#L678-L700
func runningInUserNS() bool {
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

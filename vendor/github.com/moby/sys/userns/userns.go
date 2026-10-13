// Package userns provides utilities to detect whether we are currently running
// in a Linux user namespace.
//
// This code was migrated from [libcontainer/runc], which based its implementation
// on code from [lcx/incus].
//
// [libcontainer/runc]: https://github.com/opencontainers/runc/blob/3778ae603c706494fd1e2c2faf83b406e38d687d/libcontainer/userns/userns_linux.go#L12-L49
// [lcx/incus]: https://github.com/lxc/incus/blob/e45085dd42f826b3c8c3228e9733c0b6f998eafe/shared/util.go#L678-L700
package userns

// RunningInUserNS detects whether we are currently running in a Linux
// user namespace and memoizes the result. It returns false on non-Linux
// platforms.
func RunningInUserNS() bool {
	return inUserNS()
}

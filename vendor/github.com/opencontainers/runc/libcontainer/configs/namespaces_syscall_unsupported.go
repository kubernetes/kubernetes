// +build !linux,!windows

package configs

func (n *Namespace) Syscall() int {
	panic("No namespace syscall support")
	return 0
}

// CloneFlags parses the container's Namespaces options to set the correct
// flags on clone, unshare. This function returns flags only for new namespaces.
func (n *Namespaces) CloneFlags() uintptr {
	panic("No namespace syscall support")
	return uintptr(0)
}

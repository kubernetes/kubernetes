// +build linux

package configs

import "syscall"

func (n *Namespace) Syscall() int {
	return namespaceInfo[n.Type]
}

var namespaceInfo = map[NamespaceType]int{
	NEWNET:  syscall.CLONE_NEWNET,
	NEWNS:   syscall.CLONE_NEWNS,
	NEWUSER: syscall.CLONE_NEWUSER,
	NEWIPC:  syscall.CLONE_NEWIPC,
	NEWUTS:  syscall.CLONE_NEWUTS,
	NEWPID:  syscall.CLONE_NEWPID,
}

// CloneFlags parses the container's Namespaces options to set the correct
// flags on clone, unshare. This functions returns flags only for new namespaces.
func (n *Namespaces) CloneFlags() uintptr {
	var flag int
	for _, v := range *n {
		if v.Path != "" {
			continue
		}
		flag |= namespaceInfo[v.Type]
	}
	return uintptr(flag)
}

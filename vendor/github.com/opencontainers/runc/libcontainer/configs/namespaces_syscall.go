// +build linux

package configs

import "golang.org/x/sys/unix"

func (n *Namespace) Syscall() int {
	return namespaceInfo[n.Type]
}

var namespaceInfo = map[NamespaceType]int{
	NEWNET:    unix.CLONE_NEWNET,
	NEWNS:     unix.CLONE_NEWNS,
	NEWUSER:   unix.CLONE_NEWUSER,
	NEWIPC:    unix.CLONE_NEWIPC,
	NEWUTS:    unix.CLONE_NEWUTS,
	NEWPID:    unix.CLONE_NEWPID,
	NEWCGROUP: unix.CLONE_NEWCGROUP,
}

// CloneFlags parses the container's Namespaces options to set the correct
// flags on clone, unshare. This function returns flags only for new namespaces.
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

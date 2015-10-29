// +build linux freebsd

package configs

import "fmt"

const (
	NEWNET  NamespaceType = "NEWNET"
	NEWPID  NamespaceType = "NEWPID"
	NEWNS   NamespaceType = "NEWNS"
	NEWUTS  NamespaceType = "NEWUTS"
	NEWIPC  NamespaceType = "NEWIPC"
	NEWUSER NamespaceType = "NEWUSER"
)

func NamespaceTypes() []NamespaceType {
	return []NamespaceType{
		NEWNET,
		NEWPID,
		NEWNS,
		NEWUTS,
		NEWIPC,
		NEWUSER,
	}
}

// Namespace defines configuration for each namespace.  It specifies an
// alternate path that is able to be joined via setns.
type Namespace struct {
	Type NamespaceType `json:"type"`
	Path string        `json:"path"`
}

func (n *Namespace) GetPath(pid int) string {
	if n.Path != "" {
		return n.Path
	}
	return fmt.Sprintf("/proc/%d/ns/%s", pid, n.file())
}

func (n *Namespace) file() string {
	file := ""
	switch n.Type {
	case NEWNET:
		file = "net"
	case NEWNS:
		file = "mnt"
	case NEWPID:
		file = "pid"
	case NEWIPC:
		file = "ipc"
	case NEWUSER:
		file = "user"
	case NEWUTS:
		file = "uts"
	}
	return file
}

func (n *Namespaces) Remove(t NamespaceType) bool {
	i := n.index(t)
	if i == -1 {
		return false
	}
	*n = append((*n)[:i], (*n)[i+1:]...)
	return true
}

func (n *Namespaces) Add(t NamespaceType, path string) {
	i := n.index(t)
	if i == -1 {
		*n = append(*n, Namespace{Type: t, Path: path})
		return
	}
	(*n)[i].Path = path
}

func (n *Namespaces) index(t NamespaceType) int {
	for i, ns := range *n {
		if ns.Type == t {
			return i
		}
	}
	return -1
}

func (n *Namespaces) Contains(t NamespaceType) bool {
	return n.index(t) != -1
}

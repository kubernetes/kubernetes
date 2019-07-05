package configs

import (
	"fmt"
	"os"
	"sync"
)

const (
	NEWNET    NamespaceType = "NEWNET"
	NEWPID    NamespaceType = "NEWPID"
	NEWNS     NamespaceType = "NEWNS"
	NEWUTS    NamespaceType = "NEWUTS"
	NEWIPC    NamespaceType = "NEWIPC"
	NEWUSER   NamespaceType = "NEWUSER"
	NEWCGROUP NamespaceType = "NEWCGROUP"
)

var (
	nsLock              sync.Mutex
	supportedNamespaces = make(map[NamespaceType]bool)
)

// NsName converts the namespace type to its filename
func NsName(ns NamespaceType) string {
	switch ns {
	case NEWNET:
		return "net"
	case NEWNS:
		return "mnt"
	case NEWPID:
		return "pid"
	case NEWIPC:
		return "ipc"
	case NEWUSER:
		return "user"
	case NEWUTS:
		return "uts"
	case NEWCGROUP:
		return "cgroup"
	}
	return ""
}

// IsNamespaceSupported returns whether a namespace is available or
// not
func IsNamespaceSupported(ns NamespaceType) bool {
	nsLock.Lock()
	defer nsLock.Unlock()
	supported, ok := supportedNamespaces[ns]
	if ok {
		return supported
	}
	nsFile := NsName(ns)
	// if the namespace type is unknown, just return false
	if nsFile == "" {
		return false
	}
	_, err := os.Stat(fmt.Sprintf("/proc/self/ns/%s", nsFile))
	// a namespace is supported if it exists and we have permissions to read it
	supported = err == nil
	supportedNamespaces[ns] = supported
	return supported
}

func NamespaceTypes() []NamespaceType {
	return []NamespaceType{
		NEWUSER, // Keep user NS always first, don't move it.
		NEWIPC,
		NEWUTS,
		NEWNET,
		NEWPID,
		NEWNS,
		NEWCGROUP,
	}
}

// Namespace defines configuration for each namespace.  It specifies an
// alternate path that is able to be joined via setns.
type Namespace struct {
	Type NamespaceType `json:"type"`
	Path string        `json:"path"`
}

func (n *Namespace) GetPath(pid int) string {
	return fmt.Sprintf("/proc/%d/ns/%s", pid, NsName(n.Type))
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

func (n *Namespaces) PathOf(t NamespaceType) string {
	i := n.index(t)
	if i == -1 {
		return ""
	}
	return (*n)[i].Path
}

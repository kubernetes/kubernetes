package graphdb

import (
	"path"
	"strings"
)

// Split p on /
func split(p string) []string {
	return strings.Split(p, "/")
}

// PathDepth returns the depth or number of / in a given path
func PathDepth(p string) int {
	parts := split(p)
	if len(parts) == 2 && parts[1] == "" {
		return 1
	}
	return len(parts)
}

func splitPath(p string) (parent, name string) {
	if p[0] != '/' {
		p = "/" + p
	}
	parent, name = path.Split(p)
	l := len(parent)
	if parent[l-1] == '/' {
		parent = parent[:l-1]
	}
	return
}

// +build !linux

package netlink

func (r *Route) ListFlags() []string {
	return []string{}
}

func (n *NexthopInfo) ListFlags() []string {
	return []string{}
}

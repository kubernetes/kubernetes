// +build !linux

package netlink

func (r *Route) ListFlags() []string {
	return []string{}
}

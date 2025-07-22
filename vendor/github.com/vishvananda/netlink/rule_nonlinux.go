//go:build !linux
// +build !linux

package netlink

func (r Rule) typeString() string {
	return ""
}

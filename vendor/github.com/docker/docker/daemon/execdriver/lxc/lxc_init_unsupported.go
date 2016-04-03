// +build !linux

package lxc

func finalizeNamespace(args *InitArgs) error {
	panic("Not supported on this platform")
}

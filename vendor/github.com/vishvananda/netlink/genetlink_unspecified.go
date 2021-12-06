//go:build !linux
// +build !linux

package netlink

type GenlOp struct{}

type GenlMulticastGroup struct{}

type GenlFamily struct{}

func (h *Handle) GenlFamilyList() ([]*GenlFamily, error) {
	return nil, ErrNotImplemented
}

func GenlFamilyList() ([]*GenlFamily, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) GenlFamilyGet(name string) (*GenlFamily, error) {
	return nil, ErrNotImplemented
}

func GenlFamilyGet(name string) (*GenlFamily, error) {
	return nil, ErrNotImplemented
}

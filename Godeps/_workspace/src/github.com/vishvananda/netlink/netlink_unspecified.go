// +build !linux

package netlink

import (
	"errors"
)

var (
	ErrNotImplemented = errors.New("not implemented")
)

func LinkSetUp(link *Link) error {
	return ErrNotImplemented
}

func LinkSetDown(link *Link) error {
	return ErrNotImplemented
}

func LinkSetMTU(link *Link, mtu int) error {
	return ErrNotImplemented
}

func LinkSetMaster(link *Link, master *Link) error {
	return ErrNotImplemented
}

func LinkSetNsPid(link *Link, nspid int) error {
	return ErrNotImplemented
}

func LinkSetNsFd(link *Link, fd int) error {
	return ErrNotImplemented
}

func LinkAdd(link *Link) error {
	return ErrNotImplemented
}

func LinkDel(link *Link) error {
	return ErrNotImplemented
}

func SetHairpin(link Link, mode bool) error {
	return ErrNotImplemented
}

func SetGuard(link Link, mode bool) error {
	return ErrNotImplemented
}

func SetFastLeave(link Link, mode bool) error {
	return ErrNotImplemented
}

func SetLearning(link Link, mode bool) error {
	return ErrNotImplemented
}

func SetRootBlock(link Link, mode bool) error {
	return ErrNotImplemented
}

func SetFlood(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkList() ([]Link, error) {
	return nil, ErrNotImplemented
}

func AddrAdd(link *Link, addr *Addr) error {
	return ErrNotImplemented
}

func AddrDel(link *Link, addr *Addr) error {
	return ErrNotImplemented
}

func AddrList(link *Link, family int) ([]Addr, error) {
	return nil, ErrNotImplemented
}

func RouteAdd(route *Route) error {
	return ErrNotImplemented
}

func RouteDel(route *Route) error {
	return ErrNotImplemented
}

func RouteList(link *Link, family int) ([]Route, error) {
	return nil, ErrNotImplemented
}

func XfrmPolicyAdd(policy *XfrmPolicy) error {
	return ErrNotImplemented
}

func XfrmPolicyDel(policy *XfrmPolicy) error {
	return ErrNotImplemented
}

func XfrmPolicyList(family int) ([]XfrmPolicy, error) {
	return nil, ErrNotImplemented
}

func XfrmStateAdd(policy *XfrmState) error {
	return ErrNotImplemented
}

func XfrmStateDel(policy *XfrmState) error {
	return ErrNotImplemented
}

func XfrmStateList(family int) ([]XfrmState, error) {
	return nil, ErrNotImplemented
}

func NeighAdd(neigh *Neigh) error {
	return ErrNotImplemented
}

func NeighSet(neigh *Neigh) error {
	return ErrNotImplemented
}

func NeighAppend(neigh *Neigh) error {
	return ErrNotImplemented
}

func NeighDel(neigh *Neigh) error {
	return ErrNotImplemented
}

func NeighList(linkIndex, family int) ([]Neigh, error) {
	return nil, ErrNotImplemented
}

func NeighDeserialize(m []byte) (*Ndmsg, *Neigh, error) {
	return nil, nil, ErrNotImplemented
}

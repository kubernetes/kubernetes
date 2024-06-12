// +build !linux

package netlink

import "net"

func LinkSetUp(link Link) error {
	return ErrNotImplemented
}

func LinkSetDown(link Link) error {
	return ErrNotImplemented
}

func LinkSetMTU(link Link, mtu int) error {
	return ErrNotImplemented
}

func LinkSetMaster(link Link, master Link) error {
	return ErrNotImplemented
}

func LinkSetNsPid(link Link, nspid int) error {
	return ErrNotImplemented
}

func LinkSetNsFd(link Link, fd int) error {
	return ErrNotImplemented
}

func LinkSetName(link Link, name string) error {
	return ErrNotImplemented
}

func LinkSetAlias(link Link, name string) error {
	return ErrNotImplemented
}

func LinkSetHardwareAddr(link Link, hwaddr net.HardwareAddr) error {
	return ErrNotImplemented
}

func LinkSetVfHardwareAddr(link Link, vf int, hwaddr net.HardwareAddr) error {
	return ErrNotImplemented
}

func LinkSetVfVlan(link Link, vf, vlan int) error {
	return ErrNotImplemented
}

func LinkSetVfVlanQos(link Link, vf, vlan, qos int) error {
	return ErrNotImplemented
}

func LinkSetVfTxRate(link Link, vf, rate int) error {
	return ErrNotImplemented
}

func LinkSetVfRate(link Link, vf, minRate, maxRate int) error {
	return ErrNotImplemented
}

func LinkSetNoMaster(link Link) error {
	return ErrNotImplemented
}

func LinkSetMasterByIndex(link Link, masterIndex int) error {
	return ErrNotImplemented
}

func LinkSetXdpFd(link Link, fd int) error {
	return ErrNotImplemented
}

func LinkSetXdpFdWithFlags(link Link, fd, flags int) error {
	return ErrNotImplemented
}

func LinkSetARPOff(link Link) error {
	return ErrNotImplemented
}

func LinkSetARPOn(link Link) error {
	return ErrNotImplemented
}

func LinkByName(name string) (Link, error) {
	return nil, ErrNotImplemented
}

func LinkByAlias(alias string) (Link, error) {
	return nil, ErrNotImplemented
}

func LinkByIndex(index int) (Link, error) {
	return nil, ErrNotImplemented
}

func LinkSetHairpin(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetGuard(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetFastLeave(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetLearning(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetRootBlock(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetFlood(link Link, mode bool) error {
	return ErrNotImplemented
}

func LinkSetTxQLen(link Link, qlen int) error {
	return ErrNotImplemented
}

func LinkAdd(link Link) error {
	return ErrNotImplemented
}

func LinkDel(link Link) error {
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

func AddrAdd(link Link, addr *Addr) error {
	return ErrNotImplemented
}

func AddrReplace(link Link, addr *Addr) error {
	return ErrNotImplemented
}

func AddrDel(link Link, addr *Addr) error {
	return ErrNotImplemented
}

func AddrList(link Link, family int) ([]Addr, error) {
	return nil, ErrNotImplemented
}

func RouteAdd(route *Route) error {
	return ErrNotImplemented
}

func RouteAppend(route *Route) error {
	return ErrNotImplemented
}

func RouteDel(route *Route) error {
	return ErrNotImplemented
}

func RouteGet(destination net.IP) ([]Route, error) {
	return nil, ErrNotImplemented
}

func RouteList(link Link, family int) ([]Route, error) {
	return nil, ErrNotImplemented
}

func RouteListFiltered(family int, filter *Route, filterMask uint64) ([]Route, error) {
	return nil, ErrNotImplemented
}

func RouteReplace(route *Route) error {
	return ErrNotImplemented
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

func NeighDeserialize(m []byte) (*Neigh, error) {
	return nil, ErrNotImplemented
}

func SocketGet(local, remote net.Addr) (*Socket, error) {
	return nil, ErrNotImplemented
}

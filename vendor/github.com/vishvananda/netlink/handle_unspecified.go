// +build !linux

package netlink

import (
	"net"
	"time"

	"github.com/vishvananda/netns"
)

type Handle struct{}

func NewHandle(nlFamilies ...int) (*Handle, error) {
	return nil, ErrNotImplemented
}

func NewHandleAt(ns netns.NsHandle, nlFamilies ...int) (*Handle, error) {
	return nil, ErrNotImplemented
}

func NewHandleAtFrom(newNs, curNs netns.NsHandle) (*Handle, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) Delete() {}

func (h *Handle) SupportsNetlinkFamily(nlFamily int) bool {
	return false
}

func (h *Handle) SetSocketTimeout(to time.Duration) error {
	return ErrNotImplemented
}

func (h *Handle) SetPromiscOn(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) SetPromiscOff(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetUp(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetDown(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetMTU(link Link, mtu int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetName(link Link, name string) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetAlias(link Link, name string) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetHardwareAddr(link Link, hwaddr net.HardwareAddr) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetVfHardwareAddr(link Link, vf int, hwaddr net.HardwareAddr) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetVfVlan(link Link, vf, vlan int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetVfVlanQos(link Link, vf, vlan, qos int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetVfTxRate(link Link, vf, rate int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetVfRate(link Link, vf, minRate, maxRate int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetMaster(link Link, master Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetNoMaster(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetMasterByIndex(link Link, masterIndex int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetNsPid(link Link, nspid int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetNsFd(link Link, fd int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkAdd(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkDel(link Link) error {
	return ErrNotImplemented
}

func (h *Handle) LinkByName(name string) (Link, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) LinkByAlias(alias string) (Link, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) LinkByIndex(index int) (Link, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) LinkList() ([]Link, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) LinkSetHairpin(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetGuard(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetFastLeave(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetLearning(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetRootBlock(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetFlood(link Link, mode bool) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetTxQLen(link Link, qlen int) error {
	return ErrNotImplemented
}

func (h *Handle) LinkSetGroup(link Link, group int) error {
	return ErrNotImplemented
}

func (h *Handle) setProtinfoAttr(link Link, mode bool, attr int) error {
	return ErrNotImplemented
}

func (h *Handle) AddrAdd(link Link, addr *Addr) error {
	return ErrNotImplemented
}

func (h *Handle) AddrDel(link Link, addr *Addr) error {
	return ErrNotImplemented
}

func (h *Handle) AddrList(link Link, family int) ([]Addr, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) ClassDel(class Class) error {
	return ErrNotImplemented
}

func (h *Handle) ClassChange(class Class) error {
	return ErrNotImplemented
}

func (h *Handle) ClassReplace(class Class) error {
	return ErrNotImplemented
}

func (h *Handle) ClassAdd(class Class) error {
	return ErrNotImplemented
}

func (h *Handle) ClassList(link Link, parent uint32) ([]Class, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) FilterDel(filter Filter) error {
	return ErrNotImplemented
}

func (h *Handle) FilterAdd(filter Filter) error {
	return ErrNotImplemented
}

func (h *Handle) FilterList(link Link, parent uint32) ([]Filter, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) NeighAdd(neigh *Neigh) error {
	return ErrNotImplemented
}

func (h *Handle) NeighSet(neigh *Neigh) error {
	return ErrNotImplemented
}

func (h *Handle) NeighAppend(neigh *Neigh) error {
	return ErrNotImplemented
}

func (h *Handle) NeighDel(neigh *Neigh) error {
	return ErrNotImplemented
}

func (h *Handle) NeighList(linkIndex, family int) ([]Neigh, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) NeighProxyList(linkIndex, family int) ([]Neigh, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) RouteAdd(route *Route) error {
	return ErrNotImplemented
}

func (h *Handle) RouteDel(route *Route) error {
	return ErrNotImplemented
}

func (h *Handle) RouteGet(destination net.IP) ([]Route, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) RouteList(link Link, family int) ([]Route, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) RouteListFiltered(family int, filter *Route, filterMask uint64) ([]Route, error) {
	return nil, ErrNotImplemented
}

func (h *Handle) RouteReplace(route *Route) error {
	return ErrNotImplemented
}

func (h *Handle) RuleAdd(rule *Rule) error {
	return ErrNotImplemented
}

func (h *Handle) RuleDel(rule *Rule) error {
	return ErrNotImplemented
}

func (h *Handle) RuleList(family int) ([]Rule, error) {
	return nil, ErrNotImplemented
}

package bridge

import (
	"fmt"
	"net"

	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

type link struct {
	parentIP string
	childIP  string
	ports    []types.TransportPort
	bridge   string
}

func (l *link) String() string {
	return fmt.Sprintf("%s <-> %s [%v] on %s", l.parentIP, l.childIP, l.ports, l.bridge)
}

func newLink(parentIP, childIP string, ports []types.TransportPort, bridge string) *link {
	return &link{
		childIP:  childIP,
		parentIP: parentIP,
		ports:    ports,
		bridge:   bridge,
	}

}

func (l *link) Enable() error {
	// -A == iptables append flag
	linkFunction := func() error {
		return linkContainers("-A", l.parentIP, l.childIP, l.ports, l.bridge, false)
	}

	iptables.OnReloaded(func() { linkFunction() })
	return linkFunction()
}

func (l *link) Disable() {
	// -D == iptables delete flag
	err := linkContainers("-D", l.parentIP, l.childIP, l.ports, l.bridge, true)
	if err != nil {
		logrus.Errorf("Error removing IPTables rules for a link %s due to %s", l.String(), err.Error())
	}
	// Return proper error once we move to use a proper iptables package
	// that returns typed errors
}

func linkContainers(action, parentIP, childIP string, ports []types.TransportPort, bridge string,
	ignoreErrors bool) error {
	var nfAction iptables.Action

	switch action {
	case "-A":
		nfAction = iptables.Append
	case "-I":
		nfAction = iptables.Insert
	case "-D":
		nfAction = iptables.Delete
	default:
		return InvalidIPTablesCfgError(action)
	}

	ip1 := net.ParseIP(parentIP)
	if ip1 == nil {
		return InvalidLinkIPAddrError(parentIP)
	}
	ip2 := net.ParseIP(childIP)
	if ip2 == nil {
		return InvalidLinkIPAddrError(childIP)
	}

	chain := iptables.ChainInfo{Name: DockerChain}
	for _, port := range ports {
		err := chain.Link(nfAction, ip1, ip2, int(port.Port), port.Proto.String(), bridge)
		if !ignoreErrors && err != nil {
			return err
		}
	}
	return nil
}

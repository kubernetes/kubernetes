package libnetwork

import "net"

func (n *network) addLBBackend(ip, vip net.IP, fwMark uint32, ingressPorts []*PortConfig) {
}

func (n *network) rmLBBackend(ip, vip net.IP, fwMark uint32, ingressPorts []*PortConfig, rmService bool) {
}

func (sb *sandbox) populateLoadbalancers(ep *endpoint) {
}

func arrangeIngressFilterRule() {
}

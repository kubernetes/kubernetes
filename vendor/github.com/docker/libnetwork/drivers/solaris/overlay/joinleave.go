package overlay

import (
	"fmt"
	"net"

	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/types"
	"github.com/gogo/protobuf/proto"
	"github.com/sirupsen/logrus"
)

// Join method is invoked when a Sandbox is attached to an endpoint.
func (d *driver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	if err := validateID(nid, eid); err != nil {
		return err
	}

	n := d.network(nid)
	if n == nil {
		return fmt.Errorf("could not find network with id %s", nid)
	}

	ep := n.endpoint(eid)
	if ep == nil {
		return fmt.Errorf("could not find endpoint with id %s", eid)
	}

	if n.secure && len(d.keys) == 0 {
		return fmt.Errorf("cannot join secure network: encryption keys not present")
	}

	s := n.getSubnetforIP(ep.addr)
	if s == nil {
		return fmt.Errorf("could not find subnet for endpoint %s", eid)
	}

	if err := n.obtainVxlanID(s); err != nil {
		return fmt.Errorf("couldn't get vxlan id for %q: %v", s.subnetIP.String(), err)
	}

	if err := n.joinSandbox(false); err != nil {
		return fmt.Errorf("network sandbox join failed: %v", err)
	}

	if err := n.joinSubnetSandbox(s, false); err != nil {
		return fmt.Errorf("subnet sandbox join failed for %q: %v", s.subnetIP.String(), err)
	}

	// joinSubnetSandbox gets called when an endpoint comes up on a new subnet in the
	// overlay network. Hence the Endpoint count should be updated outside joinSubnetSandbox
	n.incEndpointCount()

	// Add creating a veth Pair for Solaris

	containerIfName := "solaris-if"
	ep.ifName = containerIfName

	if err := d.writeEndpointToStore(ep); err != nil {
		return fmt.Errorf("failed to update overlay endpoint %s to local data store: %v", ep.id[0:7], err)
	}

	// Add solaris plumbing to add veth (with ep mac addr) to sandbox

	for _, sub := range n.subnets {
		if sub == s {
			continue
		}
		if err := jinfo.AddStaticRoute(sub.subnetIP, types.NEXTHOP, s.gwIP.IP); err != nil {
			logrus.Errorf("Adding subnet %s static route in network %q failed\n", s.subnetIP, n.id)
		}
	}

	if iNames := jinfo.InterfaceName(); iNames != nil {
		err := iNames.SetNames(containerIfName, "eth")
		if err != nil {
			return err
		}
	}

	d.peerDbAdd(nid, eid, ep.addr.IP, ep.addr.Mask, ep.mac,
		net.ParseIP(d.advertiseAddress), true)

	if err := d.checkEncryption(nid, nil, n.vxlanID(s), true, true); err != nil {
		logrus.Warn(err)
	}

	buf, err := proto.Marshal(&PeerRecord{
		EndpointIP:       ep.addr.String(),
		EndpointMAC:      ep.mac.String(),
		TunnelEndpointIP: d.advertiseAddress,
	})
	if err != nil {
		return err
	}

	if err := jinfo.AddTableEntry(ovPeerTable, eid, buf); err != nil {
		logrus.Errorf("overlay: Failed adding table entry to joininfo: %v", err)
	}

	d.pushLocalEndpointEvent("join", nid, eid)

	return nil
}

func (d *driver) EventNotify(etype driverapi.EventType, nid, tableName, key string, value []byte) {
	if tableName != ovPeerTable {
		logrus.Errorf("Unexpected table notification for table %s received", tableName)
		return
	}

	eid := key

	var peer PeerRecord
	if err := proto.Unmarshal(value, &peer); err != nil {
		logrus.Errorf("Failed to unmarshal peer record: %v", err)
		return
	}

	// Ignore local peers. We already know about them and they
	// should not be added to vxlan fdb.
	if peer.TunnelEndpointIP == d.advertiseAddress {
		return
	}

	addr, err := types.ParseCIDR(peer.EndpointIP)
	if err != nil {
		logrus.Errorf("Invalid peer IP %s received in event notify", peer.EndpointIP)
		return
	}

	mac, err := net.ParseMAC(peer.EndpointMAC)
	if err != nil {
		logrus.Errorf("Invalid mac %s received in event notify", peer.EndpointMAC)
		return
	}

	vtep := net.ParseIP(peer.TunnelEndpointIP)
	if vtep == nil {
		logrus.Errorf("Invalid VTEP %s received in event notify", peer.TunnelEndpointIP)
		return
	}

	if etype == driverapi.Delete {
		d.peerDelete(nid, eid, addr.IP, addr.Mask, mac, vtep, true)
		return
	}

	d.peerAdd(nid, eid, addr.IP, addr.Mask, mac, vtep, true)
}

func (d *driver) DecodeTableEntry(tablename string, key string, value []byte) (string, map[string]string) {
	return "", nil
}

// Leave method is invoked when a Sandbox detaches from an endpoint.
func (d *driver) Leave(nid, eid string) error {
	if err := validateID(nid, eid); err != nil {
		return err
	}

	n := d.network(nid)
	if n == nil {
		return fmt.Errorf("could not find network with id %s", nid)
	}

	ep := n.endpoint(eid)

	if ep == nil {
		return types.InternalMaskableErrorf("could not find endpoint with id %s", eid)
	}

	if d.notifyCh != nil {
		d.notifyCh <- ovNotify{
			action: "leave",
			nw:     n,
			ep:     ep,
		}
	}

	n.leaveSandbox()

	if err := d.checkEncryption(nid, nil, 0, true, false); err != nil {
		logrus.Warn(err)
	}

	return nil
}

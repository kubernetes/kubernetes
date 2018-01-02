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

	buf, err := proto.Marshal(&PeerRecord{
		EndpointIP:       ep.addr.String(),
		EndpointMAC:      ep.mac.String(),
		TunnelEndpointIP: n.providerAddress,
	})

	if err != nil {
		return err
	}

	if err := jinfo.AddTableEntry(ovPeerTable, eid, buf); err != nil {
		logrus.Errorf("overlay: Failed adding table entry to joininfo: %v", err)
	}

	if ep.disablegateway {
		jinfo.DisableGatewayService()
	}

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

	n := d.network(nid)
	if n == nil {
		return
	}

	// Ignore local peers. We already know about them and they
	// should not be added to vxlan fdb.
	if peer.TunnelEndpointIP == n.providerAddress {
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

	return nil
}

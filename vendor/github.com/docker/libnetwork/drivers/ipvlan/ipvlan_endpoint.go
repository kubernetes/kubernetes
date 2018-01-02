package ipvlan

import (
	"fmt"

	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/ns"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// CreateEndpoint assigns the mac, ip and endpoint id for the new container
func (d *driver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo,
	epOptions map[string]interface{}) error {
	defer osl.InitOSContext()()

	if err := validateID(nid, eid); err != nil {
		return err
	}
	n, err := d.getNetwork(nid)
	if err != nil {
		return fmt.Errorf("network id %q not found", nid)
	}
	if ifInfo.MacAddress() != nil {
		return fmt.Errorf("%s interfaces do not support custom mac address assigment", ipvlanType)
	}
	ep := &endpoint{
		id:     eid,
		nid:    nid,
		addr:   ifInfo.Address(),
		addrv6: ifInfo.AddressIPv6(),
	}
	if ep.addr == nil {
		return fmt.Errorf("create endpoint was not passed an IP address")
	}
	// disallow port mapping -p
	if opt, ok := epOptions[netlabel.PortMap]; ok {
		if _, ok := opt.([]types.PortBinding); ok {
			if len(opt.([]types.PortBinding)) > 0 {
				logrus.Warnf("%s driver does not support port mappings", ipvlanType)
			}
		}
	}
	// disallow port exposure --expose
	if opt, ok := epOptions[netlabel.ExposedPorts]; ok {
		if _, ok := opt.([]types.TransportPort); ok {
			if len(opt.([]types.TransportPort)) > 0 {
				logrus.Warnf("%s driver does not support port exposures", ipvlanType)
			}
		}
	}

	if err := d.storeUpdate(ep); err != nil {
		return fmt.Errorf("failed to save ipvlan endpoint %s to store: %v", ep.id[0:7], err)
	}

	n.addEndpoint(ep)

	return nil
}

// DeleteEndpoint remove the endpoint and associated netlink interface
func (d *driver) DeleteEndpoint(nid, eid string) error {
	defer osl.InitOSContext()()
	if err := validateID(nid, eid); err != nil {
		return err
	}
	n := d.network(nid)
	if n == nil {
		return fmt.Errorf("network id %q not found", nid)
	}
	ep := n.endpoint(eid)
	if ep == nil {
		return fmt.Errorf("endpoint id %q not found", eid)
	}
	if link, err := ns.NlHandle().LinkByName(ep.srcName); err == nil {
		ns.NlHandle().LinkDel(link)
	}

	if err := d.storeDelete(ep); err != nil {
		logrus.Warnf("Failed to remove ipvlan endpoint %s from store: %v", ep.id[0:7], err)
	}
	n.deleteEndpoint(ep.id)
	return nil
}

package overlay

import (
	"encoding/json"
	"fmt"
	"net"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/drivers/windows"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

type endpointTable map[string]*endpoint

const overlayEndpointPrefix = "overlay/endpoint"

type endpoint struct {
	id             string
	nid            string
	profileID      string
	remote         bool
	mac            net.HardwareAddr
	addr           *net.IPNet
	disablegateway bool
	portMapping    []types.PortBinding // Operation port bindings
}

func validateID(nid, eid string) error {
	if nid == "" {
		return fmt.Errorf("invalid network id")
	}

	if eid == "" {
		return fmt.Errorf("invalid endpoint id")
	}

	return nil
}

func (n *network) endpoint(eid string) *endpoint {
	n.Lock()
	defer n.Unlock()

	return n.endpoints[eid]
}

func (n *network) addEndpoint(ep *endpoint) {
	n.Lock()
	n.endpoints[ep.id] = ep
	n.Unlock()
}

func (n *network) deleteEndpoint(eid string) {
	n.Lock()
	delete(n.endpoints, eid)
	n.Unlock()
}

func (n *network) removeEndpointWithAddress(addr *net.IPNet) {
	var networkEndpoint *endpoint
	n.Lock()
	for _, ep := range n.endpoints {
		if ep.addr.IP.Equal(addr.IP) {
			networkEndpoint = ep
			break
		}
	}

	if networkEndpoint != nil {
		delete(n.endpoints, networkEndpoint.id)
	}
	n.Unlock()

	if networkEndpoint != nil {
		logrus.Debugf("Removing stale endpoint from HNS")
		_, err := hcsshim.HNSEndpointRequest("DELETE", networkEndpoint.profileID, "")

		if err != nil {
			logrus.Debugf("Failed to delete stale overlay endpoint (%s) from hns", networkEndpoint.id[0:7])
		}
	}
}

func (d *driver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo,
	epOptions map[string]interface{}) error {
	var err error
	if err = validateID(nid, eid); err != nil {
		return err
	}

	n := d.network(nid)
	if n == nil {
		return fmt.Errorf("network id %q not found", nid)
	}

	ep := n.endpoint(eid)
	if ep != nil {
		logrus.Debugf("Deleting stale endpoint %s", eid)
		n.deleteEndpoint(eid)

		_, err := hcsshim.HNSEndpointRequest("DELETE", ep.profileID, "")
		if err != nil {
			return err
		}
	}

	ep = &endpoint{
		id:   eid,
		nid:  n.id,
		addr: ifInfo.Address(),
		mac:  ifInfo.MacAddress(),
	}

	if ep.addr == nil {
		return fmt.Errorf("create endpoint was not passed interface IP address")
	}

	s := n.getSubnetforIP(ep.addr)
	if s == nil {
		return fmt.Errorf("no matching subnet for IP %q in network %q", ep.addr, nid)
	}

	// Todo: Add port bindings and qos policies here

	hnsEndpoint := &hcsshim.HNSEndpoint{
		Name:              eid,
		VirtualNetwork:    n.hnsID,
		IPAddress:         ep.addr.IP,
		EnableInternalDNS: true,
		GatewayAddress:    s.gwIP.String(),
	}

	if ep.mac != nil {
		hnsEndpoint.MacAddress = ep.mac.String()
	}

	paPolicy, err := json.Marshal(hcsshim.PaPolicy{
		Type: "PA",
		PA:   n.providerAddress,
	})

	if err != nil {
		return err
	}

	hnsEndpoint.Policies = append(hnsEndpoint.Policies, paPolicy)

	if system.GetOSVersion().Build > 16236 {
		natPolicy, err := json.Marshal(hcsshim.PaPolicy{
			Type: "OutBoundNAT",
		})

		if err != nil {
			return err
		}

		hnsEndpoint.Policies = append(hnsEndpoint.Policies, natPolicy)

		epConnectivity, err := windows.ParseEndpointConnectivity(epOptions)
		if err != nil {
			return err
		}

		pbPolicy, err := windows.ConvertPortBindings(epConnectivity.PortBindings)
		if err != nil {
			return err
		}
		hnsEndpoint.Policies = append(hnsEndpoint.Policies, pbPolicy...)

		ep.disablegateway = true
	}

	configurationb, err := json.Marshal(hnsEndpoint)
	if err != nil {
		return err
	}

	hnsresponse, err := hcsshim.HNSEndpointRequest("POST", "", string(configurationb))
	if err != nil {
		return err
	}

	ep.profileID = hnsresponse.Id

	if ep.mac == nil {
		ep.mac, err = net.ParseMAC(hnsresponse.MacAddress)
		if err != nil {
			return err
		}

		if err := ifInfo.SetMacAddress(ep.mac); err != nil {
			return err
		}
	}

	ep.portMapping, err = windows.ParsePortBindingPolicies(hnsresponse.Policies)
	if err != nil {
		hcsshim.HNSEndpointRequest("DELETE", hnsresponse.Id, "")
		return err
	}

	n.addEndpoint(ep)

	return nil
}

func (d *driver) DeleteEndpoint(nid, eid string) error {
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

	n.deleteEndpoint(eid)

	_, err := hcsshim.HNSEndpointRequest("DELETE", ep.profileID, "")
	if err != nil {
		return err
	}

	return nil
}

func (d *driver) EndpointOperInfo(nid, eid string) (map[string]interface{}, error) {
	if err := validateID(nid, eid); err != nil {
		return nil, err
	}

	n := d.network(nid)
	if n == nil {
		return nil, fmt.Errorf("network id %q not found", nid)
	}

	ep := n.endpoint(eid)
	if ep == nil {
		return nil, fmt.Errorf("endpoint id %q not found", eid)
	}

	data := make(map[string]interface{}, 1)
	data["hnsid"] = ep.profileID
	data["AllowUnqualifiedDNSQuery"] = true

	if ep.portMapping != nil {
		// Return a copy of the operational data
		pmc := make([]types.PortBinding, 0, len(ep.portMapping))
		for _, pm := range ep.portMapping {
			pmc = append(pmc, pm.GetCopy())
		}
		data[netlabel.PortMap] = pmc
	}

	return data, nil
}

package overlay

import (
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

var (
	hostMode  bool
	networkMu sync.Mutex
)

type networkTable map[string]*network

type subnet struct {
	vni      uint32
	subnetIP *net.IPNet
	gwIP     *net.IP
}

type subnetJSON struct {
	SubnetIP string
	GwIP     string
	Vni      uint32
}

type network struct {
	id              string
	name            string
	hnsID           string
	providerAddress string
	interfaceName   string
	endpoints       endpointTable
	driver          *driver
	initEpoch       int
	initErr         error
	subnets         []*subnet
	secure          bool
	sync.Mutex
}

func (d *driver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	return nil, types.NotImplementedErrorf("not implemented")
}

func (d *driver) NetworkFree(id string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) CreateNetwork(id string, option map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	var (
		networkName   string
		interfaceName string
		staleNetworks []string
	)

	if id == "" {
		return fmt.Errorf("invalid network id")
	}

	if nInfo == nil {
		return fmt.Errorf("invalid network info structure")
	}

	if len(ipV4Data) == 0 || ipV4Data[0].Pool.String() == "0.0.0.0/0" {
		return types.BadRequestErrorf("ipv4 pool is empty")
	}

	staleNetworks = make([]string, 0)
	vnis := make([]uint32, 0, len(ipV4Data))

	existingNetwork := d.network(id)
	if existingNetwork != nil {
		logrus.Debugf("Network preexists. Deleting %s", id)
		err := d.DeleteNetwork(id)
		if err != nil {
			logrus.Errorf("Error deleting stale network %s", err.Error())
		}
	}

	n := &network{
		id:        id,
		driver:    d,
		endpoints: endpointTable{},
		subnets:   []*subnet{},
	}

	genData, ok := option[netlabel.GenericData].(map[string]string)

	if !ok {
		return fmt.Errorf("Unknown generic data option")
	}

	for label, value := range genData {
		switch label {
		case "com.docker.network.windowsshim.networkname":
			networkName = value
		case "com.docker.network.windowsshim.interface":
			interfaceName = value
		case "com.docker.network.windowsshim.hnsid":
			n.hnsID = value
		case netlabel.OverlayVxlanIDList:
			vniStrings := strings.Split(value, ",")
			for _, vniStr := range vniStrings {
				vni, err := strconv.Atoi(vniStr)
				if err != nil {
					return fmt.Errorf("invalid vxlan id value %q passed", vniStr)
				}

				vnis = append(vnis, uint32(vni))
			}
		}
	}

	// If we are getting vnis from libnetwork, either we get for
	// all subnets or none.
	if len(vnis) < len(ipV4Data) {
		return fmt.Errorf("insufficient vnis(%d) passed to overlay. Windows driver requires VNIs to be prepopulated", len(vnis))
	}

	for i, ipd := range ipV4Data {
		s := &subnet{
			subnetIP: ipd.Pool,
			gwIP:     &ipd.Gateway.IP,
		}

		if len(vnis) != 0 {
			s.vni = vnis[i]
		}

		d.Lock()
		for _, network := range d.networks {
			found := false
			for _, sub := range network.subnets {
				if sub.vni == s.vni {
					staleNetworks = append(staleNetworks, network.id)
					found = true
					break
				}
			}
			if found {
				break
			}
		}
		d.Unlock()

		n.subnets = append(n.subnets, s)
	}

	for _, staleNetwork := range staleNetworks {
		d.DeleteNetwork(staleNetwork)
	}

	n.name = networkName
	if n.name == "" {
		n.name = id
	}

	n.interfaceName = interfaceName

	if nInfo != nil {
		if err := nInfo.TableEventRegister(ovPeerTable, driverapi.EndpointObject); err != nil {
			return err
		}
	}

	d.addNetwork(n)

	err := d.createHnsNetwork(n)

	if err != nil {
		d.deleteNetwork(id)
	} else {
		genData["com.docker.network.windowsshim.hnsid"] = n.hnsID
	}

	return err
}

func (d *driver) DeleteNetwork(nid string) error {
	if nid == "" {
		return fmt.Errorf("invalid network id")
	}

	n := d.network(nid)
	if n == nil {
		return types.ForbiddenErrorf("could not find network with id %s", nid)
	}

	_, err := hcsshim.HNSNetworkRequest("DELETE", n.hnsID, "")
	if err != nil {
		return types.ForbiddenErrorf(err.Error())
	}

	d.deleteNetwork(nid)

	return nil
}

func (d *driver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	return nil
}

func (d *driver) RevokeExternalConnectivity(nid, eid string) error {
	return nil
}

func (d *driver) addNetwork(n *network) {
	d.Lock()
	d.networks[n.id] = n
	d.Unlock()
}

func (d *driver) deleteNetwork(nid string) {
	d.Lock()
	delete(d.networks, nid)
	d.Unlock()
}

func (d *driver) network(nid string) *network {
	d.Lock()
	defer d.Unlock()
	return d.networks[nid]
}

// func (n *network) restoreNetworkEndpoints() error {
// 	logrus.Infof("Restoring endpoints for overlay network: %s", n.id)

// 	hnsresponse, err := hcsshim.HNSListEndpointRequest("GET", "", "")
// 	if err != nil {
// 		return err
// 	}

// 	for _, endpoint := range hnsresponse {
// 		if endpoint.VirtualNetwork != n.hnsID {
// 			continue
// 		}

// 		ep := n.convertToOverlayEndpoint(&endpoint)

// 		if ep != nil {
// 			logrus.Debugf("Restored endpoint:%s Remote:%t", ep.id, ep.remote)
// 			n.addEndpoint(ep)
// 		}
// 	}

// 	return nil
// }

func (n *network) convertToOverlayEndpoint(v *hcsshim.HNSEndpoint) *endpoint {
	ep := &endpoint{
		id:        v.Name,
		profileID: v.Id,
		nid:       n.id,
		remote:    v.IsRemoteEndpoint,
	}

	mac, err := net.ParseMAC(v.MacAddress)

	if err != nil {
		return nil
	}

	ep.mac = mac
	ep.addr = &net.IPNet{
		IP:   v.IPAddress,
		Mask: net.CIDRMask(32, 32),
	}

	return ep
}

func (d *driver) createHnsNetwork(n *network) error {

	subnets := []hcsshim.Subnet{}

	for _, s := range n.subnets {
		subnet := hcsshim.Subnet{
			AddressPrefix: s.subnetIP.String(),
		}

		if s.gwIP != nil {
			subnet.GatewayAddress = s.gwIP.String()
		}

		vsidPolicy, err := json.Marshal(hcsshim.VsidPolicy{
			Type: "VSID",
			VSID: uint(s.vni),
		})

		if err != nil {
			return err
		}

		subnet.Policies = append(subnet.Policies, vsidPolicy)
		subnets = append(subnets, subnet)
	}

	network := &hcsshim.HNSNetwork{
		Name:               n.name,
		Type:               d.Type(),
		Subnets:            subnets,
		NetworkAdapterName: n.interfaceName,
		AutomaticDNS:       true,
	}

	configurationb, err := json.Marshal(network)
	if err != nil {
		return err
	}

	configuration := string(configurationb)
	logrus.Infof("HNSNetwork Request =%v", configuration)

	hnsresponse, err := hcsshim.HNSNetworkRequest("POST", "", configuration)
	if err != nil {
		return err
	}

	n.hnsID = hnsresponse.Id
	n.providerAddress = hnsresponse.ManagementIP

	return nil
}

// contains return true if the passed ip belongs to one the network's
// subnets
func (n *network) contains(ip net.IP) bool {
	for _, s := range n.subnets {
		if s.subnetIP.Contains(ip) {
			return true
		}
	}

	return false
}

// getSubnetforIP returns the subnet to which the given IP belongs
func (n *network) getSubnetforIP(ip *net.IPNet) *subnet {
	for _, s := range n.subnets {
		// first check if the mask lengths are the same
		i, _ := s.subnetIP.Mask.Size()
		j, _ := ip.Mask.Size()
		if i != j {
			continue
		}
		if s.subnetIP.Contains(ip.IP) {
			return s
		}
	}
	return nil
}

// getMatchingSubnet return the network's subnet that matches the input
func (n *network) getMatchingSubnet(ip *net.IPNet) *subnet {
	if ip == nil {
		return nil
	}
	for _, s := range n.subnets {
		// first check if the mask lengths are the same
		i, _ := s.subnetIP.Mask.Size()
		j, _ := ip.Mask.Size()
		if i != j {
			continue
		}
		if s.subnetIP.IP.Equal(ip.IP) {
			return s
		}
	}
	return nil
}

// +build windows

// Shim for the Host Network Service (HNS) to manage networking for
// Windows Server containers and Hyper-V containers. This module
// is a basic libnetwork driver that passes all the calls to HNS
// It implements the 4 networking modes supported by HNS L2Bridge,
// L2Tunnel, NAT and Transparent(DHCP)
//
// The network are stored in memory and docker daemon ensures discovering
// and loading these networks on startup

package windows

import (
	"encoding/json"
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

// networkConfiguration for network specific configuration
type networkConfiguration struct {
	ID                 string
	Type               string
	Name               string
	HnsID              string
	RDID               string
	VLAN               uint
	VSID               uint
	DNSServers         string
	MacPools           []hcsshim.MacPool
	DNSSuffix          string
	SourceMac          string
	NetworkAdapterName string
	dbIndex            uint64
	dbExists           bool
}

// endpointConfiguration represents the user specified configuration for the sandbox endpoint
type endpointOption struct {
	MacAddress  net.HardwareAddr
	QosPolicies []types.QosPolicy
	DNSServers  []string
	DisableDNS  bool
	DisableICC  bool
}

// EndpointConnectivity stores the port bindings and exposed ports that the user has specified in epOptions.
type EndpointConnectivity struct {
	PortBindings []types.PortBinding
	ExposedPorts []types.TransportPort
}

type hnsEndpoint struct {
	id             string
	nid            string
	profileID      string
	Type           string
	macAddress     net.HardwareAddr
	epOption       *endpointOption       // User specified parameters
	epConnectivity *EndpointConnectivity // User specified parameters
	portMapping    []types.PortBinding   // Operation port bindings
	addr           *net.IPNet
	gateway        net.IP
	dbIndex        uint64
	dbExists       bool
}

type hnsNetwork struct {
	id        string
	created   bool
	config    *networkConfiguration
	endpoints map[string]*hnsEndpoint // key: endpoint id
	driver    *driver                 // The network's driver
	sync.Mutex
}

type driver struct {
	name     string
	networks map[string]*hnsNetwork
	store    datastore.DataStore
	sync.Mutex
}

const (
	errNotFound = "HNS failed with error : The object identifier does not represent a valid object. "
)

// IsBuiltinLocalDriver validates if network-type is a builtin local-scoped driver
func IsBuiltinLocalDriver(networkType string) bool {
	if "l2bridge" == networkType || "l2tunnel" == networkType || "nat" == networkType || "ics" == networkType || "transparent" == networkType {
		return true
	}

	return false
}

// New constructs a new bridge driver
func newDriver(networkType string) *driver {
	return &driver{name: networkType, networks: map[string]*hnsNetwork{}}
}

// GetInit returns an initializer for the given network type
func GetInit(networkType string) func(dc driverapi.DriverCallback, config map[string]interface{}) error {
	return func(dc driverapi.DriverCallback, config map[string]interface{}) error {
		if !IsBuiltinLocalDriver(networkType) {
			return types.BadRequestErrorf("Network type not supported: %s", networkType)
		}

		d := newDriver(networkType)

		err := d.initStore(config)
		if err != nil {
			return err
		}

		return dc.RegisterDriver(networkType, d, driverapi.Capability{
			DataScope:         datastore.LocalScope,
			ConnectivityScope: datastore.LocalScope,
		})
	}
}

func (d *driver) getNetwork(id string) (*hnsNetwork, error) {
	d.Lock()
	defer d.Unlock()

	if nw, ok := d.networks[id]; ok {
		return nw, nil
	}

	return nil, types.NotFoundErrorf("network not found: %s", id)
}

func (n *hnsNetwork) getEndpoint(eid string) (*hnsEndpoint, error) {
	n.Lock()
	defer n.Unlock()

	if ep, ok := n.endpoints[eid]; ok {
		return ep, nil
	}

	return nil, types.NotFoundErrorf("Endpoint not found: %s", eid)
}

func (d *driver) parseNetworkOptions(id string, genericOptions map[string]string) (*networkConfiguration, error) {
	config := &networkConfiguration{Type: d.name}

	for label, value := range genericOptions {
		switch label {
		case NetworkName:
			config.Name = value
		case HNSID:
			config.HnsID = value
		case RoutingDomain:
			config.RDID = value
		case Interface:
			config.NetworkAdapterName = value
		case DNSSuffix:
			config.DNSSuffix = value
		case DNSServers:
			config.DNSServers = value
		case MacPool:
			config.MacPools = make([]hcsshim.MacPool, 0)
			s := strings.Split(value, ",")
			if len(s)%2 != 0 {
				return nil, types.BadRequestErrorf("Invalid mac pool. You must specify both a start range and an end range")
			}
			for i := 0; i < len(s)-1; i += 2 {
				config.MacPools = append(config.MacPools, hcsshim.MacPool{
					StartMacAddress: s[i],
					EndMacAddress:   s[i+1],
				})
			}
		case VLAN:
			vlan, err := strconv.ParseUint(value, 10, 32)
			if err != nil {
				return nil, err
			}
			config.VLAN = uint(vlan)
		case VSID:
			vsid, err := strconv.ParseUint(value, 10, 32)
			if err != nil {
				return nil, err
			}
			config.VSID = uint(vsid)
		}
	}

	config.ID = id
	config.Type = d.name
	return config, nil
}

func (c *networkConfiguration) processIPAM(id string, ipamV4Data, ipamV6Data []driverapi.IPAMData) error {
	if len(ipamV6Data) > 0 {
		return types.ForbiddenErrorf("windowsshim driver doesn't support v6 subnets")
	}

	if len(ipamV4Data) == 0 {
		return types.BadRequestErrorf("network %s requires ipv4 configuration", id)
	}

	return nil
}

func (d *driver) EventNotify(etype driverapi.EventType, nid, tableName, key string, value []byte) {
}

func (d *driver) DecodeTableEntry(tablename string, key string, value []byte) (string, map[string]string) {
	return "", nil
}

func (d *driver) createNetwork(config *networkConfiguration) error {
	network := &hnsNetwork{
		id:        config.ID,
		endpoints: make(map[string]*hnsEndpoint),
		config:    config,
		driver:    d,
	}

	d.Lock()
	d.networks[config.ID] = network
	d.Unlock()

	return nil
}

// Create a new network
func (d *driver) CreateNetwork(id string, option map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	if _, err := d.getNetwork(id); err == nil {
		return types.ForbiddenErrorf("network %s exists", id)
	}

	genData, ok := option[netlabel.GenericData].(map[string]string)
	if !ok {
		return fmt.Errorf("Unknown generic data option")
	}

	// Parse and validate the config. It should not conflict with existing networks' config
	config, err := d.parseNetworkOptions(id, genData)
	if err != nil {
		return err
	}

	err = config.processIPAM(id, ipV4Data, ipV6Data)
	if err != nil {
		return err
	}

	err = d.createNetwork(config)

	if err != nil {
		return err
	}

	// A non blank hnsid indicates that the network was discovered
	// from HNS. No need to call HNS if this network was discovered
	// from HNS
	if config.HnsID == "" {
		subnets := []hcsshim.Subnet{}

		for _, ipData := range ipV4Data {
			subnet := hcsshim.Subnet{
				AddressPrefix: ipData.Pool.String(),
			}

			if ipData.Gateway != nil {
				subnet.GatewayAddress = ipData.Gateway.IP.String()
			}

			subnets = append(subnets, subnet)
		}

		network := &hcsshim.HNSNetwork{
			Name:               config.Name,
			Type:               d.name,
			Subnets:            subnets,
			DNSServerList:      config.DNSServers,
			DNSSuffix:          config.DNSSuffix,
			MacPools:           config.MacPools,
			SourceMac:          config.SourceMac,
			NetworkAdapterName: config.NetworkAdapterName,
		}

		if config.VLAN != 0 {
			vlanPolicy, err := json.Marshal(hcsshim.VlanPolicy{
				Type: "VLAN",
				VLAN: config.VLAN,
			})

			if err != nil {
				return err
			}
			network.Policies = append(network.Policies, vlanPolicy)
		}

		if config.VSID != 0 {
			vsidPolicy, err := json.Marshal(hcsshim.VsidPolicy{
				Type: "VSID",
				VSID: config.VSID,
			})

			if err != nil {
				return err
			}
			network.Policies = append(network.Policies, vsidPolicy)
		}

		if network.Name == "" {
			network.Name = id
		}

		configurationb, err := json.Marshal(network)
		if err != nil {
			return err
		}

		configuration := string(configurationb)
		logrus.Debugf("HNSNetwork Request =%v Address Space=%v", configuration, subnets)

		hnsresponse, err := hcsshim.HNSNetworkRequest("POST", "", configuration)
		if err != nil {
			return err
		}

		config.HnsID = hnsresponse.Id
		genData[HNSID] = config.HnsID
	}

	n, err := d.getNetwork(id)
	if err != nil {
		return err
	}
	n.created = true
	return d.storeUpdate(config)
}

func (d *driver) DeleteNetwork(nid string) error {
	n, err := d.getNetwork(nid)
	if err != nil {
		return types.InternalMaskableErrorf("%s", err)
	}

	n.Lock()
	config := n.config
	n.Unlock()

	if n.created {
		_, err = hcsshim.HNSNetworkRequest("DELETE", config.HnsID, "")
		if err != nil && err.Error() != errNotFound {
			return types.ForbiddenErrorf(err.Error())
		}
	}

	d.Lock()
	delete(d.networks, nid)
	d.Unlock()

	// delele endpoints belong to this network
	for _, ep := range n.endpoints {
		if err := d.storeDelete(ep); err != nil {
			logrus.Warnf("Failed to remove bridge endpoint %s from store: %v", ep.id[0:7], err)
		}
	}

	return d.storeDelete(config)
}

func convertQosPolicies(qosPolicies []types.QosPolicy) ([]json.RawMessage, error) {
	var qps []json.RawMessage

	// Enumerate through the qos policies specified by the user and convert
	// them into the internal structure matching the JSON blob that can be
	// understood by the HCS.
	for _, elem := range qosPolicies {
		encodedPolicy, err := json.Marshal(hcsshim.QosPolicy{
			Type: "QOS",
			MaximumOutgoingBandwidthInBytes: elem.MaxEgressBandwidth,
		})

		if err != nil {
			return nil, err
		}
		qps = append(qps, encodedPolicy)
	}
	return qps, nil
}

// ConvertPortBindings converts PortBindings to JSON for HNS request
func ConvertPortBindings(portBindings []types.PortBinding) ([]json.RawMessage, error) {
	var pbs []json.RawMessage

	// Enumerate through the port bindings specified by the user and convert
	// them into the internal structure matching the JSON blob that can be
	// understood by the HCS.
	for _, elem := range portBindings {
		proto := strings.ToUpper(elem.Proto.String())
		if proto != "TCP" && proto != "UDP" {
			return nil, fmt.Errorf("invalid protocol %s", elem.Proto.String())
		}

		if elem.HostPort != elem.HostPortEnd {
			return nil, fmt.Errorf("Windows does not support more than one host port in NAT settings")
		}

		if len(elem.HostIP) != 0 {
			return nil, fmt.Errorf("Windows does not support host IP addresses in NAT settings")
		}

		encodedPolicy, err := json.Marshal(hcsshim.NatPolicy{
			Type:         "NAT",
			ExternalPort: elem.HostPort,
			InternalPort: elem.Port,
			Protocol:     elem.Proto.String(),
		})

		if err != nil {
			return nil, err
		}
		pbs = append(pbs, encodedPolicy)
	}
	return pbs, nil
}

// ParsePortBindingPolicies parses HNS endpoint response message to PortBindings
func ParsePortBindingPolicies(policies []json.RawMessage) ([]types.PortBinding, error) {
	var bindings []types.PortBinding
	hcsPolicy := &hcsshim.NatPolicy{}

	for _, elem := range policies {

		if err := json.Unmarshal([]byte(elem), &hcsPolicy); err != nil || hcsPolicy.Type != "NAT" {
			continue
		}

		binding := types.PortBinding{
			HostPort:    hcsPolicy.ExternalPort,
			HostPortEnd: hcsPolicy.ExternalPort,
			Port:        hcsPolicy.InternalPort,
			Proto:       types.ParseProtocol(hcsPolicy.Protocol),
			HostIP:      net.IPv4(0, 0, 0, 0),
		}

		bindings = append(bindings, binding)
	}

	return bindings, nil
}

func parseEndpointOptions(epOptions map[string]interface{}) (*endpointOption, error) {
	if epOptions == nil {
		return nil, nil
	}

	ec := &endpointOption{}

	if opt, ok := epOptions[netlabel.MacAddress]; ok {
		if mac, ok := opt.(net.HardwareAddr); ok {
			ec.MacAddress = mac
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	if opt, ok := epOptions[QosPolicies]; ok {
		if policies, ok := opt.([]types.QosPolicy); ok {
			ec.QosPolicies = policies
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	if opt, ok := epOptions[netlabel.DNSServers]; ok {
		if dns, ok := opt.([]string); ok {
			ec.DNSServers = dns
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	if opt, ok := epOptions[DisableICC]; ok {
		if disableICC, ok := opt.(bool); ok {
			ec.DisableICC = disableICC
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	if opt, ok := epOptions[DisableDNS]; ok {
		if disableDNS, ok := opt.(bool); ok {
			ec.DisableDNS = disableDNS
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	return ec, nil
}

// ParseEndpointConnectivity parses options passed to CreateEndpoint, specifically port bindings, and store in a endpointConnectivity object.
func ParseEndpointConnectivity(epOptions map[string]interface{}) (*EndpointConnectivity, error) {
	if epOptions == nil {
		return nil, nil
	}

	ec := &EndpointConnectivity{}

	if opt, ok := epOptions[netlabel.PortMap]; ok {
		if bs, ok := opt.([]types.PortBinding); ok {
			ec.PortBindings = bs
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}

	if opt, ok := epOptions[netlabel.ExposedPorts]; ok {
		if ports, ok := opt.([]types.TransportPort); ok {
			ec.ExposedPorts = ports
		} else {
			return nil, fmt.Errorf("Invalid endpoint configuration")
		}
	}
	return ec, nil
}

func (d *driver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo, epOptions map[string]interface{}) error {
	n, err := d.getNetwork(nid)
	if err != nil {
		return err
	}

	// Check if endpoint id is good and retrieve corresponding endpoint
	ep, err := n.getEndpoint(eid)
	if err == nil && ep != nil {
		return driverapi.ErrEndpointExists(eid)
	}

	endpointStruct := &hcsshim.HNSEndpoint{
		VirtualNetwork: n.config.HnsID,
	}

	epOption, err := parseEndpointOptions(epOptions)
	if err != nil {
		return err
	}
	epConnectivity, err := ParseEndpointConnectivity(epOptions)
	if err != nil {
		return err
	}

	macAddress := ifInfo.MacAddress()
	// Use the macaddress if it was provided
	if macAddress != nil {
		endpointStruct.MacAddress = strings.Replace(macAddress.String(), ":", "-", -1)
	}

	endpointStruct.Policies, err = ConvertPortBindings(epConnectivity.PortBindings)
	if err != nil {
		return err
	}

	qosPolicies, err := convertQosPolicies(epOption.QosPolicies)
	if err != nil {
		return err
	}
	endpointStruct.Policies = append(endpointStruct.Policies, qosPolicies...)

	if ifInfo.Address() != nil {
		endpointStruct.IPAddress = ifInfo.Address().IP
	}

	endpointStruct.DNSServerList = strings.Join(epOption.DNSServers, ",")

	if n.driver.name == "nat" && !epOption.DisableDNS {
		endpointStruct.EnableInternalDNS = true
	}

	endpointStruct.DisableICC = epOption.DisableICC

	configurationb, err := json.Marshal(endpointStruct)
	if err != nil {
		return err
	}

	hnsresponse, err := hcsshim.HNSEndpointRequest("POST", "", string(configurationb))
	if err != nil {
		return err
	}

	mac, err := net.ParseMAC(hnsresponse.MacAddress)
	if err != nil {
		return err
	}

	// TODO For now the ip mask is not in the info generated by HNS
	endpoint := &hnsEndpoint{
		id:         eid,
		nid:        n.id,
		Type:       d.name,
		addr:       &net.IPNet{IP: hnsresponse.IPAddress, Mask: hnsresponse.IPAddress.DefaultMask()},
		macAddress: mac,
	}

	if hnsresponse.GatewayAddress != "" {
		endpoint.gateway = net.ParseIP(hnsresponse.GatewayAddress)
	}

	endpoint.profileID = hnsresponse.Id
	endpoint.epConnectivity = epConnectivity
	endpoint.epOption = epOption
	endpoint.portMapping, err = ParsePortBindingPolicies(hnsresponse.Policies)

	if err != nil {
		hcsshim.HNSEndpointRequest("DELETE", hnsresponse.Id, "")
		return err
	}

	n.Lock()
	n.endpoints[eid] = endpoint
	n.Unlock()

	if ifInfo.Address() == nil {
		ifInfo.SetIPAddress(endpoint.addr)
	}

	if macAddress == nil {
		ifInfo.SetMacAddress(endpoint.macAddress)
	}

	if err = d.storeUpdate(endpoint); err != nil {
		return fmt.Errorf("failed to save endpoint %s to store: %v", endpoint.id[0:7], err)
	}

	return nil
}

func (d *driver) DeleteEndpoint(nid, eid string) error {
	n, err := d.getNetwork(nid)
	if err != nil {
		return types.InternalMaskableErrorf("%s", err)
	}

	ep, err := n.getEndpoint(eid)
	if err != nil {
		return err
	}

	n.Lock()
	delete(n.endpoints, eid)
	n.Unlock()

	_, err = hcsshim.HNSEndpointRequest("DELETE", ep.profileID, "")
	if err != nil && err.Error() != errNotFound {
		return err
	}

	if err := d.storeDelete(ep); err != nil {
		logrus.Warnf("Failed to remove bridge endpoint %s from store: %v", ep.id[0:7], err)
	}
	return nil
}

func (d *driver) EndpointOperInfo(nid, eid string) (map[string]interface{}, error) {
	network, err := d.getNetwork(nid)
	if err != nil {
		return nil, err
	}

	ep, err := network.getEndpoint(eid)
	if err != nil {
		return nil, err
	}

	data := make(map[string]interface{}, 1)
	if network.driver.name == "nat" {
		data["AllowUnqualifiedDNSQuery"] = true
	}

	data["hnsid"] = ep.profileID
	if ep.epConnectivity.ExposedPorts != nil {
		// Return a copy of the config data
		epc := make([]types.TransportPort, 0, len(ep.epConnectivity.ExposedPorts))
		for _, tp := range ep.epConnectivity.ExposedPorts {
			epc = append(epc, tp.GetCopy())
		}
		data[netlabel.ExposedPorts] = epc
	}

	if ep.portMapping != nil {
		// Return a copy of the operational data
		pmc := make([]types.PortBinding, 0, len(ep.portMapping))
		for _, pm := range ep.portMapping {
			pmc = append(pmc, pm.GetCopy())
		}
		data[netlabel.PortMap] = pmc
	}

	if len(ep.macAddress) != 0 {
		data[netlabel.MacAddress] = ep.macAddress
	}
	return data, nil
}

// Join method is invoked when a Sandbox is attached to an endpoint.
func (d *driver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	network, err := d.getNetwork(nid)
	if err != nil {
		return err
	}

	// Ensure that the endpoint exists
	endpoint, err := network.getEndpoint(eid)
	if err != nil {
		return err
	}

	err = jinfo.SetGateway(endpoint.gateway)
	if err != nil {
		return err
	}

	// This is just a stub for now

	jinfo.DisableGatewayService()
	return nil
}

// Leave method is invoked when a Sandbox detaches from an endpoint.
func (d *driver) Leave(nid, eid string) error {
	network, err := d.getNetwork(nid)
	if err != nil {
		return types.InternalMaskableErrorf("%s", err)
	}

	// Ensure that the endpoint exists
	_, err = network.getEndpoint(eid)
	if err != nil {
		return err
	}

	// This is just a stub for now

	return nil
}

func (d *driver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	return nil
}

func (d *driver) RevokeExternalConnectivity(nid, eid string) error {
	return nil
}

func (d *driver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	return nil, types.NotImplementedErrorf("not implemented")
}

func (d *driver) NetworkFree(id string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) Type() string {
	return d.name
}

func (d *driver) IsBuiltIn() bool {
	return true
}

// DiscoverNew is a notification for a new discovery event, such as a new node joining a cluster
func (d *driver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

// DiscoverDelete is a notification for a discovery delete event, such as a node leaving a cluster
func (d *driver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

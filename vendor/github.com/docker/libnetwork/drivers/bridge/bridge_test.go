package bridge

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"regexp"
	"testing"

	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/ipamutils"
	"github.com/docker/libnetwork/iptables"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/netutils"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
	"github.com/vishvananda/netlink"
)

func init() {
	ipamutils.InitNetworks()
}

func TestEndpointMarshalling(t *testing.T) {
	ip1, _ := types.ParseCIDR("172.22.0.9/16")
	ip2, _ := types.ParseCIDR("2001:db8::9")
	mac, _ := net.ParseMAC("ac:bd:24:57:66:77")
	e := &bridgeEndpoint{
		id:         "d2c015a1fe5930650cbcd50493efba0500bcebd8ee1f4401a16319f8a567de33",
		nid:        "ee33fbb43c323f1920b6b35a0101552ac22ede960d0e5245e9738bccc68b2415",
		addr:       ip1,
		addrv6:     ip2,
		macAddress: mac,
		srcName:    "veth123456",
		config:     &endpointConfiguration{MacAddress: mac},
		containerConfig: &containerConfiguration{
			ParentEndpoints: []string{"one", "due", "three"},
			ChildEndpoints:  []string{"four", "five", "six"},
		},
		extConnConfig: &connectivityConfiguration{
			ExposedPorts: []types.TransportPort{
				{
					Proto: 6,
					Port:  uint16(18),
				},
			},
			PortBindings: []types.PortBinding{
				{
					Proto:       6,
					IP:          net.ParseIP("17210.33.9.56"),
					Port:        uint16(18),
					HostPort:    uint16(3000),
					HostPortEnd: uint16(14000),
				},
			},
		},
		portMapping: []types.PortBinding{
			{
				Proto:       17,
				IP:          net.ParseIP("172.33.9.56"),
				Port:        uint16(99),
				HostIP:      net.ParseIP("10.10.100.2"),
				HostPort:    uint16(9900),
				HostPortEnd: uint16(10000),
			},
			{
				Proto:       6,
				IP:          net.ParseIP("171.33.9.56"),
				Port:        uint16(55),
				HostIP:      net.ParseIP("10.11.100.2"),
				HostPort:    uint16(5500),
				HostPortEnd: uint16(55000),
			},
		},
	}

	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}

	ee := &bridgeEndpoint{}
	err = json.Unmarshal(b, ee)
	if err != nil {
		t.Fatal(err)
	}

	if e.id != ee.id || e.nid != ee.nid || e.srcName != ee.srcName || !bytes.Equal(e.macAddress, ee.macAddress) ||
		!types.CompareIPNet(e.addr, ee.addr) || !types.CompareIPNet(e.addrv6, ee.addrv6) ||
		!compareEpConfig(e.config, ee.config) ||
		!compareContainerConfig(e.containerConfig, ee.containerConfig) ||
		!compareConnConfig(e.extConnConfig, ee.extConnConfig) ||
		!compareBindings(e.portMapping, ee.portMapping) {
		t.Fatalf("JSON marsh/unmarsh failed.\nOriginal:\n%#v\nDecoded:\n%#v", e, ee)
	}
}

func compareEpConfig(a, b *endpointConfiguration) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	return bytes.Equal(a.MacAddress, b.MacAddress)
}

func compareContainerConfig(a, b *containerConfiguration) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if len(a.ParentEndpoints) != len(b.ParentEndpoints) ||
		len(a.ChildEndpoints) != len(b.ChildEndpoints) {
		return false
	}
	for i := 0; i < len(a.ParentEndpoints); i++ {
		if a.ParentEndpoints[i] != b.ParentEndpoints[i] {
			return false
		}
	}
	for i := 0; i < len(a.ChildEndpoints); i++ {
		if a.ChildEndpoints[i] != b.ChildEndpoints[i] {
			return false
		}
	}
	return true
}

func compareConnConfig(a, b *connectivityConfiguration) bool {
	if a == b {
		return true
	}
	if a == nil || b == nil {
		return false
	}
	if len(a.ExposedPorts) != len(b.ExposedPorts) ||
		len(a.PortBindings) != len(b.PortBindings) {
		return false
	}
	for i := 0; i < len(a.ExposedPorts); i++ {
		if !a.ExposedPorts[i].Equal(&b.ExposedPorts[i]) {
			return false
		}
	}
	for i := 0; i < len(a.PortBindings); i++ {
		if !a.PortBindings[i].Equal(&b.PortBindings[i]) {
			return false
		}
	}
	return true
}

func compareBindings(a, b []types.PortBinding) bool {
	if len(a) != len(b) {
		return false
	}
	for i := 0; i < len(a); i++ {
		if !a[i].Equal(&b[i]) {
			return false
		}
	}
	return true
}

func getIPv4Data(t *testing.T, iface string) []driverapi.IPAMData {
	ipd := driverapi.IPAMData{AddressSpace: "full"}
	nws, _, err := netutils.ElectInterfaceAddresses(iface)
	if err != nil {
		t.Fatal(err)
	}
	ipd.Pool = nws[0]
	// Set network gateway to X.X.X.1
	ipd.Gateway = types.GetIPNetCopy(nws[0])
	ipd.Gateway.IP[len(ipd.Gateway.IP)-1] = 1
	return []driverapi.IPAMData{ipd}
}

func TestCreateFullOptions(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}
	d := newDriver()

	config := &configuration{
		EnableIPForwarding: true,
		EnableIPTables:     true,
	}

	// Test this scenario: Default gw address does not belong to
	// container network and it's greater than bridge address
	cnw, _ := types.ParseCIDR("172.16.122.0/24")
	bnw, _ := types.ParseCIDR("172.16.0.0/24")
	br, _ := types.ParseCIDR("172.16.0.1/16")
	defgw, _ := types.ParseCIDR("172.16.0.100/16")

	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	netOption := make(map[string]interface{})
	netOption[netlabel.EnableIPv6] = true
	netOption[netlabel.GenericData] = &networkConfiguration{
		BridgeName: DefaultBridgeName,
	}

	ipdList := []driverapi.IPAMData{
		{
			Pool:         bnw,
			Gateway:      br,
			AuxAddresses: map[string]*net.IPNet{DefaultGatewayV4AuxKey: defgw},
		},
	}
	err := d.CreateNetwork("dummy", netOption, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	// Verify the IP address allocated for the endpoint belongs to the container network
	epOptions := make(map[string]interface{})
	te := newTestEndpoint(cnw, 10)
	err = d.CreateEndpoint("dummy", "ep1", te.Interface(), epOptions)
	if err != nil {
		t.Fatalf("Failed to create an endpoint : %s", err.Error())
	}

	if !cnw.Contains(te.Interface().Address().IP) {
		t.Fatalf("endpoint got assigned address outside of container network(%s): %s", cnw.String(), te.Interface().Address())
	}
}

func TestCreateNoConfig(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}
	d := newDriver()

	netconfig := &networkConfiguration{BridgeName: DefaultBridgeName}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	if err := d.CreateNetwork("dummy", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}
}

func TestCreateFullOptionsLabels(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}
	d := newDriver()

	config := &configuration{
		EnableIPForwarding: true,
	}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	bndIPs := "127.0.0.1"
	nwV6s := "2001:db8:2600:2700:2800::/80"
	gwV6s := "2001:db8:2600:2700:2800::25/80"
	nwV6, _ := types.ParseCIDR(nwV6s)
	gwV6, _ := types.ParseCIDR(gwV6s)

	labels := map[string]string{
		BridgeName:         DefaultBridgeName,
		DefaultBridge:      "true",
		EnableICC:          "true",
		EnableIPMasquerade: "true",
		DefaultBindingIP:   bndIPs,
	}

	netOption := make(map[string]interface{})
	netOption[netlabel.EnableIPv6] = true
	netOption[netlabel.GenericData] = labels

	ipdList := getIPv4Data(t, "")
	ipd6List := []driverapi.IPAMData{
		{
			Pool: nwV6,
			AuxAddresses: map[string]*net.IPNet{
				DefaultGatewayV6AuxKey: gwV6,
			},
		},
	}

	err := d.CreateNetwork("dummy", netOption, nil, ipdList, ipd6List)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	nw, ok := d.networks["dummy"]
	if !ok {
		t.Fatal("Cannot find dummy network in bridge driver")
	}

	if nw.config.BridgeName != DefaultBridgeName {
		t.Fatal("incongruent name in bridge network")
	}

	if !nw.config.EnableIPv6 {
		t.Fatal("incongruent EnableIPv6 in bridge network")
	}

	if !nw.config.EnableICC {
		t.Fatal("incongruent EnableICC in bridge network")
	}

	if !nw.config.EnableIPMasquerade {
		t.Fatal("incongruent EnableIPMasquerade in bridge network")
	}

	bndIP := net.ParseIP(bndIPs)
	if !bndIP.Equal(nw.config.DefaultBindingIP) {
		t.Fatalf("Unexpected: %v", nw.config.DefaultBindingIP)
	}

	if !types.CompareIPNet(nw.config.AddressIPv6, nwV6) {
		t.Fatalf("Unexpected: %v", nw.config.AddressIPv6)
	}

	if !gwV6.IP.Equal(nw.config.DefaultGatewayIPv6) {
		t.Fatalf("Unexpected: %v", nw.config.DefaultGatewayIPv6)
	}

	// In short here we are testing --fixed-cidr-v6 daemon option
	// plus --mac-address run option
	mac, _ := net.ParseMAC("aa:bb:cc:dd:ee:ff")
	epOptions := map[string]interface{}{netlabel.MacAddress: mac}
	te := newTestEndpoint(ipdList[0].Pool, 20)
	err = d.CreateEndpoint("dummy", "ep1", te.Interface(), epOptions)
	if err != nil {
		t.Fatal(err)
	}

	if !nwV6.Contains(te.Interface().AddressIPv6().IP) {
		t.Fatalf("endpoint got assigned address outside of container network(%s): %s", nwV6.String(), te.Interface().AddressIPv6())
	}
	if te.Interface().AddressIPv6().IP.String() != "2001:db8:2600:2700:2800:aabb:ccdd:eeff" {
		t.Fatalf("Unexpected endpoint IPv6 address: %v", te.Interface().AddressIPv6().IP)
	}
}

func TestCreate(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	d := newDriver()

	if err := d.configure(nil); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	netconfig := &networkConfiguration{BridgeName: DefaultBridgeName}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	if err := d.CreateNetwork("dummy", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	err := d.CreateNetwork("dummy", genericOption, nil, getIPv4Data(t, ""), nil)
	if err == nil {
		t.Fatal("Expected bridge driver to refuse creation of second network with default name")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatal("Creation of second network with default name failed with unexpected error type")
	}
}

func TestCreateFail(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	d := newDriver()

	if err := d.configure(nil); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	netconfig := &networkConfiguration{BridgeName: "dummy0", DefaultBridge: true}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	if err := d.CreateNetwork("dummy", genericOption, nil, getIPv4Data(t, ""), nil); err == nil {
		t.Fatal("Bridge creation was expected to fail")
	}
}

func TestCreateMultipleNetworks(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	d := newDriver()

	config := &configuration{
		EnableIPTables: true,
	}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	config1 := &networkConfiguration{BridgeName: "net_test_1"}
	genericOption = make(map[string]interface{})
	genericOption[netlabel.GenericData] = config1
	if err := d.CreateNetwork("1", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	config2 := &networkConfiguration{BridgeName: "net_test_2"}
	genericOption[netlabel.GenericData] = config2
	if err := d.CreateNetwork("2", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	// Verify the network isolation rules are installed, each network subnet should appear 2 times
	verifyV4INCEntries(d.networks, 2, t)

	config3 := &networkConfiguration{BridgeName: "net_test_3"}
	genericOption[netlabel.GenericData] = config3
	if err := d.CreateNetwork("3", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	// Verify the network isolation rules are installed, each network subnet should appear 4 times
	verifyV4INCEntries(d.networks, 6, t)

	config4 := &networkConfiguration{BridgeName: "net_test_4"}
	genericOption[netlabel.GenericData] = config4
	if err := d.CreateNetwork("4", genericOption, nil, getIPv4Data(t, ""), nil); err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	// Now 6 times
	verifyV4INCEntries(d.networks, 12, t)

	d.DeleteNetwork("1")
	verifyV4INCEntries(d.networks, 6, t)

	d.DeleteNetwork("2")
	verifyV4INCEntries(d.networks, 2, t)

	d.DeleteNetwork("3")
	verifyV4INCEntries(d.networks, 0, t)

	d.DeleteNetwork("4")
	verifyV4INCEntries(d.networks, 0, t)
}

func verifyV4INCEntries(networks map[string]*bridgeNetwork, numEntries int, t *testing.T) {
	out, err := iptables.Raw("-nvL", IsolationChain)
	if err != nil {
		t.Fatal(err)
	}

	found := 0
	for _, x := range networks {
		for _, y := range networks {
			if x == y {
				continue
			}
			re := regexp.MustCompile(x.config.BridgeName + " " + y.config.BridgeName)
			matches := re.FindAllString(string(out[:]), -1)
			if len(matches) != 1 {
				t.Fatalf("Cannot find expected inter-network isolation rules in IP Tables:\n%s", string(out[:]))
			}
			found++
		}
	}

	if found != numEntries {
		t.Fatalf("Cannot find expected number (%d) of inter-network isolation rules in IP Tables:\n%s\nFound %d", numEntries, string(out[:]), found)
	}
}

type testInterface struct {
	mac     net.HardwareAddr
	addr    *net.IPNet
	addrv6  *net.IPNet
	srcName string
	dstName string
}

type testEndpoint struct {
	iface          *testInterface
	gw             net.IP
	gw6            net.IP
	hostsPath      string
	resolvConfPath string
	routes         []types.StaticRoute
}

func newTestEndpoint(nw *net.IPNet, ordinal byte) *testEndpoint {
	addr := types.GetIPNetCopy(nw)
	addr.IP[len(addr.IP)-1] = ordinal
	return &testEndpoint{iface: &testInterface{addr: addr}}
}

func (te *testEndpoint) Interface() driverapi.InterfaceInfo {
	if te.iface != nil {
		return te.iface
	}

	return nil
}

func (i *testInterface) MacAddress() net.HardwareAddr {
	return i.mac
}

func (i *testInterface) Address() *net.IPNet {
	return i.addr
}

func (i *testInterface) AddressIPv6() *net.IPNet {
	return i.addrv6
}

func (i *testInterface) SetMacAddress(mac net.HardwareAddr) error {
	if i.mac != nil {
		return types.ForbiddenErrorf("endpoint interface MAC address present (%s). Cannot be modified with %s.", i.mac, mac)
	}
	if mac == nil {
		return types.BadRequestErrorf("tried to set nil MAC address to endpoint interface")
	}
	i.mac = types.GetMacCopy(mac)
	return nil
}

func (i *testInterface) SetIPAddress(address *net.IPNet) error {
	if address.IP == nil {
		return types.BadRequestErrorf("tried to set nil IP address to endpoint interface")
	}
	if address.IP.To4() == nil {
		return setAddress(&i.addrv6, address)
	}
	return setAddress(&i.addr, address)
}

func setAddress(ifaceAddr **net.IPNet, address *net.IPNet) error {
	if *ifaceAddr != nil {
		return types.ForbiddenErrorf("endpoint interface IP present (%s). Cannot be modified with (%s).", *ifaceAddr, address)
	}
	*ifaceAddr = types.GetIPNetCopy(address)
	return nil
}

func (i *testInterface) SetNames(srcName string, dstName string) error {
	i.srcName = srcName
	i.dstName = dstName
	return nil
}

func (te *testEndpoint) InterfaceName() driverapi.InterfaceNameInfo {
	if te.iface != nil {
		return te.iface
	}

	return nil
}

func (te *testEndpoint) SetGateway(gw net.IP) error {
	te.gw = gw
	return nil
}

func (te *testEndpoint) SetGatewayIPv6(gw6 net.IP) error {
	te.gw6 = gw6
	return nil
}

func (te *testEndpoint) AddStaticRoute(destination *net.IPNet, routeType int, nextHop net.IP) error {
	te.routes = append(te.routes, types.StaticRoute{Destination: destination, RouteType: routeType, NextHop: nextHop})
	return nil
}

func (te *testEndpoint) AddTableEntry(tableName string, key string, value []byte) error {
	return nil
}

func (te *testEndpoint) DisableGatewayService() {}

func TestQueryEndpointInfo(t *testing.T) {
	testQueryEndpointInfo(t, true)
}

func TestQueryEndpointInfoHairpin(t *testing.T) {
	testQueryEndpointInfo(t, false)
}

func testQueryEndpointInfo(t *testing.T, ulPxyEnabled bool) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}
	d := newDriver()

	config := &configuration{
		EnableIPTables:      true,
		EnableUserlandProxy: ulPxyEnabled,
	}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	netconfig := &networkConfiguration{
		BridgeName: DefaultBridgeName,
		EnableICC:  false,
	}
	genericOption = make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	ipdList := getIPv4Data(t, "")
	err := d.CreateNetwork("net1", genericOption, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	sbOptions := make(map[string]interface{})
	sbOptions[netlabel.PortMap] = getPortMapping()

	te := newTestEndpoint(ipdList[0].Pool, 11)
	err = d.CreateEndpoint("net1", "ep1", te.Interface(), nil)
	if err != nil {
		t.Fatalf("Failed to create an endpoint : %s", err.Error())
	}

	err = d.Join("net1", "ep1", "sbox", te, sbOptions)
	if err != nil {
		t.Fatalf("Failed to join the endpoint: %v", err)
	}

	err = d.ProgramExternalConnectivity("net1", "ep1", sbOptions)
	if err != nil {
		t.Fatalf("Failed to program external connectivity: %v", err)
	}

	network, ok := d.networks["net1"]
	if !ok {
		t.Fatalf("Cannot find network %s inside driver", "net1")
	}
	ep, _ := network.endpoints["ep1"]
	data, err := d.EndpointOperInfo(network.id, ep.id)
	if err != nil {
		t.Fatalf("Failed to ask for endpoint operational data:  %v", err)
	}
	pmd, ok := data[netlabel.PortMap]
	if !ok {
		t.Fatal("Endpoint operational data does not contain port mapping data")
	}
	pm, ok := pmd.([]types.PortBinding)
	if !ok {
		t.Fatal("Unexpected format for port mapping in endpoint operational data")
	}
	if len(ep.portMapping) != len(pm) {
		t.Fatal("Incomplete data for port mapping in endpoint operational data")
	}
	for i, pb := range ep.portMapping {
		if !pb.Equal(&pm[i]) {
			t.Fatal("Unexpected data for port mapping in endpoint operational data")
		}
	}

	err = d.RevokeExternalConnectivity("net1", "ep1")
	if err != nil {
		t.Fatal(err)
	}

	// release host mapped ports
	err = d.Leave("net1", "ep1")
	if err != nil {
		t.Fatal(err)
	}
}

func getExposedPorts() []types.TransportPort {
	return []types.TransportPort{
		{Proto: types.TCP, Port: uint16(5000)},
		{Proto: types.UDP, Port: uint16(400)},
		{Proto: types.TCP, Port: uint16(600)},
	}
}

func getPortMapping() []types.PortBinding {
	return []types.PortBinding{
		{Proto: types.TCP, Port: uint16(230), HostPort: uint16(23000)},
		{Proto: types.UDP, Port: uint16(200), HostPort: uint16(22000)},
		{Proto: types.TCP, Port: uint16(120), HostPort: uint16(12000)},
	}
}

func TestLinkContainers(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	d := newDriver()

	config := &configuration{
		EnableIPTables: true,
	}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = config

	if err := d.configure(genericOption); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	netconfig := &networkConfiguration{
		BridgeName: DefaultBridgeName,
		EnableICC:  false,
	}
	genericOption = make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	ipdList := getIPv4Data(t, "")
	err := d.CreateNetwork("net1", genericOption, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	te1 := newTestEndpoint(ipdList[0].Pool, 11)
	err = d.CreateEndpoint("net1", "ep1", te1.Interface(), nil)
	if err != nil {
		t.Fatalf("Failed to create an endpoint : %s", err.Error())
	}

	exposedPorts := getExposedPorts()
	sbOptions := make(map[string]interface{})
	sbOptions[netlabel.ExposedPorts] = exposedPorts

	err = d.Join("net1", "ep1", "sbox", te1, sbOptions)
	if err != nil {
		t.Fatalf("Failed to join the endpoint: %v", err)
	}

	err = d.ProgramExternalConnectivity("net1", "ep1", sbOptions)
	if err != nil {
		t.Fatalf("Failed to program external connectivity: %v", err)
	}

	addr1 := te1.iface.addr
	if addr1.IP.To4() == nil {
		t.Fatal("No Ipv4 address assigned to the endpoint:  ep1")
	}

	te2 := newTestEndpoint(ipdList[0].Pool, 22)
	err = d.CreateEndpoint("net1", "ep2", te2.Interface(), nil)
	if err != nil {
		t.Fatalf("Failed to create an endpoint : %s", err.Error())
	}

	addr2 := te2.iface.addr
	if addr2.IP.To4() == nil {
		t.Fatal("No Ipv4 address assigned to the endpoint:  ep2")
	}

	sbOptions = make(map[string]interface{})
	sbOptions[netlabel.GenericData] = options.Generic{
		"ChildEndpoints": []string{"ep1"},
	}

	err = d.Join("net1", "ep2", "", te2, sbOptions)
	if err != nil {
		t.Fatal("Failed to link ep1 and ep2")
	}

	err = d.ProgramExternalConnectivity("net1", "ep2", sbOptions)
	if err != nil {
		t.Fatalf("Failed to program external connectivity: %v", err)
	}

	out, err := iptables.Raw("-L", DockerChain)
	for _, pm := range exposedPorts {
		regex := fmt.Sprintf("%s dpt:%d", pm.Proto.String(), pm.Port)
		re := regexp.MustCompile(regex)
		matches := re.FindAllString(string(out[:]), -1)
		if len(matches) != 1 {
			t.Fatalf("IP Tables programming failed %s", string(out[:]))
		}

		regex = fmt.Sprintf("%s spt:%d", pm.Proto.String(), pm.Port)
		matched, _ := regexp.MatchString(regex, string(out[:]))
		if !matched {
			t.Fatalf("IP Tables programming failed %s", string(out[:]))
		}
	}

	err = d.RevokeExternalConnectivity("net1", "ep2")
	if err != nil {
		t.Fatalf("Failed to revoke external connectivity: %v", err)
	}

	err = d.Leave("net1", "ep2")
	if err != nil {
		t.Fatal("Failed to unlink ep1 and ep2")
	}

	out, err = iptables.Raw("-L", DockerChain)
	for _, pm := range exposedPorts {
		regex := fmt.Sprintf("%s dpt:%d", pm.Proto.String(), pm.Port)
		re := regexp.MustCompile(regex)
		matches := re.FindAllString(string(out[:]), -1)
		if len(matches) != 0 {
			t.Fatalf("Leave should have deleted relevant IPTables rules  %s", string(out[:]))
		}

		regex = fmt.Sprintf("%s spt:%d", pm.Proto.String(), pm.Port)
		matched, _ := regexp.MatchString(regex, string(out[:]))
		if matched {
			t.Fatalf("Leave should have deleted relevant IPTables rules  %s", string(out[:]))
		}
	}

	// Error condition test with an invalid endpoint-id "ep4"
	sbOptions = make(map[string]interface{})
	sbOptions[netlabel.GenericData] = options.Generic{
		"ChildEndpoints": []string{"ep1", "ep4"},
	}

	err = d.Join("net1", "ep2", "", te2, sbOptions)
	if err != nil {
		t.Fatal(err)
	}
	err = d.ProgramExternalConnectivity("net1", "ep2", sbOptions)
	if err != nil {
		out, err = iptables.Raw("-L", DockerChain)
		for _, pm := range exposedPorts {
			regex := fmt.Sprintf("%s dpt:%d", pm.Proto.String(), pm.Port)
			re := regexp.MustCompile(regex)
			matches := re.FindAllString(string(out[:]), -1)
			if len(matches) != 0 {
				t.Fatalf("Error handling should rollback relevant IPTables rules  %s", string(out[:]))
			}

			regex = fmt.Sprintf("%s spt:%d", pm.Proto.String(), pm.Port)
			matched, _ := regexp.MatchString(regex, string(out[:]))
			if matched {
				t.Fatalf("Error handling should rollback relevant IPTables rules  %s", string(out[:]))
			}
		}
	} else {
		t.Fatal("Expected Join to fail given link conditions are not satisfied")
	}
}

func TestValidateConfig(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Test mtu
	c := networkConfiguration{Mtu: -2}
	err := c.Validate()
	if err == nil {
		t.Fatal("Failed to detect invalid MTU number")
	}

	c.Mtu = 9000
	err = c.Validate()
	if err != nil {
		t.Fatal("unexpected validation error on MTU number")
	}

	// Bridge network
	_, network, _ := net.ParseCIDR("172.28.0.0/16")
	c = networkConfiguration{
		AddressIPv4: network,
	}

	err = c.Validate()
	if err != nil {
		t.Fatal(err)
	}

	// Test v4 gw
	c.DefaultGatewayIPv4 = net.ParseIP("172.27.30.234")
	err = c.Validate()
	if err == nil {
		t.Fatal("Failed to detect invalid default gateway")
	}

	c.DefaultGatewayIPv4 = net.ParseIP("172.28.30.234")
	err = c.Validate()
	if err != nil {
		t.Fatal("Unexpected validation error on default gateway")
	}

	// Test v6 gw
	_, v6nw, _ := net.ParseCIDR("2001:db8:ae:b004::/64")
	c = networkConfiguration{
		EnableIPv6:         true,
		AddressIPv6:        v6nw,
		DefaultGatewayIPv6: net.ParseIP("2001:db8:ac:b004::bad:a55"),
	}
	err = c.Validate()
	if err == nil {
		t.Fatal("Failed to detect invalid v6 default gateway")
	}

	c.DefaultGatewayIPv6 = net.ParseIP("2001:db8:ae:b004::bad:a55")
	err = c.Validate()
	if err != nil {
		t.Fatal("Unexpected validation error on v6 default gateway")
	}

	c.AddressIPv6 = nil
	err = c.Validate()
	if err == nil {
		t.Fatal("Failed to detect invalid v6 default gateway")
	}

	c.AddressIPv6 = nil
	err = c.Validate()
	if err == nil {
		t.Fatal("Failed to detect invalid v6 default gateway")
	}
}

func TestSetDefaultGw(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	d := newDriver()

	if err := d.configure(nil); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	_, subnetv6, _ := net.ParseCIDR("2001:db8:ea9:9abc:b0c4::/80")

	ipdList := getIPv4Data(t, "")
	gw4 := types.GetIPCopy(ipdList[0].Pool.IP).To4()
	gw4[3] = 254
	gw6 := net.ParseIP("2001:db8:ea9:9abc:b0c4::254")

	config := &networkConfiguration{
		BridgeName:         DefaultBridgeName,
		AddressIPv6:        subnetv6,
		DefaultGatewayIPv4: gw4,
		DefaultGatewayIPv6: gw6,
	}

	genericOption := make(map[string]interface{})
	genericOption[netlabel.EnableIPv6] = true
	genericOption[netlabel.GenericData] = config

	err := d.CreateNetwork("dummy", genericOption, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}

	te := newTestEndpoint(ipdList[0].Pool, 10)
	err = d.CreateEndpoint("dummy", "ep", te.Interface(), nil)
	if err != nil {
		t.Fatalf("Failed to create endpoint: %v", err)
	}

	err = d.Join("dummy", "ep", "sbox", te, nil)
	if err != nil {
		t.Fatalf("Failed to join endpoint: %v", err)
	}

	if !gw4.Equal(te.gw) {
		t.Fatalf("Failed to configure default gateway. Expected %v. Found %v", gw4, te.gw)
	}

	if !gw6.Equal(te.gw6) {
		t.Fatalf("Failed to configure default gateway. Expected %v. Found %v", gw6, te.gw6)
	}
}

func TestCleanupIptableRules(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()
	bridgeChain := []iptables.ChainInfo{
		{Name: DockerChain, Table: iptables.Nat},
		{Name: DockerChain, Table: iptables.Filter},
		{Name: IsolationChain, Table: iptables.Filter},
	}
	if _, _, _, err := setupIPChains(&configuration{EnableIPTables: true}); err != nil {
		t.Fatalf("Error setting up ip chains: %v", err)
	}
	for _, chainInfo := range bridgeChain {
		if !iptables.ExistChain(chainInfo.Name, chainInfo.Table) {
			t.Fatalf("iptables chain %s of %s table should have been created", chainInfo.Name, chainInfo.Table)
		}
	}
	removeIPChains()
	for _, chainInfo := range bridgeChain {
		if iptables.ExistChain(chainInfo.Name, chainInfo.Table) {
			t.Fatalf("iptables chain %s of %s table should have been deleted", chainInfo.Name, chainInfo.Table)
		}
	}
}

func TestCreateWithExistingBridge(t *testing.T) {
	defer testutils.SetupTestOSContext(t)()
	d := newDriver()

	if err := d.configure(nil); err != nil {
		t.Fatalf("Failed to setup driver config: %v", err)
	}

	brName := "br111"
	br := &netlink.Bridge{
		LinkAttrs: netlink.LinkAttrs{
			Name: brName,
		},
	}
	if err := netlink.LinkAdd(br); err != nil {
		t.Fatalf("Failed to create bridge interface: %v", err)
	}
	defer netlink.LinkDel(br)
	if err := netlink.LinkSetUp(br); err != nil {
		t.Fatalf("Failed to set bridge interface up: %v", err)
	}

	ip := net.IP{192, 168, 122, 1}
	addr := &netlink.Addr{IPNet: &net.IPNet{
		IP:   ip,
		Mask: net.IPv4Mask(255, 255, 255, 0),
	}}
	if err := netlink.AddrAdd(br, addr); err != nil {
		t.Fatalf("Failed to add IP address to bridge: %v", err)
	}

	netconfig := &networkConfiguration{BridgeName: brName}
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = netconfig

	if err := d.CreateNetwork(brName, genericOption, nil, getIPv4Data(t, brName), nil); err != nil {
		t.Fatalf("Failed to create bridge network: %v", err)
	}

	nw, err := d.getNetwork(brName)
	if err != nil {
		t.Fatalf("Failed to getNetwork(%s): %v", brName, err)
	}

	addrs4, _, err := nw.bridge.addresses()
	if err != nil {
		t.Fatalf("Failed to get the bridge network's address: %v", err)
	}

	if !addrs4[0].IP.Equal(ip) {
		t.Fatal("Creating bridge network with existing bridge interface unexpectedly modified the IP address of the bridge")
	}

	if err := d.DeleteNetwork(brName); err != nil {
		t.Fatalf("Failed to delete network %s: %v", brName, err)
	}

	if _, err := netlink.LinkByName(brName); err != nil {
		t.Fatal("Deleting bridge network that using existing bridge interface unexpectedly deleted the bridge interface")
	}
}

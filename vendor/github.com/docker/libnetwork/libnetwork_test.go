package libnetwork_test

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"sync"
	"testing"

	"github.com/docker/docker/pkg/plugins"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/config"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netns"
)

const (
	bridgeNetType = "bridge"
)

var controller libnetwork.NetworkController

func TestMain(m *testing.M) {
	if reexec.Init() {
		return
	}

	if err := createController(); err != nil {
		logrus.Errorf("Error creating controller: %v", err)
		os.Exit(1)
	}

	x := m.Run()
	controller.Stop()
	os.Exit(x)
}

func createController() error {
	var err error

	// Cleanup local datastore file
	os.Remove(datastore.DefaultScopes("")[datastore.LocalScope].Client.Address)

	option := options.Generic{
		"EnableIPForwarding": true,
	}

	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = option

	cfgOptions, err := libnetwork.OptionBoltdbWithRandomDBFile()
	if err != nil {
		return err
	}
	controller, err = libnetwork.New(append(cfgOptions, config.OptionDriverConfig(bridgeNetType, genericOption))...)
	return err
}

func createTestNetwork(networkType, networkName string, netOption options.Generic, ipamV4Configs, ipamV6Configs []*libnetwork.IpamConf) (libnetwork.Network, error) {
	return controller.NewNetwork(networkType, networkName, "",
		libnetwork.NetworkOptionGeneric(netOption),
		libnetwork.NetworkOptionIpam(ipamapi.DefaultIPAM, "", ipamV4Configs, ipamV6Configs, nil))
}

func getEmptyGenericOption() map[string]interface{} {
	genericOption := make(map[string]interface{})
	genericOption[netlabel.GenericData] = options.Generic{}
	return genericOption
}

func getPortMapping() []types.PortBinding {
	return []types.PortBinding{
		{Proto: types.TCP, Port: uint16(230), HostPort: uint16(23000)},
		{Proto: types.UDP, Port: uint16(200), HostPort: uint16(22000)},
		{Proto: types.TCP, Port: uint16(120), HostPort: uint16(12000)},
		{Proto: types.TCP, Port: uint16(320), HostPort: uint16(32000), HostPortEnd: uint16(32999)},
		{Proto: types.UDP, Port: uint16(420), HostPort: uint16(42000), HostPortEnd: uint16(42001)},
	}
}

func TestNull(t *testing.T) {
	cnt, err := controller.NewSandbox("null_container",
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	if err != nil {
		t.Fatal(err)
	}

	network, err := createTestNetwork("null", "testnull", options.Generic{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	ep, err := network.CreateEndpoint("testep")
	if err != nil {
		t.Fatal(err)
	}

	err = ep.Join(cnt)
	if err != nil {
		t.Fatal(err)
	}

	err = ep.Leave(cnt)
	if err != nil {
		t.Fatal(err)
	}

	if err := ep.Delete(false); err != nil {
		t.Fatal(err)
	}

	if err := cnt.Delete(); err != nil {
		t.Fatal(err)
	}

	// host type is special network. Cannot be removed.
	err = network.Delete()
	if err == nil {
		t.Fatal(err)
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Unexpected error type")
	}
}

func TestBridge(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.EnableIPv6: true,
		netlabel.GenericData: options.Generic{
			"BridgeName":         "testnetwork",
			"EnableICC":          true,
			"EnableIPMasquerade": true,
		},
	}
	ipamV4ConfList := []*libnetwork.IpamConf{{PreferredPool: "192.168.100.0/24", Gateway: "192.168.100.1"}}
	ipamV6ConfList := []*libnetwork.IpamConf{{PreferredPool: "fe90::/64", Gateway: "fe90::22"}}

	network, err := createTestNetwork(bridgeNetType, "testnetwork", netOption, ipamV4ConfList, ipamV6ConfList)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := network.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := network.CreateEndpoint("testep")
	if err != nil {
		t.Fatal(err)
	}

	sb, err := controller.NewSandbox(containerID, libnetwork.OptionPortMapping(getPortMapping()))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(sb)
	if err != nil {
		t.Fatal(err)
	}

	epInfo, err := ep.DriverInfo()
	if err != nil {
		t.Fatal(err)
	}
	pmd, ok := epInfo[netlabel.PortMap]
	if !ok {
		t.Fatalf("Could not find expected info in endpoint data")
	}
	pm, ok := pmd.([]types.PortBinding)
	if !ok {
		t.Fatalf("Unexpected format for port mapping in endpoint operational data")
	}
	if len(pm) != 5 {
		t.Fatalf("Incomplete data for port mapping in endpoint operational data: %d", len(pm))
	}
}

func TestUnknownDriver(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	_, err := createTestNetwork("unknowndriver", "testnetwork", options.Generic{}, nil, nil)
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(types.NotFoundError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestNilRemoteDriver(t *testing.T) {
	_, err := controller.NewNetwork("framerelay", "dummy", "",
		libnetwork.NetworkOptionGeneric(getEmptyGenericOption()))
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(types.NotFoundError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestNetworkName(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}

	_, err := createTestNetwork(bridgeNetType, "", netOption, nil, nil)
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(libnetwork.ErrInvalidName); !ok {
		t.Fatalf("Expected to fail with ErrInvalidName error. Got %v", err)
	}

	networkName := "testnetwork"
	n, err := createTestNetwork(bridgeNetType, networkName, netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	if n.Name() != networkName {
		t.Fatalf("Expected network name %s, got %s", networkName, n.Name())
	}
}

func TestNetworkType(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}

	n, err := createTestNetwork(bridgeNetType, "testnetwork", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	if n.Type() != bridgeNetType {
		t.Fatalf("Expected network type %s, got %s", bridgeNetType, n.Type())
	}
}

func TestNetworkID(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}

	n, err := createTestNetwork(bridgeNetType, "testnetwork", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	if n.ID() == "" {
		t.Fatal("Expected non-empty network id")
	}
}

func TestDeleteNetworkWithActiveEndpoints(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		"BridgeName": "testnetwork",
	}
	option := options.Generic{
		netlabel.GenericData: netOption,
	}

	network, err := createTestNetwork(bridgeNetType, "testnetwork", option, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	ep, err := network.CreateEndpoint("testep")
	if err != nil {
		t.Fatal(err)
	}

	err = network.Delete()
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(*libnetwork.ActiveEndpointsError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}

	// Done testing. Now cleanup.
	if err := ep.Delete(false); err != nil {
		t.Fatal(err)
	}

	if err := network.Delete(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworkConfig(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Verify config network cannot inherit another config network
	configNetwork, err := controller.NewNetwork("bridge", "config_network0", "",
		libnetwork.NetworkOptionConfigOnly(),
		libnetwork.NetworkOptionConfigFrom("anotherConfigNw"))

	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}

	// Create supported config network
	netOption := options.Generic{
		"EnableICC": false,
	}
	option := options.Generic{
		netlabel.GenericData: netOption,
	}
	ipamV4ConfList := []*libnetwork.IpamConf{{PreferredPool: "192.168.100.0/24", SubPool: "192.168.100.128/25", Gateway: "192.168.100.1"}}
	ipamV6ConfList := []*libnetwork.IpamConf{{PreferredPool: "2001:db8:abcd::/64", SubPool: "2001:db8:abcd::ef99/80", Gateway: "2001:db8:abcd::22"}}

	netOptions := []libnetwork.NetworkOption{
		libnetwork.NetworkOptionConfigOnly(),
		libnetwork.NetworkOptionEnableIPv6(true),
		libnetwork.NetworkOptionGeneric(option),
		libnetwork.NetworkOptionIpam("default", "", ipamV4ConfList, ipamV6ConfList, nil),
	}

	configNetwork, err = controller.NewNetwork(bridgeNetType, "config_network0", "", netOptions...)
	if err != nil {
		t.Fatal(err)
	}

	// Verify a config-only network cannot be created with network operator configurations
	for i, opt := range []libnetwork.NetworkOption{
		libnetwork.NetworkOptionInternalNetwork(),
		libnetwork.NetworkOptionAttachable(true),
		libnetwork.NetworkOptionIngress(true),
	} {
		_, err = controller.NewNetwork(bridgeNetType, "testBR", "",
			libnetwork.NetworkOptionConfigOnly(), opt)
		if err == nil {
			t.Fatalf("Expected to fail. But instead succeeded for option: %d", i)
		}
		if _, ok := err.(types.ForbiddenError); !ok {
			t.Fatalf("Did not fail with expected error. Actual error: %v", err)
		}
	}

	// Verify a network cannot be created with both config-from and network specific configurations
	for i, opt := range []libnetwork.NetworkOption{
		libnetwork.NetworkOptionEnableIPv6(true),
		libnetwork.NetworkOptionIpam("my-ipam", "", nil, nil, nil),
		libnetwork.NetworkOptionIpam("", "", ipamV4ConfList, nil, nil),
		libnetwork.NetworkOptionIpam("", "", nil, ipamV6ConfList, nil),
		libnetwork.NetworkOptionLabels(map[string]string{"number": "two"}),
		libnetwork.NetworkOptionDriverOpts(map[string]string{"com.docker.network.mtu": "1600"}),
	} {
		_, err = controller.NewNetwork(bridgeNetType, "testBR", "",
			libnetwork.NetworkOptionConfigFrom("config_network0"), opt)
		if err == nil {
			t.Fatalf("Expected to fail. But instead succeeded for option: %d", i)
		}
		if _, ok := err.(types.ForbiddenError); !ok {
			t.Fatalf("Did not fail with expected error. Actual error: %v", err)
		}
	}

	// Create a valid network
	network, err := controller.NewNetwork(bridgeNetType, "testBR", "",
		libnetwork.NetworkOptionConfigFrom("config_network0"))
	if err != nil {
		t.Fatal(err)
	}

	// Verify the config network cannot be removed
	err = configNetwork.Delete()
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}

	// Delete network
	if err := network.Delete(); err != nil {
		t.Fatal(err)
	}

	// Verify the config network can now be removed
	if err := configNetwork.Delete(); err != nil {
		t.Fatal(err)
	}

}

func TestUnknownNetwork(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		"BridgeName": "testnetwork",
	}
	option := options.Generic{
		netlabel.GenericData: netOption,
	}

	network, err := createTestNetwork(bridgeNetType, "testnetwork", option, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = network.Delete()
	if err != nil {
		t.Fatal(err)
	}

	err = network.Delete()
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(*libnetwork.UnknownNetworkError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestUnknownEndpoint(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		"BridgeName": "testnetwork",
	}
	option := options.Generic{
		netlabel.GenericData: netOption,
	}
	ipamV4ConfList := []*libnetwork.IpamConf{{PreferredPool: "192.168.100.0/24"}}

	network, err := createTestNetwork(bridgeNetType, "testnetwork", option, ipamV4ConfList, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = network.CreateEndpoint("")
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}
	if _, ok := err.(libnetwork.ErrInvalidName); !ok {
		t.Fatalf("Expected to fail with ErrInvalidName error. Actual error: %v", err)
	}

	ep, err := network.CreateEndpoint("testep")
	if err != nil {
		t.Fatal(err)
	}

	err = ep.Delete(false)
	if err != nil {
		t.Fatal(err)
	}

	// Done testing. Now cleanup
	if err := network.Delete(); err != nil {
		t.Fatal(err)
	}
}

func TestNetworkEndpointsWalkers(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Create network 1 and add 2 endpoint: ep11, ep12
	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network1",
		},
	}

	net1, err := createTestNetwork(bridgeNetType, "network1", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := net1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep11, err := net1.CreateEndpoint("ep11")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep11.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	ep12, err := net1.CreateEndpoint("ep12")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep12.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	// Test list methods on net1
	epList1 := net1.Endpoints()
	if len(epList1) != 2 {
		t.Fatalf("Endpoints() returned wrong number of elements: %d instead of 2", len(epList1))
	}
	// endpoint order is not guaranteed
	for _, e := range epList1 {
		if e != ep11 && e != ep12 {
			t.Fatal("Endpoints() did not return all the expected elements")
		}
	}

	// Test Endpoint Walk method
	var epName string
	var epWanted libnetwork.Endpoint
	wlk := func(ep libnetwork.Endpoint) bool {
		if ep.Name() == epName {
			epWanted = ep
			return true
		}
		return false
	}

	// Look for ep1 on network1
	epName = "ep11"
	net1.WalkEndpoints(wlk)
	if epWanted == nil {
		t.Fatal(err)
	}
	if ep11 != epWanted {
		t.Fatal(err)
	}

	current := len(controller.Networks())

	// Create network 2
	netOption = options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network2",
		},
	}

	net2, err := createTestNetwork(bridgeNetType, "network2", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := net2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	// Test Networks method
	if len(controller.Networks()) != current+1 {
		t.Fatalf("Did not find the expected number of networks")
	}

	// Test Network Walk method
	var netName string
	var netWanted libnetwork.Network
	nwWlk := func(nw libnetwork.Network) bool {
		if nw.Name() == netName {
			netWanted = nw
			return true
		}
		return false
	}

	// Look for network named "network1" and "network2"
	netName = "network1"
	controller.WalkNetworks(nwWlk)
	if netWanted == nil {
		t.Fatal(err)
	}
	if net1.ID() != netWanted.ID() {
		t.Fatal(err)
	}

	netName = "network2"
	controller.WalkNetworks(nwWlk)
	if netWanted == nil {
		t.Fatal(err)
	}
	if net2.ID() != netWanted.ID() {
		t.Fatal(err)
	}
}

func TestDuplicateEndpoint(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}
	n, err := createTestNetwork(bridgeNetType, "testnetwork", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	ep2, err := n.CreateEndpoint("ep1")
	defer func() {
		// Cleanup ep2 as well, else network cleanup might fail for failure cases
		if ep2 != nil {
			if err := ep2.Delete(false); err != nil {
				t.Fatal(err)
			}
		}
	}()

	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestControllerQuery(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Create network 1
	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network1",
		},
	}
	net1, err := createTestNetwork(bridgeNetType, "network1", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := net1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	// Create network 2
	netOption = options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network2",
		},
	}
	net2, err := createTestNetwork(bridgeNetType, "network2", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := net2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	_, err = controller.NetworkByName("")
	if err == nil {
		t.Fatalf("NetworkByName() succeeded with invalid target name")
	}
	if _, ok := err.(libnetwork.ErrInvalidName); !ok {
		t.Fatalf("Expected NetworkByName() to fail with ErrInvalidName error. Got: %v", err)
	}

	_, err = controller.NetworkByID("")
	if err == nil {
		t.Fatalf("NetworkByID() succeeded with invalid target id")
	}
	if _, ok := err.(libnetwork.ErrInvalidID); !ok {
		t.Fatalf("NetworkByID() failed with unexpected error: %v", err)
	}

	g, err := controller.NetworkByID("network1")
	if err == nil {
		t.Fatalf("Unexpected success for NetworkByID(): %v", g)
	}
	if _, ok := err.(libnetwork.ErrNoSuchNetwork); !ok {
		t.Fatalf("NetworkByID() failed with unexpected error: %v", err)
	}

	g, err = controller.NetworkByName("network1")
	if err != nil {
		t.Fatalf("Unexpected failure for NetworkByName(): %v", err)
	}
	if g == nil {
		t.Fatalf("NetworkByName() did not find the network")
	}

	if g != net1 {
		t.Fatalf("NetworkByName() returned the wrong network")
	}

	g, err = controller.NetworkByID(net1.ID())
	if err != nil {
		t.Fatalf("Unexpected failure for NetworkByID(): %v", err)
	}
	if net1.ID() != g.ID() {
		t.Fatalf("NetworkByID() returned unexpected element: %v", g)
	}

	g, err = controller.NetworkByName("network2")
	if err != nil {
		t.Fatalf("Unexpected failure for NetworkByName(): %v", err)
	}
	if g == nil {
		t.Fatalf("NetworkByName() did not find the network")
	}

	if g != net2 {
		t.Fatalf("NetworkByName() returned the wrong network")
	}

	g, err = controller.NetworkByID(net2.ID())
	if err != nil {
		t.Fatalf("Unexpected failure for NetworkByID(): %v", err)
	}
	if net2.ID() != g.ID() {
		t.Fatalf("NetworkByID() returned unexpected element: %v", g)
	}
}

func TestNetworkQuery(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Create network 1 and add 2 endpoint: ep11, ep12
	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network1",
		},
	}
	net1, err := createTestNetwork(bridgeNetType, "network1", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := net1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep11, err := net1.CreateEndpoint("ep11")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep11.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	ep12, err := net1.CreateEndpoint("ep12")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep12.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	e, err := net1.EndpointByName("ep11")
	if err != nil {
		t.Fatal(err)
	}
	if ep11 != e {
		t.Fatalf("EndpointByName() returned %v instead of %v", e, ep11)
	}

	e, err = net1.EndpointByName("")
	if err == nil {
		t.Fatalf("EndpointByName() succeeded with invalid target name")
	}
	if _, ok := err.(libnetwork.ErrInvalidName); !ok {
		t.Fatalf("Expected EndpointByName() to fail with ErrInvalidName error. Got: %v", err)
	}

	e, err = net1.EndpointByName("IamNotAnEndpoint")
	if err == nil {
		t.Fatalf("EndpointByName() succeeded with unknown target name")
	}
	if _, ok := err.(libnetwork.ErrNoSuchEndpoint); !ok {
		t.Fatal(err)
	}
	if e != nil {
		t.Fatalf("EndpointByName(): expected nil, got %v", e)
	}

	e, err = net1.EndpointByID(ep12.ID())
	if err != nil {
		t.Fatal(err)
	}
	if ep12.ID() != e.ID() {
		t.Fatalf("EndpointByID() returned %v instead of %v", e, ep12)
	}

	e, err = net1.EndpointByID("")
	if err == nil {
		t.Fatalf("EndpointByID() succeeded with invalid target id")
	}
	if _, ok := err.(libnetwork.ErrInvalidID); !ok {
		t.Fatalf("EndpointByID() failed with unexpected error: %v", err)
	}
}

const containerID = "valid_c"

type fakeSandbox struct{}

func (f *fakeSandbox) ID() string {
	return "fake sandbox"
}

func (f *fakeSandbox) ContainerID() string {
	return ""
}

func (f *fakeSandbox) Key() string {
	return "fake key"
}

func (f *fakeSandbox) Labels() map[string]interface{} {
	return nil
}

func (f *fakeSandbox) Statistics() (map[string]*types.InterfaceStatistics, error) {
	return nil, nil
}

func (f *fakeSandbox) Refresh(opts ...libnetwork.SandboxOption) error {
	return nil
}

func (f *fakeSandbox) Delete() error {
	return nil
}

func (f *fakeSandbox) Rename(name string) error {
	return nil
}

func (f *fakeSandbox) SetKey(key string) error {
	return nil
}

func (f *fakeSandbox) ResolveName(name string, ipType int) ([]net.IP, bool) {
	return nil, false
}

func (f *fakeSandbox) ResolveIP(ip string) string {
	return ""
}

func (f *fakeSandbox) ResolveService(name string) ([]*net.SRV, []net.IP) {
	return nil, nil
}

func (f *fakeSandbox) Endpoints() []libnetwork.Endpoint {
	return nil
}

func (f *fakeSandbox) EnableService() error {
	return nil
}

func (f *fakeSandbox) DisableService() error {
	return nil
}

func TestEndpointDeleteWithActiveContainer(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	n, err := createTestNetwork(bridgeNetType, "testnetwork", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	n2, err := createTestNetwork(bridgeNetType, "testnetwork2", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork2",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep.Delete(false)
		if err != nil {
			t.Fatal(err)
		}
	}()

	cnt, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	defer func() {
		if err := cnt.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(cnt)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep.Leave(cnt)
		if err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Delete(false)
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if _, ok := err.(*libnetwork.ActiveContainerError); !ok {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestEndpointMultipleJoins(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	n, err := createTestNetwork(bridgeNetType, "testmultiple", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testmultiple",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	sbx1, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	defer func() {
		if err := sbx1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	sbx2, err := controller.NewSandbox("c2")
	defer func() {
		if err := sbx2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(sbx1)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep.Leave(sbx1)
		if err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(sbx2)
	if err == nil {
		t.Fatal("Expected to fail multiple joins for the same endpoint")
	}

	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Failed with unexpected error type: %T. Desc: %s", err, err.Error())
	}

}

func TestLeaveAll(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	n, err := createTestNetwork(bridgeNetType, "testnetwork", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		// If this goes through, it means cnt.Delete() effectively detached from all the endpoints
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	n2, err := createTestNetwork(bridgeNetType, "testnetwork2", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork2",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep1, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}

	ep2, err := n2.CreateEndpoint("ep2")
	if err != nil {
		t.Fatal(err)
	}

	cnt, err := controller.NewSandbox("leaveall")
	if err != nil {
		t.Fatal(err)
	}

	err = ep1.Join(cnt)
	if err != nil {
		t.Fatalf("Failed to join ep1: %v", err)
	}

	err = ep2.Join(cnt)
	if err != nil {
		t.Fatalf("Failed to join ep2: %v", err)
	}

	err = cnt.Delete()
	if err != nil {
		t.Fatal(err)
	}
}

func TestContainerInvalidLeave(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	n, err := createTestNetwork(bridgeNetType, "testnetwork", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	cnt, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cnt.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Leave(cnt)
	if err == nil {
		t.Fatal("Expected to fail leave from an endpoint which has no active join")
	}
	if _, ok := err.(types.ForbiddenError); !ok {
		t.Fatalf("Failed with unexpected error type: %T. Desc: %s", err, err.Error())
	}

	if err = ep.Leave(nil); err == nil {
		t.Fatalf("Expected to fail leave nil Sandbox")
	}
	if _, ok := err.(types.BadRequestError); !ok {
		t.Fatalf("Unexpected error type returned: %T. Desc: %s", err, err.Error())
	}

	fsbx := &fakeSandbox{}
	if err = ep.Leave(fsbx); err == nil {
		t.Fatalf("Expected to fail leave with invalid Sandbox")
	}
	if _, ok := err.(types.BadRequestError); !ok {
		t.Fatalf("Unexpected error type returned: %T. Desc: %s", err, err.Error())
	}
}

func TestEndpointUpdateParent(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	n, err := createTestNetwork("bridge", "testnetwork", options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep1, err := n.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}

	ep2, err := n.CreateEndpoint("ep2")
	if err != nil {
		t.Fatal(err)
	}

	sbx1, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sbx1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	sbx2, err := controller.NewSandbox("c2",
		libnetwork.OptionHostname("test2"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionHostsPath("/var/lib/docker/test_network/container2/hosts"),
		libnetwork.OptionExtraHost("web", "192.168.0.2"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sbx2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep1.Join(sbx1)
	if err != nil {
		t.Fatal(err)
	}

	err = ep2.Join(sbx2)
	if err != nil {
		t.Fatal(err)
	}
}

func TestInvalidRemoteDriver(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		t.Skip("Skipping test when not running inside a Container")
	}

	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	if server == nil {
		t.Fatal("Failed to start an HTTP Server")
	}
	defer server.Close()

	type pluginRequest struct {
		name string
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintln(w, `{"Implements": ["InvalidDriver"]}`)
	})

	if err := os.MkdirAll("/etc/docker/plugins", 0755); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll("/etc/docker/plugins"); err != nil {
			t.Fatal(err)
		}
	}()

	if err := ioutil.WriteFile("/etc/docker/plugins/invalid-network-driver.spec", []byte(server.URL), 0644); err != nil {
		t.Fatal(err)
	}

	ctrlr, err := libnetwork.New()
	if err != nil {
		t.Fatal(err)
	}
	defer ctrlr.Stop()

	_, err = ctrlr.NewNetwork("invalid-network-driver", "dummy", "",
		libnetwork.NetworkOptionGeneric(getEmptyGenericOption()))
	if err == nil {
		t.Fatal("Expected to fail. But instead succeeded")
	}

	if err != plugins.ErrNotImplements {
		t.Fatalf("Did not fail with expected error. Actual error: %v", err)
	}
}

func TestValidRemoteDriver(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		t.Skip("Skipping test when not running inside a Container")
	}

	mux := http.NewServeMux()
	server := httptest.NewServer(mux)
	if server == nil {
		t.Fatal("Failed to start an HTTP Server")
	}
	defer server.Close()

	type pluginRequest struct {
		name string
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintf(w, `{"Implements": ["%s"]}`, driverapi.NetworkPluginEndpointType)
	})
	mux.HandleFunc(fmt.Sprintf("/%s.CreateNetwork", driverapi.NetworkPluginEndpointType), func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintf(w, "null")
	})

	if err := os.MkdirAll("/etc/docker/plugins", 0755); err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := os.RemoveAll("/etc/docker/plugins"); err != nil {
			t.Fatal(err)
		}
	}()

	if err := ioutil.WriteFile("/etc/docker/plugins/valid-network-driver.spec", []byte(server.URL), 0644); err != nil {
		t.Fatal(err)
	}

	n, err := controller.NewNetwork("valid-network-driver", "dummy", "",
		libnetwork.NetworkOptionGeneric(getEmptyGenericOption()))
	if err != nil {
		// Only fail if we could not find the plugin driver
		if _, ok := err.(types.NotFoundError); ok {
			t.Fatal(err)
		}
		return
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()
}

var (
	once   sync.Once
	start  = make(chan struct{})
	done   = make(chan chan struct{}, numThreads-1)
	origns = netns.None()
	testns = netns.None()
	sboxes = make([]libnetwork.Sandbox, numThreads)
)

const (
	iterCnt    = 25
	numThreads = 3
	first      = 1
	last       = numThreads
	debug      = false
)

func createGlobalInstance(t *testing.T) {
	var err error
	defer close(start)

	origns, err = netns.Get()
	if err != nil {
		t.Fatal(err)
	}

	if testutils.IsRunningInContainer() {
		testns = origns
	} else {
		testns, err = netns.New()
		if err != nil {
			t.Fatal(err)
		}
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "network",
		},
	}

	net1, err := controller.NetworkByName("testhost")
	if err != nil {
		t.Fatal(err)
	}

	net2, err := createTestNetwork("bridge", "network2", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, err = net1.CreateEndpoint("pep1")
	if err != nil {
		t.Fatal(err)
	}

	_, err = net2.CreateEndpoint("pep2")
	if err != nil {
		t.Fatal(err)
	}

	_, err = net2.CreateEndpoint("pep3")
	if err != nil {
		t.Fatal(err)
	}

	if sboxes[first-1], err = controller.NewSandbox(fmt.Sprintf("%drace", first), libnetwork.OptionUseDefaultSandbox()); err != nil {
		t.Fatal(err)
	}
	for thd := first + 1; thd <= last; thd++ {
		if sboxes[thd-1], err = controller.NewSandbox(fmt.Sprintf("%drace", thd)); err != nil {
			t.Fatal(err)
		}
	}
}

func debugf(format string, a ...interface{}) (int, error) {
	if debug {
		return fmt.Printf(format, a...)
	}

	return 0, nil
}

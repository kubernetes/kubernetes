// +build windows

package windows

import (
	"net"
	"testing"

	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
)

func testNetwork(networkType string, t *testing.T) {
	d := newDriver(networkType)
	bnw, _ := types.ParseCIDR("172.16.0.0/24")
	br, _ := types.ParseCIDR("172.16.0.1/16")

	netOption := make(map[string]interface{})
	networkOptions := map[string]string{
		NetworkName: "TestNetwork",
	}

	netOption[netlabel.GenericData] = networkOptions
	ipdList := []driverapi.IPAMData{
		{
			Pool:    bnw,
			Gateway: br,
		},
	}

	err := d.CreateNetwork("dummy", netOption, nil, ipdList, nil)
	if err != nil {
		t.Fatalf("Failed to create bridge: %v", err)
	}
	defer func() {
		err = d.DeleteNetwork("dummy")
		if err != nil {
			t.Fatalf("Failed to create bridge: %v", err)
		}
	}()

	epOptions := make(map[string]interface{})
	te := &testEndpoint{}
	err = d.CreateEndpoint("dummy", "ep1", te.Interface(), epOptions)
	if err != nil {
		t.Fatalf("Failed to create an endpoint : %s", err.Error())
	}

	err = d.DeleteEndpoint("dummy", "ep1")
	if err != nil {
		t.Fatalf("Failed to delete an endpoint : %s", err.Error())
	}
}

func TestNAT(t *testing.T) {
	testNetwork("nat", t)
}

func TestTransparent(t *testing.T) {
	testNetwork("transparent", t)
}

type testEndpoint struct {
	t                     *testing.T
	src                   string
	dst                   string
	address               string
	macAddress            string
	gateway               string
	disableGatewayService bool
}

func (test *testEndpoint) Interface() driverapi.InterfaceInfo {
	return test
}

func (test *testEndpoint) Address() *net.IPNet {
	if test.address == "" {
		return nil
	}
	nw, _ := types.ParseCIDR(test.address)
	return nw
}

func (test *testEndpoint) AddressIPv6() *net.IPNet {
	return nil
}

func (test *testEndpoint) MacAddress() net.HardwareAddr {
	if test.macAddress == "" {
		return nil
	}
	mac, _ := net.ParseMAC(test.macAddress)
	return mac
}

func (test *testEndpoint) SetMacAddress(mac net.HardwareAddr) error {
	if test.macAddress != "" {
		return types.ForbiddenErrorf("endpoint interface MAC address present (%s). Cannot be modified with %s.", test.macAddress, mac)
	}

	if mac == nil {
		return types.BadRequestErrorf("tried to set nil MAC address to endpoint interface")
	}
	test.macAddress = mac.String()
	return nil
}

func (test *testEndpoint) SetIPAddress(address *net.IPNet) error {
	if address.IP == nil {
		return types.BadRequestErrorf("tried to set nil IP address to endpoint interface")
	}

	test.address = address.String()
	return nil
}

func (test *testEndpoint) InterfaceName() driverapi.InterfaceNameInfo {
	return test
}

func (test *testEndpoint) SetGateway(ipv4 net.IP) error {
	return nil
}

func (test *testEndpoint) SetGatewayIPv6(ipv6 net.IP) error {
	return nil
}

func (test *testEndpoint) SetNames(src string, dst string) error {
	return nil
}

func (test *testEndpoint) AddStaticRoute(destination *net.IPNet, routeType int, nextHop net.IP) error {
	return nil
}

func (test *testEndpoint) DisableGatewayService() {
	test.disableGatewayService = true
}

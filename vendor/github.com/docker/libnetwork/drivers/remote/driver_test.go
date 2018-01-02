package remote

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/docker/docker/pkg/plugins"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func decodeToMap(r *http.Request) (res map[string]interface{}, err error) {
	err = json.NewDecoder(r.Body).Decode(&res)
	return
}

func handle(t *testing.T, mux *http.ServeMux, method string, h func(map[string]interface{}) interface{}) {
	mux.HandleFunc(fmt.Sprintf("/%s.%s", driverapi.NetworkPluginEndpointType, method), func(w http.ResponseWriter, r *http.Request) {
		ask, err := decodeToMap(r)
		if err != nil && err != io.EOF {
			t.Fatal(err)
		}
		answer := h(ask)
		err = json.NewEncoder(w).Encode(&answer)
		if err != nil {
			t.Fatal(err)
		}
	})
}

func setupPlugin(t *testing.T, name string, mux *http.ServeMux) func() {
	if err := os.MkdirAll("/etc/docker/plugins", 0755); err != nil {
		t.Fatal(err)
	}

	server := httptest.NewServer(mux)
	if server == nil {
		t.Fatal("Failed to start an HTTP Server")
	}

	if err := ioutil.WriteFile(fmt.Sprintf("/etc/docker/plugins/%s.spec", name), []byte(server.URL), 0644); err != nil {
		t.Fatal(err)
	}

	mux.HandleFunc("/Plugin.Activate", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/vnd.docker.plugins.v1+json")
		fmt.Fprintf(w, `{"Implements": ["%s"]}`, driverapi.NetworkPluginEndpointType)
	})

	return func() {
		if err := os.RemoveAll("/etc/docker/plugins"); err != nil {
			t.Fatal(err)
		}
		server.Close()
	}
}

type testEndpoint struct {
	t                     *testing.T
	src                   string
	dst                   string
	address               string
	addressIPv6           string
	macAddress            string
	gateway               string
	gatewayIPv6           string
	resolvConfPath        string
	hostsPath             string
	nextHop               string
	destination           string
	routeType             int
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
	if test.addressIPv6 == "" {
		return nil
	}
	nw, _ := types.ParseCIDR(test.addressIPv6)
	return nw
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
	if address.IP.To4() == nil {
		return setAddress(&test.addressIPv6, address)
	}
	return setAddress(&test.address, address)
}

func setAddress(ifaceAddr *string, address *net.IPNet) error {
	if *ifaceAddr != "" {
		return types.ForbiddenErrorf("endpoint interface IP present (%s). Cannot be modified with (%s).", *ifaceAddr, address)
	}
	*ifaceAddr = address.String()
	return nil
}

func (test *testEndpoint) InterfaceName() driverapi.InterfaceNameInfo {
	return test
}

func compareIPs(t *testing.T, kind string, shouldBe string, supplied net.IP) {
	ip := net.ParseIP(shouldBe)
	if ip == nil {
		t.Fatalf(`Invalid IP to test against: "%s"`, shouldBe)
	}
	if !ip.Equal(supplied) {
		t.Fatalf(`%s IPs are not equal: expected "%s", got %v`, kind, shouldBe, supplied)
	}
}

func compareIPNets(t *testing.T, kind string, shouldBe string, supplied net.IPNet) {
	_, net, _ := net.ParseCIDR(shouldBe)
	if net == nil {
		t.Fatalf(`Invalid IP network to test against: "%s"`, shouldBe)
	}
	if !types.CompareIPNet(net, &supplied) {
		t.Fatalf(`%s IP networks are not equal: expected "%s", got %v`, kind, shouldBe, supplied)
	}
}

func (test *testEndpoint) SetGateway(ipv4 net.IP) error {
	compareIPs(test.t, "Gateway", test.gateway, ipv4)
	return nil
}

func (test *testEndpoint) SetGatewayIPv6(ipv6 net.IP) error {
	compareIPs(test.t, "GatewayIPv6", test.gatewayIPv6, ipv6)
	return nil
}

func (test *testEndpoint) SetNames(src string, dst string) error {
	if test.src != src {
		test.t.Fatalf(`Wrong SrcName; expected "%s", got "%s"`, test.src, src)
	}
	if test.dst != dst {
		test.t.Fatalf(`Wrong DstPrefix; expected "%s", got "%s"`, test.dst, dst)
	}
	return nil
}

func (test *testEndpoint) AddStaticRoute(destination *net.IPNet, routeType int, nextHop net.IP) error {
	compareIPNets(test.t, "Destination", test.destination, *destination)
	compareIPs(test.t, "NextHop", test.nextHop, nextHop)

	if test.routeType != routeType {
		test.t.Fatalf(`Wrong RouteType; expected "%d", got "%d"`, test.routeType, routeType)
	}

	return nil
}

func (test *testEndpoint) DisableGatewayService() {
	test.disableGatewayService = true
}

func (test *testEndpoint) AddTableEntry(tableName string, key string, value []byte) error {
	return nil
}

func TestGetEmptyCapabilities(t *testing.T) {
	var plugin = "test-net-driver-empty-cap"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	handle(t, mux, "GetCapabilities", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}

	d := newDriver(plugin, p.Client())
	if d.Type() != plugin {
		t.Fatal("Driver type does not match that given")
	}

	_, err = d.(*driver).getCapabilities()
	if err == nil {
		t.Fatal("There should be error reported when get empty capability")
	}
}

func TestGetExtraCapabilities(t *testing.T) {
	var plugin = "test-net-driver-extra-cap"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	handle(t, mux, "GetCapabilities", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{
			"Scope":             "local",
			"foo":               "bar",
			"ConnectivityScope": "global",
		}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}

	d := newDriver(plugin, p.Client())
	if d.Type() != plugin {
		t.Fatal("Driver type does not match that given")
	}

	c, err := d.(*driver).getCapabilities()
	if err != nil {
		t.Fatal(err)
	} else if c.DataScope != datastore.LocalScope {
		t.Fatalf("get capability '%s', expecting 'local'", c.DataScope)
	} else if c.ConnectivityScope != datastore.GlobalScope {
		t.Fatalf("get capability '%s', expecting %q", c.ConnectivityScope, datastore.GlobalScope)
	}
}

func TestGetInvalidCapabilities(t *testing.T) {
	var plugin = "test-net-driver-invalid-cap"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	handle(t, mux, "GetCapabilities", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{
			"Scope": "fake",
		}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}

	d := newDriver(plugin, p.Client())
	if d.Type() != plugin {
		t.Fatal("Driver type does not match that given")
	}

	_, err = d.(*driver).getCapabilities()
	if err == nil {
		t.Fatal("There should be error reported when get invalid capability")
	}
}

func TestRemoteDriver(t *testing.T) {
	var plugin = "test-net-driver"

	ep := &testEndpoint{
		t:              t,
		src:            "vethsrc",
		dst:            "vethdst",
		address:        "192.168.5.7/16",
		addressIPv6:    "2001:DB8::5:7/48",
		macAddress:     "ab:cd:ef:ee:ee:ee",
		gateway:        "192.168.0.1",
		gatewayIPv6:    "2001:DB8::1",
		hostsPath:      "/here/comes/the/host/path",
		resolvConfPath: "/there/goes/the/resolv/conf",
		destination:    "10.0.0.0/8",
		nextHop:        "10.0.0.1",
		routeType:      1,
	}

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	var networkID string

	handle(t, mux, "GetCapabilities", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{
			"Scope": "global",
		}
	})
	handle(t, mux, "CreateNetwork", func(msg map[string]interface{}) interface{} {
		nid := msg["NetworkID"]
		var ok bool
		if networkID, ok = nid.(string); !ok {
			t.Fatal("RPC did not include network ID string")
		}
		return map[string]interface{}{}
	})
	handle(t, mux, "DeleteNetwork", func(msg map[string]interface{}) interface{} {
		if nid, ok := msg["NetworkID"]; !ok || nid != networkID {
			t.Fatal("Network ID missing or does not match that created")
		}
		return map[string]interface{}{}
	})
	handle(t, mux, "CreateEndpoint", func(msg map[string]interface{}) interface{} {
		iface := map[string]interface{}{
			"MacAddress":  ep.macAddress,
			"Address":     ep.address,
			"AddressIPv6": ep.addressIPv6,
		}
		return map[string]interface{}{
			"Interface": iface,
		}
	})
	handle(t, mux, "Join", func(msg map[string]interface{}) interface{} {
		options := msg["Options"].(map[string]interface{})
		foo, ok := options["foo"].(string)
		if !ok || foo != "fooValue" {
			t.Fatalf("Did not receive expected foo string in request options: %+v", msg)
		}
		return map[string]interface{}{
			"Gateway":        ep.gateway,
			"GatewayIPv6":    ep.gatewayIPv6,
			"HostsPath":      ep.hostsPath,
			"ResolvConfPath": ep.resolvConfPath,
			"InterfaceName": map[string]interface{}{
				"SrcName":   ep.src,
				"DstPrefix": ep.dst,
			},
			"StaticRoutes": []map[string]interface{}{
				{
					"Destination": ep.destination,
					"RouteType":   ep.routeType,
					"NextHop":     ep.nextHop,
				},
			},
		}
	})
	handle(t, mux, "Leave", func(msg map[string]interface{}) interface{} {
		return map[string]string{}
	})
	handle(t, mux, "DeleteEndpoint", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{}
	})
	handle(t, mux, "EndpointOperInfo", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{
			"Value": map[string]string{
				"Arbitrary": "key",
				"Value":     "pairs?",
			},
		}
	})
	handle(t, mux, "DiscoverNew", func(msg map[string]interface{}) interface{} {
		return map[string]string{}
	})
	handle(t, mux, "DiscoverDelete", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}

	d := newDriver(plugin, p.Client())
	if d.Type() != plugin {
		t.Fatal("Driver type does not match that given")
	}

	c, err := d.(*driver).getCapabilities()
	if err != nil {
		t.Fatal(err)
	} else if c.DataScope != datastore.GlobalScope {
		t.Fatalf("get capability '%s', expecting 'global'", c.DataScope)
	}

	netID := "dummy-network"
	err = d.CreateNetwork(netID, map[string]interface{}{}, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	endID := "dummy-endpoint"
	ifInfo := &testEndpoint{}
	err = d.CreateEndpoint(netID, endID, ifInfo, map[string]interface{}{})
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(ep.MacAddress(), ifInfo.MacAddress()) || !types.CompareIPNet(ep.Address(), ifInfo.Address()) ||
		!types.CompareIPNet(ep.AddressIPv6(), ifInfo.AddressIPv6()) {
		t.Fatalf("Unexpected InterfaceInfo data. Expected (%s, %s, %s). Got (%v, %v, %v)",
			ep.MacAddress(), ep.Address(), ep.AddressIPv6(),
			ifInfo.MacAddress(), ifInfo.Address(), ifInfo.AddressIPv6())
	}

	joinOpts := map[string]interface{}{"foo": "fooValue"}
	err = d.Join(netID, endID, "sandbox-key", ep, joinOpts)
	if err != nil {
		t.Fatal(err)
	}
	if _, err = d.EndpointOperInfo(netID, endID); err != nil {
		t.Fatal(err)
	}
	if err = d.Leave(netID, endID); err != nil {
		t.Fatal(err)
	}
	if err = d.DeleteEndpoint(netID, endID); err != nil {
		t.Fatal(err)
	}
	if err = d.DeleteNetwork(netID); err != nil {
		t.Fatal(err)
	}

	data := discoverapi.NodeDiscoveryData{
		Address: "192.168.1.1",
	}
	if err = d.DiscoverNew(discoverapi.NodeDiscovery, data); err != nil {
		t.Fatal(err)
	}
	if err = d.DiscoverDelete(discoverapi.NodeDiscovery, data); err != nil {
		t.Fatal(err)
	}
}

func TestDriverError(t *testing.T) {
	var plugin = "test-net-driver-error"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	handle(t, mux, "CreateEndpoint", func(msg map[string]interface{}) interface{} {
		return map[string]interface{}{
			"Err": "this should get raised as an error",
		}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}

	driver := newDriver(plugin, p.Client())

	if err := driver.CreateEndpoint("dummy", "dummy", &testEndpoint{t: t}, map[string]interface{}{}); err == nil {
		t.Fatal("Expected error from driver")
	}
}

func TestMissingValues(t *testing.T) {
	var plugin = "test-net-driver-missing"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	ep := &testEndpoint{
		t: t,
	}

	handle(t, mux, "CreateEndpoint", func(msg map[string]interface{}) interface{} {
		iface := map[string]interface{}{
			"Address":     ep.address,
			"AddressIPv6": ep.addressIPv6,
			"MacAddress":  ep.macAddress,
		}
		return map[string]interface{}{
			"Interface": iface,
		}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}
	driver := newDriver(plugin, p.Client())

	if err := driver.CreateEndpoint("dummy", "dummy", ep, map[string]interface{}{}); err != nil {
		t.Fatal(err)
	}
}

type rollbackEndpoint struct {
}

func (r *rollbackEndpoint) Interface() driverapi.InterfaceInfo {
	return r
}

func (r *rollbackEndpoint) MacAddress() net.HardwareAddr {
	return nil
}

func (r *rollbackEndpoint) Address() *net.IPNet {
	return nil
}

func (r *rollbackEndpoint) AddressIPv6() *net.IPNet {
	return nil
}

func (r *rollbackEndpoint) SetMacAddress(mac net.HardwareAddr) error {
	return errors.New("invalid mac")
}

func (r *rollbackEndpoint) SetIPAddress(ip *net.IPNet) error {
	return errors.New("invalid ip")
}

func TestRollback(t *testing.T) {
	var plugin = "test-net-driver-rollback"

	mux := http.NewServeMux()
	defer setupPlugin(t, plugin, mux)()

	rolledback := false

	handle(t, mux, "CreateEndpoint", func(msg map[string]interface{}) interface{} {
		iface := map[string]interface{}{
			"Address":     "192.168.4.5/16",
			"AddressIPv6": "",
			"MacAddress":  "7a:12:34:56:78:90",
		}
		return map[string]interface{}{
			"Interface": interface{}(iface),
		}
	})
	handle(t, mux, "DeleteEndpoint", func(msg map[string]interface{}) interface{} {
		rolledback = true
		return map[string]interface{}{}
	})

	p, err := plugins.Get(plugin, driverapi.NetworkPluginEndpointType)
	if err != nil {
		t.Fatal(err)
	}
	driver := newDriver(plugin, p.Client())

	ep := &rollbackEndpoint{}

	if err := driver.CreateEndpoint("dummy", "dummy", ep.Interface(), map[string]interface{}{}); err == nil {
		t.Fatal("Expected error from driver")
	}
	if !rolledback {
		t.Fatal("Expected to have had DeleteEndpoint called")
	}
}

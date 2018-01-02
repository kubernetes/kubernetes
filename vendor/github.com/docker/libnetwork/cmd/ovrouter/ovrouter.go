package main

import (
	"fmt"
	"net"
	"os"
	"os/signal"

	"github.com/docker/docker/pkg/plugingetter"
	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/drivers/overlay"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/vishvananda/netlink"
)

type router struct {
	d driverapi.Driver
}

type endpoint struct {
	addr *net.IPNet
	mac  net.HardwareAddr
	name string
}

func (r *router) GetPluginGetter() plugingetter.PluginGetter {
	return nil
}

func (r *router) RegisterDriver(name string, driver driverapi.Driver, c driverapi.Capability) error {
	r.d = driver
	return nil
}

func (ep *endpoint) Interface() driverapi.InterfaceInfo {
	return nil
}

func (ep *endpoint) SetMacAddress(mac net.HardwareAddr) error {
	if ep.mac != nil {
		return types.ForbiddenErrorf("endpoint interface MAC address present (%s). Cannot be modified with %s.", ep.mac, mac)
	}
	if mac == nil {
		return types.BadRequestErrorf("tried to set nil MAC address to endpoint interface")
	}
	ep.mac = types.GetMacCopy(mac)
	return nil
}

func (ep *endpoint) SetIPAddress(address *net.IPNet) error {
	if address.IP == nil {
		return types.BadRequestErrorf("tried to set nil IP address to endpoint interface")
	}
	if address.IP.To4() == nil {
		return types.NotImplementedErrorf("do not support ipv6 yet")
	}
	if ep.addr != nil {
		return types.ForbiddenErrorf("endpoint interface IP present (%s). Cannot be modified with %s.", ep.addr, address)
	}
	ep.addr = types.GetIPNetCopy(address)
	return nil
}

func (ep *endpoint) MacAddress() net.HardwareAddr {
	return types.GetMacCopy(ep.mac)
}

func (ep *endpoint) Address() *net.IPNet {
	return types.GetIPNetCopy(ep.addr)
}

func (ep *endpoint) AddressIPv6() *net.IPNet {
	return nil
}

func (ep *endpoint) InterfaceName() driverapi.InterfaceNameInfo {
	return ep
}

func (ep *endpoint) SetNames(srcName, dstPrefix string) error {
	ep.name = srcName
	return nil
}

func (ep *endpoint) SetGateway(net.IP) error {
	return nil
}

func (ep *endpoint) SetGatewayIPv6(net.IP) error {
	return nil
}

func (ep *endpoint) AddStaticRoute(destination *net.IPNet, routeType int,
	nextHop net.IP) error {
	return nil
}

func (ep *endpoint) AddTableEntry(tableName string, key string, value []byte) error {
	return nil
}

func (ep *endpoint) DisableGatewayService() {}

func main() {
	if reexec.Init() {
		return
	}

	opt := make(map[string]interface{})
	if len(os.Args) > 1 {
		opt[netlabel.OverlayBindInterface] = os.Args[1]
	}
	if len(os.Args) > 2 {
		opt[netlabel.OverlayNeighborIP] = os.Args[2]
	}
	if len(os.Args) > 3 {
		opt[netlabel.GlobalKVProvider] = os.Args[3]
	}
	if len(os.Args) > 4 {
		opt[netlabel.GlobalKVProviderURL] = os.Args[4]
	}

	r := &router{}
	if err := overlay.Init(r, opt); err != nil {
		fmt.Printf("Failed to initialize overlay driver: %v\n", err)
		os.Exit(1)
	}

	if err := r.d.CreateNetwork("testnetwork",
		map[string]interface{}{}, nil, nil, nil); err != nil {
		fmt.Printf("Failed to create network in the driver: %v\n", err)
		os.Exit(1)
	}

	ep := &endpoint{}
	if err := r.d.CreateEndpoint("testnetwork", "testep",
		ep, map[string]interface{}{}); err != nil {
		fmt.Printf("Failed to create endpoint in the driver: %v\n", err)
		os.Exit(1)
	}

	if err := r.d.Join("testnetwork", "testep",
		"", ep, map[string]interface{}{}); err != nil {
		fmt.Printf("Failed to join an endpoint in the driver: %v\n", err)
		os.Exit(1)
	}

	link, err := netlink.LinkByName(ep.name)
	if err != nil {
		fmt.Printf("Failed to find the container interface with name %s: %v\n",
			ep.name, err)
		os.Exit(1)
	}

	ipAddr := &netlink.Addr{IPNet: ep.addr, Label: ""}
	if err := netlink.AddrAdd(link, ipAddr); err != nil {
		fmt.Printf("Failed to add address to the interface: %v\n", err)
		os.Exit(1)
	}

	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt, os.Kill)

	for {
		select {
		case <-sigCh:
			r.d.Leave("testnetwork", "testep")
			overlay.Fini(r.d)
			os.Exit(0)
		}
	}
}

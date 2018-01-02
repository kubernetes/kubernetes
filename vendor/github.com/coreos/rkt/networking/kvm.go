// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// kvm.go file provides networking supporting functions for kvm flavor
package networking

import (
	"bufio"
	"crypto/sha512"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"

	"github.com/appc/spec/schema/types"
	"github.com/containernetworking/cni/pkg/ip"
	cnitypes "github.com/containernetworking/cni/pkg/types"
	cniutils "github.com/containernetworking/cni/pkg/utils"
	cnisysctl "github.com/containernetworking/cni/pkg/utils/sysctl"
	"github.com/hashicorp/errwrap"
	"github.com/vishvananda/netlink"

	"github.com/coreos/rkt/common"
	commonnet "github.com/coreos/rkt/common/networking"
	"github.com/coreos/rkt/networking/tuntap"
)

const (
	defaultBrName     = "kvm-cni0"
	defaultSubnetFile = "/run/flannel/subnet.env"
	defaultMTU        = 1500
)

type BridgeNetConf struct {
	NetConf
	BrName string `json:"bridge"`
	IsGw   bool   `json:"isGateway"`
}

// setupTapDevice creates persistent tap device
// and returns a newly created netlink.Link structure
func setupTapDevice(podID types.UUID) (netlink.Link, error) {
	// network device names are limited to 16 characters
	// the suffix %d will be replaced by the kernel with a suitable number
	nameTemplate := fmt.Sprintf("rkt-%s-tap%%d", podID.String()[0:4])
	ifName, err := tuntap.CreatePersistentIface(nameTemplate, tuntap.Tap)
	if err != nil {
		return nil, errwrap.Wrap(errors.New("tuntap persist"), err)
	}

	link, err := netlink.LinkByName(ifName)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("cannot find link %q", ifName), err)
	}

	if err := netlink.LinkSetUp(link); err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("cannot set link up %q", ifName), err)
	}
	return link, nil
}

type MacVTapNetConf struct {
	NetConf
	Master string `json:"master"`
	Mode   string `json:"mode"`
}

const (
	IPv4InterfaceArpProxySysctlTemplate = "net.ipv4.conf.%s.proxy_arp"
)

// setupTapDevice creates persistent macvtap device
// and returns a newly created netlink.Link structure
// using part of pod hash and interface number in interface name
func setupMacVTapDevice(podID types.UUID, config MacVTapNetConf, interfaceNumber int) (netlink.Link, error) {
	master, err := netlink.LinkByName(config.Master)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("cannot find master device '%v'", config.Master), err)
	}
	var mode netlink.MacvlanMode
	switch config.Mode {
	// if not set - defaults to bridge mode as in:
	// https://github.com/coreos/rkt/blob/master/Documentation/networking.md#macvlan
	case "", "bridge":
		mode = netlink.MACVLAN_MODE_BRIDGE
	case "private":
		mode = netlink.MACVLAN_MODE_PRIVATE
	case "vepa":
		mode = netlink.MACVLAN_MODE_VEPA
	case "passthru":
		mode = netlink.MACVLAN_MODE_PASSTHRU
	default:
		return nil, fmt.Errorf("unsupported macvtap mode: %v", config.Mode)
	}
	mtu := master.Attrs().MTU
	if config.MTU != 0 {
		mtu = config.MTU
	}
	interfaceName := fmt.Sprintf("rkt-%s-vtap%d", podID.String()[0:4], interfaceNumber)
	link := &netlink.Macvtap{
		Macvlan: netlink.Macvlan{
			LinkAttrs: netlink.LinkAttrs{
				Name:        interfaceName,
				MTU:         mtu,
				ParentIndex: master.Attrs().Index,
			},
			Mode: mode,
		},
	}

	if err := netlink.LinkAdd(link); err != nil {
		return nil, errwrap.Wrap(errors.New("cannot create macvtap interface"), err)
	}

	// TODO: duplicate following lines for ipv6 support, when it will be added in other places
	ipv4SysctlValueName := fmt.Sprintf(IPv4InterfaceArpProxySysctlTemplate, interfaceName)
	if _, err := cnisysctl.Sysctl(ipv4SysctlValueName, "1"); err != nil {
		// remove the newly added link and ignore errors, because we already are in a failed state
		_ = netlink.LinkDel(link)
		return nil, errwrap.Wrap(fmt.Errorf("failed to set proxy_arp on newly added interface %q", interfaceName), err)
	}

	if err := netlink.LinkSetUp(link); err != nil {
		// remove the newly added link and ignore errors, because we already are in a failed state
		_ = netlink.LinkDel(link)
		return nil, errwrap.Wrap(errors.New("cannot set up macvtap interface"), err)
	}
	return link, nil
}

// kvmSetupNetAddressing calls IPAM plugin (with a hack) to reserve an IP to be
// used by newly create tuntap pair
// in result it updates activeNet.runtime configuration
func kvmSetupNetAddressing(network *Networking, n activeNet, ifName string) error {
	// TODO: very ugly hack, that go through upper plugin, down to ipam plugin
	if err := ip.EnableIP4Forward(); err != nil {
		return errwrap.Wrap(errors.New("failed to enable forwarding"), err)
	}

	// patch plugin type only for single IPAM run time, then revert this change
	original_type := n.conf.Type
	n.conf.Type = n.conf.IPAM.Type
	output, err := network.execNetPlugin("ADD", &n, ifName)
	n.conf.Type = original_type
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("problem executing network plugin %q (%q)", n.conf.IPAM.Type, ifName), err)
	}

	result := cnitypes.Result{}
	if err = json.Unmarshal(output, &result); err != nil {
		return errwrap.Wrap(fmt.Errorf("error parsing %q result", n.conf.Name), err)
	}

	if result.IP4 == nil {
		return fmt.Errorf("net-plugin returned no IPv4 configuration")
	}

	n.runtime.MergeCNIResult(result)

	return nil
}

func ensureHasAddr(link netlink.Link, ipn *net.IPNet) error {
	addrs, err := netlink.AddrList(link, syscall.AF_INET)
	if err != nil && err != syscall.ENOENT {
		return errwrap.Wrap(errors.New("could not get list of IP addresses"), err)
	}

	// if there're no addresses on the interface, it's ok -- we'll add one
	if len(addrs) > 0 {
		ipnStr := ipn.String()
		for _, a := range addrs {
			// string comp is actually easiest for doing IPNet comps
			if a.IPNet.String() == ipnStr {
				return nil
			}
		}
		return fmt.Errorf("%q already has an IP address different from %v", link.Attrs().Name, ipn.String())
	}

	addr := &netlink.Addr{IPNet: ipn, Label: link.Attrs().Name}
	if err := netlink.AddrAdd(link, addr); err != nil {
		return errwrap.Wrap(fmt.Errorf("could not add IP address to %q", link.Attrs().Name), err)
	}
	return nil
}

func bridgeByName(name string) (*netlink.Bridge, error) {
	l, err := netlink.LinkByName(name)
	if err != nil {
		return nil, errwrap.Wrap(fmt.Errorf("could not lookup %q", name), err)
	}
	br, ok := l.(*netlink.Bridge)
	if !ok {
		return nil, fmt.Errorf("%q already exists but is not a bridge", name)
	}
	return br, nil
}

func ensureBridgeIsUp(brName string, mtu int) (*netlink.Bridge, error) {
	br := &netlink.Bridge{
		LinkAttrs: netlink.LinkAttrs{
			Name: brName,
			MTU:  mtu,
		},
	}

	if err := netlink.LinkAdd(br); err != nil {
		if err != syscall.EEXIST {
			return nil, errwrap.Wrap(fmt.Errorf("could not add %q", brName), err)
		}

		// it's ok if the device already exists as long as config is similar
		br, err = bridgeByName(brName)
		if err != nil {
			return nil, err
		}
	}

	if err := netlink.LinkSetUp(br); err != nil {
		return nil, err
	}

	return br, nil
}

func addRoute(link netlink.Link, podIP net.IP) error {
	route := netlink.Route{
		LinkIndex: link.Attrs().Index,
		Scope:     netlink.SCOPE_LINK,
		Dst: &net.IPNet{
			IP:   podIP,
			Mask: net.IPv4Mask(0xff, 0xff, 0xff, 0xff),
		},
	}
	return netlink.RouteAdd(&route)
}

func removeAllRoutesOnLink(link netlink.Link) error {
	routes, err := netlink.RouteList(link, netlink.FAMILY_V4)
	if err != nil {
		return errwrap.Wrap(fmt.Errorf("cannot list routes on link %q", link.Attrs().Name), err)
	}

	for _, route := range routes {
		if err := netlink.RouteDel(&route); err != nil {
			return errwrap.Wrap(fmt.Errorf("error in time of route removal for route %q", route), err)
		}
	}

	return nil
}

func getChainName(podUUIDString, confName string) string {
	h := sha512.Sum512([]byte(podUUIDString))
	return fmt.Sprintf("CNI-%s-%x", confName, h[:8])
}

type FlannelNetConf struct {
	NetConf

	SubnetFile string                 `json:"subnetFile"`
	Delegate   map[string]interface{} `json:"delegate"`
}

func loadFlannelNetConf(bytes []byte) (*FlannelNetConf, error) {
	n := &FlannelNetConf{
		SubnetFile: defaultSubnetFile,
	}
	if err := json.Unmarshal(bytes, n); err != nil {
		return nil, errwrap.Wrap(errors.New("failed to load netconf"), err)
	}
	return n, nil
}

type subnetEnv struct {
	nw     *net.IPNet
	sn     *net.IPNet
	mtu    int
	ipmasq bool
}

func loadFlannelSubnetEnv(fn string) (*subnetEnv, error) {
	f, err := os.Open(fn)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	se := &subnetEnv{}

	s := bufio.NewScanner(f)
	for s.Scan() {
		parts := strings.SplitN(s.Text(), "=", 2)
		switch parts[0] {
		case "FLANNEL_NETWORK":
			_, se.nw, err = net.ParseCIDR(parts[1])
			if err != nil {
				return nil, err
			}

		case "FLANNEL_SUBNET":
			_, se.sn, err = net.ParseCIDR(parts[1])
			if err != nil {
				return nil, err
			}

		case "FLANNEL_MTU":
			mtu, err := strconv.ParseUint(parts[1], 10, 32)
			if err != nil {
				return nil, err
			}
			se.mtu = int(mtu)

		case "FLANNEL_IPMASQ":
			se.ipmasq = parts[1] == "true"
		}
	}
	if err := s.Err(); err != nil {
		return nil, err
	}

	return se, nil
}

func hasKey(m map[string]interface{}, k string) bool {
	_, ok := m[k]
	return ok
}

func isString(i interface{}) bool {
	_, ok := i.(string)
	return ok
}

func kvmTransformFlannelNetwork(net *activeNet) error {
	n, err := loadFlannelNetConf(net.confBytes)
	if err != nil {
		return err
	}

	fenv, err := loadFlannelSubnetEnv(n.SubnetFile)
	if err != nil {
		return err
	}

	if n.Delegate == nil {
		n.Delegate = make(map[string]interface{})
	} else {
		if hasKey(n.Delegate, "type") && !isString(n.Delegate["type"]) {
			return fmt.Errorf("'delegate' dictionary, if present, must have (string) 'type' field")
		}
		if hasKey(n.Delegate, "name") {
			return fmt.Errorf("'delegate' dictionary must not have 'name' field, it'll be set by flannel")
		}
		if hasKey(n.Delegate, "ipam") {
			return fmt.Errorf("'delegate' dictionary must not have 'ipam' field, it'll be set by flannel")
		}
	}

	n.Delegate["name"] = n.Name

	if !hasKey(n.Delegate, "type") {
		n.Delegate["type"] = "bridge"
	}

	if !hasKey(n.Delegate, "isDefaultGateway") {
		n.Delegate["isDefaultGateway"] = false
	}

	if !hasKey(n.Delegate, "ipMasq") {
		// if flannel is not doing ipmasq, we should
		ipmasq := !fenv.ipmasq
		n.Delegate["ipMasq"] = ipmasq
	}

	if !hasKey(n.Delegate, "mtu") {
		mtu := fenv.mtu
		n.Delegate["mtu"] = mtu
	}

	if n.Delegate["type"].(string) == "bridge" {
		if !hasKey(n.Delegate, "isGateway") {
			n.Delegate["isGateway"] = true
		}
	}

	n.Delegate["ipam"] = map[string]interface{}{
		"type":   "host-local",
		"subnet": fenv.sn.String(),
		"routes": []cnitypes.Route{
			{
				Dst: *fenv.nw,
			},
		},
	}

	bytes, err := json.Marshal(n.Delegate)
	if err != nil {
		return errwrap.Wrap(errors.New("error in marshaling generated network settings"), err)
	}

	net.runtime.IP4 = &cnitypes.IPConfig{}
	*net = activeNet{
		confBytes: bytes,
		conf:      &NetConf{},
		runtime:   net.runtime,
	}
	net.conf.Name = n.Name
	net.conf.Type = n.Delegate["type"].(string)
	net.conf.IPMasq = n.Delegate["ipMasq"].(bool)
	net.conf.MTU = n.Delegate["mtu"].(int)
	net.conf.IsDefaultGateway = n.Delegate["isDefaultGateway"].(bool)
	net.conf.IPAM.Type = "host-local"
	return nil
}

// kvmSetup prepare new Networking to be used in kvm environment based on tuntap pair interfaces
// to allow communication with virtual machine created by lkvm tool
func kvmSetup(podRoot string, podID types.UUID, fps []commonnet.ForwardedPort, netList common.NetList, localConfig string, noDNS bool) (*Networking, error) {
	network := Networking{
		podEnv: podEnv{
			podRoot:      podRoot,
			podID:        podID,
			netsLoadList: netList,
			localConfig:  localConfig,
		},
	}
	var e error

	// If there's a network set as default in CNI configuration
	defaultGatewaySet := false

	_, defaultNet, err := net.ParseCIDR("0.0.0.0/0")
	if err != nil {
		return nil, errwrap.Wrap(errors.New("error when parsing net address"), err)
	}

	network.nets, e = network.loadNets()
	if e != nil {
		return nil, errwrap.Wrap(errors.New("error loading network definitions"), e)
	}

	// did stage0 already make /etc/rkt-resolv.conf (i.e. --dns passed)
	resolvPath := filepath.Join(common.Stage1RootfsPath(podRoot), "etc/rkt-resolv.conf")
	_, err = os.Stat(resolvPath)
	if err != nil && !os.IsNotExist(err) {
		return nil, errwrap.Wrap(fmt.Errorf("error statting /etc/rkt-resolv.conf"), err)
	}
	podHasResolvConf := err == nil

	for i, n := range network.nets {
		if n.conf.Type == "flannel" {
			if err := kvmTransformFlannelNetwork(&n); err != nil {
				return nil, errwrap.Wrap(errors.New("cannot transform flannel network into basic network"), err)
			}
		}
		switch n.conf.Type {
		case "ptp":
			link, err := setupTapDevice(podID)
			if err != nil {
				return nil, err
			}
			ifName := link.Attrs().Name
			n.runtime.IfName = ifName

			err = kvmSetupNetAddressing(&network, n, ifName)
			if err != nil {
				return nil, err
			}

			// add address to host tap device
			err = ensureHasAddr(
				link,
				&net.IPNet{
					IP:   n.runtime.IP4.Gateway,
					Mask: net.IPMask(n.runtime.Mask),
				},
			)
			if err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("cannot add address to host tap device %q", ifName), err)
			}

			if err := removeAllRoutesOnLink(link); err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("cannot remove route on host tap device %q", ifName), err)
			}

			if err := addRoute(link, n.runtime.IP); err != nil {
				return nil, errwrap.Wrap(errors.New("cannot add on host direct route to pod"), err)
			}

		case "bridge":
			config := BridgeNetConf{
				NetConf: NetConf{
					MTU: defaultMTU,
				},
				BrName: defaultBrName,
			}
			if err := json.Unmarshal(n.confBytes, &config); err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("error parsing %q result", n.conf.Name), err)
			}

			br, err := ensureBridgeIsUp(config.BrName, config.MTU)
			if err != nil {
				return nil, errwrap.Wrap(errors.New("error in time of bridge setup"), err)
			}
			link, err := setupTapDevice(podID)
			if err != nil {
				return nil, errwrap.Wrap(errors.New("can not setup tap device"), err)
			}
			err = netlink.LinkSetMaster(link, br)
			if err != nil {
				rErr := tuntap.RemovePersistentIface(n.runtime.IfName, tuntap.Tap)
				if rErr != nil {
					stderr.PrintE("warning: could not cleanup tap interface", rErr)
				}
				return nil, errwrap.Wrap(errors.New("can not add tap interface to bridge"), err)
			}

			ifName := link.Attrs().Name
			n.runtime.IfName = ifName

			err = kvmSetupNetAddressing(&network, n, ifName)
			if err != nil {
				return nil, err
			}

			if n.conf.IsDefaultGateway {
				n.runtime.IP4.Routes = append(
					n.runtime.IP4.Routes,
					cnitypes.Route{Dst: *defaultNet, GW: n.runtime.IP4.Gateway},
				)
				defaultGatewaySet = true
				config.IsGw = true
			}

			if config.IsGw {
				err = ensureHasAddr(
					br,
					&net.IPNet{
						IP:   n.runtime.IP4.Gateway,
						Mask: net.IPMask(n.runtime.Mask),
					},
				)

				if err != nil {
					return nil, errwrap.Wrap(fmt.Errorf("cannot add address to host bridge device %q", br.Name), err)
				}
			}

		case "macvlan":
			config := MacVTapNetConf{}
			if err := json.Unmarshal(n.confBytes, &config); err != nil {
				return nil, errwrap.Wrap(fmt.Errorf("error parsing %q result", n.conf.Name), err)
			}
			link, err := setupMacVTapDevice(podID, config, i)
			if err != nil {
				return nil, err
			}
			ifName := link.Attrs().Name
			n.runtime.IfName = ifName

			err = kvmSetupNetAddressing(&network, n, ifName)
			if err != nil {
				return nil, err
			}

		default:
			return nil, fmt.Errorf("network %q have unsupported type: %q", n.conf.Name, n.conf.Type)
		}

		// Check if there's any other network set as default gateway
		if defaultGatewaySet {
			for _, route := range n.runtime.IP4.Routes {
				if (defaultNet.String() == route.Dst.String()) && !n.conf.IsDefaultGateway {
					return nil, fmt.Errorf("flannel config enables default gateway and IPAM sets default gateway via %q", n.runtime.IP4.Gateway)
				}
			}
		}

		// Generate rkt-resolv.conf if it's not already there.
		// The first network plugin that supplies a non-empty
		// DNS response will win, unless noDNS is true (--dns passed to rkt run)
		if !common.IsDNSZero(&n.runtime.DNS) && !noDNS {
			if !podHasResolvConf {
				err := ioutil.WriteFile(
					resolvPath,
					[]byte(common.MakeResolvConf(n.runtime.DNS, "Generated by rkt from network "+n.conf.Name)),
					0644)
				if err != nil {
					return nil, errwrap.Wrap(fmt.Errorf("error creating resolv.conf"), err)
				}
				podHasResolvConf = true
			} else {
				stderr.Printf("Warning: network %v plugin specified DNS configuration, but DNS already supplied", n.conf.Name)
			}
		}

		if n.conf.IPMasq {
			chain := cniutils.FormatChainName(n.conf.Name, podID.String())
			comment := cniutils.FormatComment(n.conf.Name, podID.String())
			if err := ip.SetupIPMasq(&net.IPNet{
				IP:   n.runtime.IP,
				Mask: net.IPMask(n.runtime.Mask),
			}, chain, comment); err != nil {
				return nil, err
			}
		}
		network.nets[i] = n
	}
	podIP, err := network.GetForwardableNetPodIP()
	if err != nil {
		return nil, err
	}
	if err := network.setupForwarding(); err != nil {
		network.teardownForwarding()
		return nil, err
	}
	if err := network.forwardPorts(fps, podIP); err != nil {
		network.teardownForwarding()
		return nil, err
	}

	return &network, nil
}

/*
extend Networking struct with methods to clean up kvm specific network configurations
*/

// teardownKvmNets teardown every active networking from networking by
// removing tuntap interface and releasing its ip from IPAM plugin
func (n *Networking) teardownKvmNets() {
	for _, an := range n.nets {
		if an.conf.Type == "flannel" {
			if err := kvmTransformFlannelNetwork(&an); err != nil {
				stderr.PrintE("error transforming flannel network", err)
				continue
			}
		}

		switch an.conf.Type {
		case "ptp", "bridge":
			// remove tuntap interface
			tuntap.RemovePersistentIface(an.runtime.IfName, tuntap.Tap)

		case "macvlan":
			link, err := netlink.LinkByName(an.runtime.IfName)
			if err != nil {
				stderr.PrintE(fmt.Sprintf("cannot find link `%v`", an.runtime.IfName), err)
				continue
			} else {
				err := netlink.LinkDel(link)
				if err != nil {
					stderr.PrintE(fmt.Sprintf("cannot remove link `%v`", an.runtime.IfName), err)
					continue
				}
			}

		default:
			stderr.Printf("unsupported network type: %q", an.conf.Type)
			continue
		}
		// ugly hack again to directly call IPAM plugin to release IP
		an.conf.Type = an.conf.IPAM.Type

		_, err := n.execNetPlugin("DEL", &an, an.runtime.IfName)
		if err != nil {
			stderr.PrintE("error executing network plugin", err)
		}
		// remove masquerading if it was prepared
		if an.conf.IPMasq {
			chain := cniutils.FormatChainName(an.conf.Name, n.podID.String())
			comment := cniutils.FormatChainName(an.conf.Name, n.podID.String())
			err := ip.TeardownIPMasq(&net.IPNet{
				IP:   an.runtime.IP,
				Mask: net.IPMask(an.runtime.Mask),
			}, chain, comment)
			if err != nil {
				stderr.PrintE("error on removing masquerading", err)
			}
		}
	}
}

// kvmTeardown network teardown for kvm flavor based pods
// similar to Networking.Teardown but without host namespaces
func (n *Networking) kvmTeardown() {

	if err := n.teardownForwarding(); err != nil {
		stderr.PrintE("error removing forwarded ports (kvm)", err)
	}
	n.teardownKvmNets()
}

// Following methods implements behavior of netDescriber by activeNet
// (behavior required by stage1/init/kvm package and its kernel parameters configuration)

func (an activeNet) HostIP() net.IP {
	return an.runtime.HostIP
}
func (an activeNet) GuestIP() net.IP {
	return an.runtime.IP
}
func (an activeNet) IfName() string {
	if an.conf.Type == "macvlan" {
		// macvtap device passed as parameter to lkvm binary have different
		// kind of name, path to /dev/tapN made with N as link index
		link, err := netlink.LinkByName(an.runtime.IfName)
		if err != nil {
			stderr.PrintE(fmt.Sprintf("cannot get interface '%v'", an.runtime.IfName), err)
			return ""
		}
		return fmt.Sprintf("/dev/tap%d", link.Attrs().Index)
	}
	return an.runtime.IfName
}
func (an activeNet) Mask() net.IP {
	return an.runtime.Mask
}
func (an activeNet) Name() string {
	return an.conf.Name
}
func (an activeNet) IPMasq() bool {
	return an.conf.IPMasq
}
func (an activeNet) Gateway() net.IP {
	return an.runtime.IP4.Gateway
}
func (an activeNet) Routes() []cnitypes.Route {
	return an.runtime.IP4.Routes
}

// GetActiveNetworks returns activeNets to be used as NetDescriptors
// by plugins, which are required for stage1 executor to run (only for KVM)
func (e *Networking) GetActiveNetworks() []activeNet {
	return e.nets
}

package libnetwork_test

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"os/exec"
	"runtime"
	"strconv"
	"strings"
	"testing"

	"github.com/docker/docker/pkg/reexec"
	"github.com/docker/libnetwork"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/options"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
	"github.com/opencontainers/runc/libcontainer"
	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/sirupsen/logrus"
	"github.com/vishvananda/netlink"
	"github.com/vishvananda/netns"
)

func TestHost(t *testing.T) {
	sbx1, err := controller.NewSandbox("host_c1",
		libnetwork.OptionHostname("test1"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"),
		libnetwork.OptionUseDefaultSandbox())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sbx1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	sbx2, err := controller.NewSandbox("host_c2",
		libnetwork.OptionHostname("test2"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"),
		libnetwork.OptionUseDefaultSandbox())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sbx2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	network, err := createTestNetwork("host", "testhost", options.Generic{}, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	ep1, err := network.CreateEndpoint("testep1")
	if err != nil {
		t.Fatal(err)
	}

	if err := ep1.Join(sbx1); err != nil {
		t.Fatal(err)
	}

	ep2, err := network.CreateEndpoint("testep2")
	if err != nil {
		t.Fatal(err)
	}

	if err := ep2.Join(sbx2); err != nil {
		t.Fatal(err)
	}

	if err := ep1.Leave(sbx1); err != nil {
		t.Fatal(err)
	}

	if err := ep2.Leave(sbx2); err != nil {
		t.Fatal(err)
	}

	if err := ep1.Delete(false); err != nil {
		t.Fatal(err)
	}

	if err := ep2.Delete(false); err != nil {
		t.Fatal(err)
	}

	// Try to create another host endpoint and join/leave that.
	cnt3, err := controller.NewSandbox("host_c3",
		libnetwork.OptionHostname("test3"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"),
		libnetwork.OptionUseDefaultSandbox())
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := cnt3.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep3, err := network.CreateEndpoint("testep3")
	if err != nil {
		t.Fatal(err)
	}

	if err := ep3.Join(sbx2); err != nil {
		t.Fatal(err)
	}

	if err := ep3.Leave(sbx2); err != nil {
		t.Fatal(err)
	}

	if err := ep3.Delete(false); err != nil {
		t.Fatal(err)
	}
}

// Testing IPV6 from MAC address
func TestBridgeIpv6FromMac(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName":         "testipv6mac",
			"EnableICC":          true,
			"EnableIPMasquerade": true,
		},
	}
	ipamV4ConfList := []*libnetwork.IpamConf{{PreferredPool: "192.168.100.0/24", Gateway: "192.168.100.1"}}
	ipamV6ConfList := []*libnetwork.IpamConf{{PreferredPool: "fe90::/64", Gateway: "fe90::22"}}

	network, err := controller.NewNetwork(bridgeNetType, "testipv6mac", "",
		libnetwork.NetworkOptionGeneric(netOption),
		libnetwork.NetworkOptionEnableIPv6(true),
		libnetwork.NetworkOptionIpam(ipamapi.DefaultIPAM, "", ipamV4ConfList, ipamV6ConfList, nil),
		libnetwork.NetworkOptionDeferIPv6Alloc(true))
	if err != nil {
		t.Fatal(err)
	}

	mac := net.HardwareAddr{0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff}
	epOption := options.Generic{netlabel.MacAddress: mac}

	ep, err := network.CreateEndpoint("testep", libnetwork.EndpointOptionGeneric(epOption))
	if err != nil {
		t.Fatal(err)
	}

	iface := ep.Info().Iface()
	if !bytes.Equal(iface.MacAddress(), mac) {
		t.Fatalf("Unexpected mac address: %v", iface.MacAddress())
	}

	ip, expIP, _ := net.ParseCIDR("fe90::aabb:ccdd:eeff/64")
	expIP.IP = ip
	if !types.CompareIPNet(expIP, iface.AddressIPv6()) {
		t.Fatalf("Expected %v. Got: %v", expIP, iface.AddressIPv6())
	}

	if err := ep.Delete(false); err != nil {
		t.Fatal(err)
	}

	if err := network.Delete(); err != nil {
		t.Fatal(err)
	}
}

func checkSandbox(t *testing.T, info libnetwork.EndpointInfo) {
	key := info.Sandbox().Key()
	sbNs, err := netns.GetFromPath(key)
	if err != nil {
		t.Fatalf("Failed to get network namespace path %q: %v", key, err)
	}
	defer sbNs.Close()

	nh, err := netlink.NewHandleAt(sbNs)
	if err != nil {
		t.Fatal(err)
	}

	_, err = nh.LinkByName("eth0")
	if err != nil {
		t.Fatalf("Could not find the interface eth0 inside the sandbox: %v", err)
	}

	_, err = nh.LinkByName("eth1")
	if err != nil {
		t.Fatalf("Could not find the interface eth1 inside the sandbox: %v", err)
	}
}

func TestEndpointJoin(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	// Create network 1 and add 2 endpoint: ep11, ep12
	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName":         "testnetwork1",
			"EnableICC":          true,
			"EnableIPMasquerade": true,
		},
	}
	ipamV6ConfList := []*libnetwork.IpamConf{{PreferredPool: "fe90::/64", Gateway: "fe90::22"}}
	n1, err := controller.NewNetwork(bridgeNetType, "testnetwork1", "",
		libnetwork.NetworkOptionGeneric(netOption),
		libnetwork.NetworkOptionEnableIPv6(true),
		libnetwork.NetworkOptionIpam(ipamapi.DefaultIPAM, "", nil, ipamV6ConfList, nil),
		libnetwork.NetworkOptionDeferIPv6Alloc(true))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep1, err := n1.CreateEndpoint("ep1")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep1.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	// Validate if ep.Info() only gives me IP address info and not names and gateway during CreateEndpoint()
	info := ep1.Info()
	iface := info.Iface()
	if iface.Address() != nil && iface.Address().IP.To4() == nil {
		t.Fatalf("Invalid IP address returned: %v", iface.Address())
	}
	if iface.AddressIPv6() != nil && iface.AddressIPv6().IP == nil {
		t.Fatalf("Invalid IPv6 address returned: %v", iface.Address())
	}

	if len(info.Gateway()) != 0 {
		t.Fatalf("Expected empty gateway for an empty endpoint. Instead found a gateway: %v", info.Gateway())
	}
	if len(info.GatewayIPv6()) != 0 {
		t.Fatalf("Expected empty gateway for an empty ipv6 endpoint. Instead found a gateway: %v", info.GatewayIPv6())
	}

	if info.Sandbox() != nil {
		t.Fatalf("Expected an empty sandbox key for an empty endpoint. Instead found a non-empty sandbox key: %s", info.Sandbox().Key())
	}

	// test invalid joins
	err = ep1.Join(nil)
	if err == nil {
		t.Fatalf("Expected to fail join with nil Sandbox")
	}
	if _, ok := err.(types.BadRequestError); !ok {
		t.Fatalf("Unexpected error type returned: %T", err)
	}

	fsbx := &fakeSandbox{}
	if err = ep1.Join(fsbx); err == nil {
		t.Fatalf("Expected to fail join with invalid Sandbox")
	}
	if _, ok := err.(types.BadRequestError); !ok {
		t.Fatalf("Unexpected error type returned: %T", err)
	}

	sb, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	if err != nil {
		t.Fatal(err)
	}

	defer func() {
		if err := sb.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep1.Join(sb)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep1.Leave(sb)
		if err != nil {
			t.Fatal(err)
		}
	}()

	// Validate if ep.Info() only gives valid gateway and sandbox key after has container has joined.
	info = ep1.Info()
	if len(info.Gateway()) == 0 {
		t.Fatalf("Expected a valid gateway for a joined endpoint. Instead found an invalid gateway: %v", info.Gateway())
	}
	if len(info.GatewayIPv6()) == 0 {
		t.Fatalf("Expected a valid ipv6 gateway for a joined endpoint. Instead found an invalid gateway: %v", info.GatewayIPv6())
	}

	if info.Sandbox() == nil {
		t.Fatalf("Expected an non-empty sandbox key for a joined endpoint. Instead found an empty sandbox key")
	}

	// Check endpoint provided container information
	if ep1.Info().Sandbox().Key() != sb.Key() {
		t.Fatalf("Endpoint Info returned unexpected sandbox key: %s", sb.Key())
	}

	// Attempt retrieval of endpoint interfaces statistics
	stats, err := sb.Statistics()
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := stats["eth0"]; !ok {
		t.Fatalf("Did not find eth0 statistics")
	}

	// Now test the container joining another network
	n2, err := createTestNetwork(bridgeNetType, "testnetwork2",
		options.Generic{
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

	ep2, err := n2.CreateEndpoint("ep2")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := ep2.Delete(false); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep2.Join(sb)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep2.Leave(sb)
		if err != nil {
			t.Fatal(err)
		}
	}()

	if ep1.Info().Sandbox().Key() != ep2.Info().Sandbox().Key() {
		t.Fatalf("ep1 and ep2 returned different container sandbox key")
	}

	checkSandbox(t, info)
}

func TestExternalKey(t *testing.T) {
	externalKeyTest(t, false)
}

func externalKeyTest(t *testing.T, reexec bool) {
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

	ep2, err := n2.CreateEndpoint("ep2")
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep2.Delete(false)
		if err != nil {
			t.Fatal(err)
		}
	}()

	cnt, err := controller.NewSandbox(containerID,
		libnetwork.OptionHostname("test"),
		libnetwork.OptionDomainname("docker.io"),
		libnetwork.OptionUseExternalKey(),
		libnetwork.OptionExtraHost("web", "192.168.0.1"))
	defer func() {
		if err := cnt.Delete(); err != nil {
			t.Fatal(err)
		}
		osl.GC()
	}()

	// Join endpoint to sandbox before SetKey
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

	sbox := ep.Info().Sandbox()
	if sbox == nil {
		t.Fatalf("Expected to have a valid Sandbox")
	}

	if reexec {
		err := reexecSetKey("this-must-fail", containerID, controller.ID())
		if err == nil {
			t.Fatalf("SetExternalKey must fail if the corresponding namespace is not created")
		}
	} else {
		// Setting an non-existing key (namespace) must fail
		if err := sbox.SetKey("this-must-fail"); err == nil {
			t.Fatalf("Setkey must fail if the corresponding namespace is not created")
		}
	}

	// Create a new OS sandbox using the osl API before using it in SetKey
	if extOsBox, err := osl.NewSandbox("ValidKey", true, false); err != nil {
		t.Fatalf("Failed to create new osl sandbox")
	} else {
		defer func() {
			if err := extOsBox.Destroy(); err != nil {
				logrus.Warnf("Failed to remove os sandbox: %v", err)
			}
		}()
	}

	if reexec {
		err := reexecSetKey("ValidKey", containerID, controller.ID())
		if err != nil {
			t.Fatalf("SetExternalKey failed with %v", err)
		}
	} else {
		if err := sbox.SetKey("ValidKey"); err != nil {
			t.Fatalf("Setkey failed with %v", err)
		}
	}

	// Join endpoint to sandbox after SetKey
	err = ep2.Join(sbox)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep2.Leave(sbox)
		if err != nil {
			t.Fatal(err)
		}
	}()

	if ep.Info().Sandbox().Key() != ep2.Info().Sandbox().Key() {
		t.Fatalf("ep1 and ep2 returned different container sandbox key")
	}

	checkSandbox(t, ep.Info())
}

func reexecSetKey(key string, containerID string, controllerID string) error {
	var (
		state libcontainer.State
		b     []byte
		err   error
	)

	state.NamespacePaths = make(map[configs.NamespaceType]string)
	state.NamespacePaths[configs.NamespaceType("NEWNET")] = key
	if b, err = json.Marshal(state); err != nil {
		return err
	}
	cmd := &exec.Cmd{
		Path:   reexec.Self(),
		Args:   append([]string{"libnetwork-setkey"}, containerID, controllerID),
		Stdin:  strings.NewReader(string(b)),
		Stdout: os.Stdout,
		Stderr: os.Stderr,
	}
	return cmd.Run()
}

func TestEnableIPv6(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	tmpResolvConf := []byte("search pommesfrites.fr\nnameserver 12.34.56.78\nnameserver 2001:4860:4860::8888\n")
	expectedResolvConf := []byte("search pommesfrites.fr\nnameserver 127.0.0.11\nnameserver 2001:4860:4860::8888\noptions ndots:0\n")
	//take a copy of resolv.conf for restoring after test completes
	resolvConfSystem, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		t.Fatal(err)
	}
	//cleanup
	defer func() {
		if err := ioutil.WriteFile("/etc/resolv.conf", resolvConfSystem, 0644); err != nil {
			t.Fatal(err)
		}
	}()

	netOption := options.Generic{
		netlabel.EnableIPv6: true,
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}
	ipamV6ConfList := []*libnetwork.IpamConf{{PreferredPool: "fe99::/64", Gateway: "fe99::9"}}

	n, err := createTestNetwork("bridge", "testnetwork", netOption, nil, ipamV6ConfList)
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

	if err := ioutil.WriteFile("/etc/resolv.conf", tmpResolvConf, 0644); err != nil {
		t.Fatal(err)
	}

	resolvConfPath := "/tmp/libnetwork_test/resolv.conf"
	defer os.Remove(resolvConfPath)

	sb, err := controller.NewSandbox(containerID, libnetwork.OptionResolvConfPath(resolvConfPath))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep1.Join(sb)
	if err != nil {
		t.Fatal(err)
	}

	content, err := ioutil.ReadFile(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(content, expectedResolvConf) {
		t.Fatalf("Expected:\n%s\nGot:\n%s", string(expectedResolvConf), string(content))
	}

	if err != nil {
		t.Fatal(err)
	}
}

func TestResolvConfHost(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	tmpResolvConf := []byte("search localhost.net\nnameserver 127.0.0.1\nnameserver 2001:4860:4860::8888\n")

	//take a copy of resolv.conf for restoring after test completes
	resolvConfSystem, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		t.Fatal(err)
	}
	//cleanup
	defer func() {
		if err := ioutil.WriteFile("/etc/resolv.conf", resolvConfSystem, 0644); err != nil {
			t.Fatal(err)
		}
	}()

	n, err := controller.NetworkByName("testhost")
	if err != nil {
		t.Fatal(err)
	}

	ep1, err := n.CreateEndpoint("ep1", libnetwork.CreateOptionDisableResolution())
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile("/etc/resolv.conf", tmpResolvConf, 0644); err != nil {
		t.Fatal(err)
	}

	resolvConfPath := "/tmp/libnetwork_test/resolv.conf"
	defer os.Remove(resolvConfPath)

	sb, err := controller.NewSandbox(containerID,
		libnetwork.OptionResolvConfPath(resolvConfPath),
		libnetwork.OptionOriginResolvConfPath("/etc/resolv.conf"))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep1.Join(sb)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		err = ep1.Leave(sb)
		if err != nil {
			t.Fatal(err)
		}
	}()

	finfo, err := os.Stat(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	fmode := (os.FileMode)(0644)
	if finfo.Mode() != fmode {
		t.Fatalf("Expected file mode %s, got %s", fmode.String(), finfo.Mode().String())
	}

	content, err := ioutil.ReadFile(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(content, tmpResolvConf) {
		t.Fatalf("Expected:\n%s\nGot:\n%s", string(tmpResolvConf), string(content))
	}
}

func TestResolvConf(t *testing.T) {
	if !testutils.IsRunningInContainer() {
		defer testutils.SetupTestOSContext(t)()
	}

	tmpResolvConf1 := []byte("search pommesfrites.fr\nnameserver 12.34.56.78\nnameserver 2001:4860:4860::8888\n")
	tmpResolvConf2 := []byte("search pommesfrites.fr\nnameserver 112.34.56.78\nnameserver 2001:4860:4860::8888\n")
	expectedResolvConf1 := []byte("search pommesfrites.fr\nnameserver 127.0.0.11\noptions ndots:0\n")
	tmpResolvConf3 := []byte("search pommesfrites.fr\nnameserver 113.34.56.78\n")

	//take a copy of resolv.conf for restoring after test completes
	resolvConfSystem, err := ioutil.ReadFile("/etc/resolv.conf")
	if err != nil {
		t.Fatal(err)
	}
	//cleanup
	defer func() {
		if err := ioutil.WriteFile("/etc/resolv.conf", resolvConfSystem, 0644); err != nil {
			t.Fatal(err)
		}
	}()

	netOption := options.Generic{
		netlabel.GenericData: options.Generic{
			"BridgeName": "testnetwork",
		},
	}
	n, err := createTestNetwork("bridge", "testnetwork", netOption, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := n.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	ep, err := n.CreateEndpoint("ep")
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile("/etc/resolv.conf", tmpResolvConf1, 0644); err != nil {
		t.Fatal(err)
	}

	resolvConfPath := "/tmp/libnetwork_test/resolv.conf"
	defer os.Remove(resolvConfPath)

	sb1, err := controller.NewSandbox(containerID, libnetwork.OptionResolvConfPath(resolvConfPath))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb1.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(sb1)
	if err != nil {
		t.Fatal(err)
	}

	finfo, err := os.Stat(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	fmode := (os.FileMode)(0644)
	if finfo.Mode() != fmode {
		t.Fatalf("Expected file mode %s, got %s", fmode.String(), finfo.Mode().String())
	}

	content, err := ioutil.ReadFile(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(content, expectedResolvConf1) {
		fmt.Printf("\n%v\n%v\n", expectedResolvConf1, content)
		t.Fatalf("Expected:\n%s\nGot:\n%s", string(expectedResolvConf1), string(content))
	}

	err = ep.Leave(sb1)
	if err != nil {
		t.Fatal(err)
	}

	if err := ioutil.WriteFile("/etc/resolv.conf", tmpResolvConf2, 0644); err != nil {
		t.Fatal(err)
	}

	sb2, err := controller.NewSandbox(containerID+"_2", libnetwork.OptionResolvConfPath(resolvConfPath))
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := sb2.Delete(); err != nil {
			t.Fatal(err)
		}
	}()

	err = ep.Join(sb2)
	if err != nil {
		t.Fatal(err)
	}

	content, err = ioutil.ReadFile(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(content, expectedResolvConf1) {
		t.Fatalf("Expected:\n%s\nGot:\n%s", string(expectedResolvConf1), string(content))
	}

	if err := ioutil.WriteFile(resolvConfPath, tmpResolvConf3, 0644); err != nil {
		t.Fatal(err)
	}

	err = ep.Leave(sb2)
	if err != nil {
		t.Fatal(err)
	}

	err = ep.Join(sb2)
	if err != nil {
		t.Fatal(err)
	}

	content, err = ioutil.ReadFile(resolvConfPath)
	if err != nil {
		t.Fatal(err)
	}

	if !bytes.Equal(content, tmpResolvConf3) {
		t.Fatalf("Expected:\n%s\nGot:\n%s", string(tmpResolvConf3), string(content))
	}
}

func parallelJoin(t *testing.T, rc libnetwork.Sandbox, ep libnetwork.Endpoint, thrNumber int) {
	debugf("J%d.", thrNumber)
	var err error

	sb := sboxes[thrNumber-1]
	err = ep.Join(sb)

	runtime.LockOSThread()
	if err != nil {
		if _, ok := err.(types.ForbiddenError); !ok {
			t.Fatalf("thread %d: %v", thrNumber, err)
		}
		debugf("JE%d(%v).", thrNumber, err)
	}
	debugf("JD%d.", thrNumber)
}

func parallelLeave(t *testing.T, rc libnetwork.Sandbox, ep libnetwork.Endpoint, thrNumber int) {
	debugf("L%d.", thrNumber)
	var err error

	sb := sboxes[thrNumber-1]

	err = ep.Leave(sb)
	runtime.LockOSThread()
	if err != nil {
		if _, ok := err.(types.ForbiddenError); !ok {
			t.Fatalf("thread %d: %v", thrNumber, err)
		}
		debugf("LE%d(%v).", thrNumber, err)
	}
	debugf("LD%d.", thrNumber)
}

func runParallelTests(t *testing.T, thrNumber int) {
	var (
		ep  libnetwork.Endpoint
		sb  libnetwork.Sandbox
		err error
	)

	t.Parallel()

	pTest := flag.Lookup("test.parallel")
	if pTest == nil {
		t.Skip("Skipped because test.parallel flag not set;")
	}
	numParallel, err := strconv.Atoi(pTest.Value.String())
	if err != nil {
		t.Fatal(err)
	}
	if numParallel < numThreads {
		t.Skip("Skipped because t.parallel was less than ", numThreads)
	}

	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	if thrNumber == first {
		createGlobalInstance(t)
	}

	if thrNumber != first {
		<-start

		thrdone := make(chan struct{})
		done <- thrdone
		defer close(thrdone)

		if thrNumber == last {
			defer close(done)
		}

		err = netns.Set(testns)
		if err != nil {
			t.Fatal(err)
		}
	}
	defer netns.Set(origns)

	net1, err := controller.NetworkByName("testhost")
	if err != nil {
		t.Fatal(err)
	}
	if net1 == nil {
		t.Fatal("Could not find testhost")
	}

	net2, err := controller.NetworkByName("network2")
	if err != nil {
		t.Fatal(err)
	}
	if net2 == nil {
		t.Fatal("Could not find network2")
	}

	epName := fmt.Sprintf("pep%d", thrNumber)

	if thrNumber == first {
		ep, err = net1.EndpointByName(epName)
	} else {
		ep, err = net2.EndpointByName(epName)
	}

	if err != nil {
		t.Fatal(err)
	}
	if ep == nil {
		t.Fatal("Got nil ep with no error")
	}

	cid := fmt.Sprintf("%drace", thrNumber)
	controller.WalkSandboxes(libnetwork.SandboxContainerWalker(&sb, cid))
	if sb == nil {
		t.Fatalf("Got nil sandbox for container: %s", cid)
	}

	for i := 0; i < iterCnt; i++ {
		parallelJoin(t, sb, ep, thrNumber)
		parallelLeave(t, sb, ep, thrNumber)
	}

	debugf("\n")

	err = sb.Delete()
	if err != nil {
		t.Fatal(err)
	}
	if thrNumber == first {
		for thrdone := range done {
			<-thrdone
		}

		testns.Close()
		if err := net2.Delete(); err != nil {
			t.Fatal(err)
		}
	} else {
		err = ep.Delete(false)
		if err != nil {
			t.Fatal(err)
		}
	}
}

func TestParallel1(t *testing.T) {
	runParallelTests(t, 1)
}

func TestParallel2(t *testing.T) {
	runParallelTests(t, 2)
}

func TestParallel3(t *testing.T) {
	runParallelTests(t, 3)
}

func TestNullIpam(t *testing.T) {
	_, err := controller.NewNetwork(bridgeNetType, "testnetworkinternal", "", libnetwork.NetworkOptionIpam(ipamapi.NullIPAM, "", nil, nil, nil))
	if err == nil || err.Error() != "ipv4 pool is empty" {
		t.Fatal("bridge network should complain empty pool")
	}
}

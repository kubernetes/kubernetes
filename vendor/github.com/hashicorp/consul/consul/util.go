package consul

import (
	"encoding/binary"
	"fmt"
	"net"
	"os"
	"path/filepath"
	"runtime"
	"strconv"

	"github.com/hashicorp/serf/serf"
)

/*
 * Contains an entry for each private block:
 * 10.0.0.0/8
 * 100.64.0.0/10
 * 127.0.0.0/8
 * 169.254.0.0/16
 * 172.16.0.0/12
 * 192.168.0.0/16
 */
var privateBlocks []*net.IPNet

// serverparts is used to return the parts of a server role
type serverParts struct {
	Name       string
	Datacenter string
	Port       int
	Bootstrap  bool
	Expect     int
	Version    int
	Addr       net.Addr
}

func (s *serverParts) String() string {
	return fmt.Sprintf("%s (Addr: %s) (DC: %s)", s.Name, s.Addr, s.Datacenter)
}

func init() {
	// Add each private block
	privateBlocks = make([]*net.IPNet, 6)

	_, block, err := net.ParseCIDR("10.0.0.0/8")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[0] = block

	_, block, err = net.ParseCIDR("100.64.0.0/10")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[1] = block

	_, block, err = net.ParseCIDR("127.0.0.0/8")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[2] = block

	_, block, err = net.ParseCIDR("169.254.0.0/16")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[3] = block

	_, block, err = net.ParseCIDR("172.16.0.0/12")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[4] = block

	_, block, err = net.ParseCIDR("192.168.0.0/16")
	if err != nil {
		panic(fmt.Sprintf("Bad cidr. Got %v", err))
	}
	privateBlocks[5] = block
}

// ensurePath is used to make sure a path exists
func ensurePath(path string, dir bool) error {
	if !dir {
		path = filepath.Dir(path)
	}
	return os.MkdirAll(path, 0755)
}

// CanServersUnderstandProtocol checks to see if all the servers in the given
// list understand the given protocol version. If there are no servers in the
// list then this will return false.
func CanServersUnderstandProtocol(members []serf.Member, version uint8) (bool, error) {
	numServers, numWhoGrok := 0, 0
	for _, m := range members {
		if m.Tags["role"] != "consul" {
			continue
		}
		numServers++

		vsn_min, err := strconv.Atoi(m.Tags["vsn_min"])
		if err != nil {
			return false, err
		}

		vsn_max, err := strconv.Atoi(m.Tags["vsn_max"])
		if err != nil {
			return false, err
		}

		v := int(version)
		if (v >= vsn_min) && (v <= vsn_max) {
			numWhoGrok++
		}
	}
	return (numServers > 0) && (numWhoGrok == numServers), nil
}

// Returns if a member is a consul server. Returns a bool,
// the datacenter, and the rpc port
func isConsulServer(m serf.Member) (bool, *serverParts) {
	if m.Tags["role"] != "consul" {
		return false, nil
	}

	datacenter := m.Tags["dc"]
	_, bootstrap := m.Tags["bootstrap"]

	expect := 0
	expect_str, ok := m.Tags["expect"]
	var err error
	if ok {
		expect, err = strconv.Atoi(expect_str)
		if err != nil {
			return false, nil
		}
	}

	port_str := m.Tags["port"]
	port, err := strconv.Atoi(port_str)
	if err != nil {
		return false, nil
	}

	vsn_str := m.Tags["vsn"]
	vsn, err := strconv.Atoi(vsn_str)
	if err != nil {
		return false, nil
	}

	addr := &net.TCPAddr{IP: m.Addr, Port: port}

	parts := &serverParts{
		Name:       m.Name,
		Datacenter: datacenter,
		Port:       port,
		Bootstrap:  bootstrap,
		Expect:     expect,
		Addr:       addr,
		Version:    vsn,
	}
	return true, parts
}

// Returns if a member is a consul node. Returns a bool,
// and the datacenter.
func isConsulNode(m serf.Member) (bool, string) {
	if m.Tags["role"] != "node" {
		return false, ""
	}
	return true, m.Tags["dc"]
}

// Returns if the given IP is in a private block
func isPrivateIP(ip_str string) bool {
	ip := net.ParseIP(ip_str)
	for _, priv := range privateBlocks {
		if priv.Contains(ip) {
			return true
		}
	}
	return false
}

// Returns addresses from interfaces that is up
func activeInterfaceAddresses() ([]net.Addr, error) {
	var upAddrs []net.Addr
	var loAddrs []net.Addr

	interfaces, err := net.Interfaces()
	if err != nil {
		return nil, fmt.Errorf("Failed to get interfaces: %v", err)
	}

	for _, iface := range interfaces {
		// Require interface to be up
		if iface.Flags&net.FlagUp == 0 {
			continue
		}

		addresses, err := iface.Addrs()
		if err != nil {
			return nil, fmt.Errorf("Failed to get interface addresses: %v", err)
		}

		if iface.Flags&net.FlagLoopback != 0 {
			loAddrs = append(loAddrs, addresses...)
			continue
		}

		upAddrs = append(upAddrs, addresses...)
	}

	if len(upAddrs) == 0 {
		return loAddrs, nil
	}

	return upAddrs, nil
}

// GetPrivateIP is used to return the first private IP address
// associated with an interface on the machine
func GetPrivateIP() (net.IP, error) {
	addresses, err := activeInterfaceAddresses()
	if err != nil {
		return nil, fmt.Errorf("Failed to get interface addresses: %v", err)
	}

	return getPrivateIP(addresses)
}

func getPrivateIP(addresses []net.Addr) (net.IP, error) {
	var candidates []net.IP

	// Find private IPv4 address
	for _, rawAddr := range addresses {
		var ip net.IP
		switch addr := rawAddr.(type) {
		case *net.IPAddr:
			ip = addr.IP
		case *net.IPNet:
			ip = addr.IP
		default:
			continue
		}

		if ip.To4() == nil {
			continue
		}
		if !isPrivateIP(ip.String()) {
			continue
		}
		candidates = append(candidates, ip)
	}
	numIps := len(candidates)
	switch numIps {
	case 0:
		return nil, fmt.Errorf("No private IP address found")
	case 1:
		return candidates[0], nil
	default:
		return nil, fmt.Errorf("Multiple private IPs found. Please configure one.")
	}

}

// Converts bytes to an integer
func bytesToUint64(b []byte) uint64 {
	return binary.BigEndian.Uint64(b)
}

// Converts a uint to a byte slice
func uint64ToBytes(u uint64) []byte {
	buf := make([]byte, 8)
	binary.BigEndian.PutUint64(buf, u)
	return buf
}

// runtimeStats is used to return various runtime information
func runtimeStats() map[string]string {
	return map[string]string{
		"os":         runtime.GOOS,
		"arch":       runtime.GOARCH,
		"version":    runtime.Version(),
		"max_procs":  strconv.FormatInt(int64(runtime.GOMAXPROCS(0)), 10),
		"goroutines": strconv.FormatInt(int64(runtime.NumGoroutine()), 10),
		"cpu_count":  strconv.FormatInt(int64(runtime.NumCPU()), 10),
	}
}

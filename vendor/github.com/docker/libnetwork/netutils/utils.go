// Network utility functions.

package netutils

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"

	"github.com/docker/libnetwork/types"
)

var (
	// ErrNetworkOverlapsWithNameservers preformatted error
	ErrNetworkOverlapsWithNameservers = errors.New("requested network overlaps with nameserver")
	// ErrNetworkOverlaps preformatted error
	ErrNetworkOverlaps = errors.New("requested network overlaps with existing network")
	// ErrNoDefaultRoute preformatted error
	ErrNoDefaultRoute = errors.New("no default route")
)

// CheckNameserverOverlaps checks whether the passed network overlaps with any of the nameservers
func CheckNameserverOverlaps(nameservers []string, toCheck *net.IPNet) error {
	if len(nameservers) > 0 {
		for _, ns := range nameservers {
			_, nsNetwork, err := net.ParseCIDR(ns)
			if err != nil {
				return err
			}
			if NetworkOverlaps(toCheck, nsNetwork) {
				return ErrNetworkOverlapsWithNameservers
			}
		}
	}
	return nil
}

// NetworkOverlaps detects overlap between one IPNet and another
func NetworkOverlaps(netX *net.IPNet, netY *net.IPNet) bool {
	return netX.Contains(netY.IP) || netY.Contains(netX.IP)
}

// NetworkRange calculates the first and last IP addresses in an IPNet
func NetworkRange(network *net.IPNet) (net.IP, net.IP) {
	if network == nil {
		return nil, nil
	}

	firstIP := network.IP.Mask(network.Mask)
	lastIP := types.GetIPCopy(firstIP)
	for i := 0; i < len(firstIP); i++ {
		lastIP[i] = firstIP[i] | ^network.Mask[i]
	}

	if network.IP.To4() != nil {
		firstIP = firstIP.To4()
		lastIP = lastIP.To4()
	}

	return firstIP, lastIP
}

// GetIfaceAddr returns the first IPv4 address and slice of IPv6 addresses for the specified network interface
func GetIfaceAddr(name string) (net.Addr, []net.Addr, error) {
	iface, err := net.InterfaceByName(name)
	if err != nil {
		return nil, nil, err
	}
	addrs, err := iface.Addrs()
	if err != nil {
		return nil, nil, err
	}
	var addrs4 []net.Addr
	var addrs6 []net.Addr
	for _, addr := range addrs {
		ip := (addr.(*net.IPNet)).IP
		if ip4 := ip.To4(); ip4 != nil {
			addrs4 = append(addrs4, addr)
		} else if ip6 := ip.To16(); len(ip6) == net.IPv6len {
			addrs6 = append(addrs6, addr)
		}
	}
	switch {
	case len(addrs4) == 0:
		return nil, nil, fmt.Errorf("Interface %v has no IPv4 addresses", name)
	case len(addrs4) > 1:
		fmt.Printf("Interface %v has more than 1 IPv4 address. Defaulting to using %v\n",
			name, (addrs4[0].(*net.IPNet)).IP)
	}
	return addrs4[0], addrs6, nil
}

func genMAC(ip net.IP) net.HardwareAddr {
	hw := make(net.HardwareAddr, 6)
	// The first byte of the MAC address has to comply with these rules:
	// 1. Unicast: Set the least-significant bit to 0.
	// 2. Address is locally administered: Set the second-least-significant bit (U/L) to 1.
	hw[0] = 0x02
	// The first 24 bits of the MAC represent the Organizationally Unique Identifier (OUI).
	// Since this address is locally administered, we can do whatever we want as long as
	// it doesn't conflict with other addresses.
	hw[1] = 0x42
	// Fill the remaining 4 bytes based on the input
	if ip == nil {
		rand.Read(hw[2:])
	} else {
		copy(hw[2:], ip.To4())
	}
	return hw
}

// GenerateRandomMAC returns a new 6-byte(48-bit) hardware address (MAC)
func GenerateRandomMAC() net.HardwareAddr {
	return genMAC(nil)
}

// GenerateMACFromIP returns a locally administered MAC address where the 4 least
// significant bytes are derived from the IPv4 address.
func GenerateMACFromIP(ip net.IP) net.HardwareAddr {
	return genMAC(ip)
}

// GenerateRandomName returns a new name joined with a prefix.  This size
// specified is used to truncate the randomly generated value
func GenerateRandomName(prefix string, size int) (string, error) {
	id := make([]byte, 32)
	if _, err := io.ReadFull(rand.Reader, id); err != nil {
		return "", err
	}
	return prefix + hex.EncodeToString(id)[:size], nil
}

// ReverseIP accepts a V4 or V6 IP string in the canonical form and returns a reversed IP in
// the dotted decimal form . This is used to setup the IP to service name mapping in the optimal
// way for the DNS PTR queries.
func ReverseIP(IP string) string {
	var reverseIP []string

	if net.ParseIP(IP).To4() != nil {
		reverseIP = strings.Split(IP, ".")
		l := len(reverseIP)
		for i, j := 0, l-1; i < l/2; i, j = i+1, j-1 {
			reverseIP[i], reverseIP[j] = reverseIP[j], reverseIP[i]
		}
	} else {
		reverseIP = strings.Split(IP, ":")

		// Reversed IPv6 is represented in dotted decimal instead of the typical
		// colon hex notation
		for key := range reverseIP {
			if len(reverseIP[key]) == 0 { // expand the compressed 0s
				reverseIP[key] = strings.Repeat("0000", 8-strings.Count(IP, ":"))
			} else if len(reverseIP[key]) < 4 { // 0-padding needed
				reverseIP[key] = strings.Repeat("0", 4-len(reverseIP[key])) + reverseIP[key]
			}
		}

		reverseIP = strings.Split(strings.Join(reverseIP, ""), "")

		l := len(reverseIP)
		for i, j := 0, l-1; i < l/2; i, j = i+1, j-1 {
			reverseIP[i], reverseIP[j] = reverseIP[j], reverseIP[i]
		}
	}

	return strings.Join(reverseIP, ".")
}

// ParseAlias parses and validates the specified string as an alias format (name:alias)
func ParseAlias(val string) (string, string, error) {
	if val == "" {
		return "", "", errors.New("empty string specified for alias")
	}
	arr := strings.Split(val, ":")
	if len(arr) > 2 {
		return "", "", fmt.Errorf("bad format for alias: %s", val)
	}
	if len(arr) == 1 {
		return val, val, nil
	}
	return arr[0], arr[1], nil
}

// ValidateAlias validates that the specified string has a valid alias format (containerName:alias).
func ValidateAlias(val string) (string, error) {
	if _, _, err := ParseAlias(val); err != nil {
		return val, err
	}
	return val, nil
}

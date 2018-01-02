package ipam

import (
	"fmt"
	"net"

	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/types"
)

type ipVersion int

const (
	v4 = 4
	v6 = 6
)

func getAddressRange(pool string, masterNw *net.IPNet) (*AddressRange, error) {
	ip, nw, err := net.ParseCIDR(pool)
	if err != nil {
		return nil, ipamapi.ErrInvalidSubPool
	}
	lIP, e := types.GetHostPartIP(nw.IP, masterNw.Mask)
	if e != nil {
		return nil, fmt.Errorf("failed to compute range's lowest ip address: %v", e)
	}
	bIP, e := types.GetBroadcastIP(nw.IP, nw.Mask)
	if e != nil {
		return nil, fmt.Errorf("failed to compute range's broadcast ip address: %v", e)
	}
	hIP, e := types.GetHostPartIP(bIP, masterNw.Mask)
	if e != nil {
		return nil, fmt.Errorf("failed to compute range's highest ip address: %v", e)
	}
	nw.IP = ip
	return &AddressRange{nw, ipToUint64(types.GetMinimalIP(lIP)), ipToUint64(types.GetMinimalIP(hIP))}, nil
}

// It generates the ip address in the passed subnet specified by
// the passed host address ordinal
func generateAddress(ordinal uint64, network *net.IPNet) net.IP {
	var address [16]byte

	// Get network portion of IP
	if getAddressVersion(network.IP) == v4 {
		copy(address[:], network.IP.To4())
	} else {
		copy(address[:], network.IP)
	}

	end := len(network.Mask)
	addIntToIP(address[:end], ordinal)

	return net.IP(address[:end])
}

func getAddressVersion(ip net.IP) ipVersion {
	if ip.To4() == nil {
		return v6
	}
	return v4
}

// Adds the ordinal IP to the current array
// 192.168.0.0 + 53 => 192.168.0.53
func addIntToIP(array []byte, ordinal uint64) {
	for i := len(array) - 1; i >= 0; i-- {
		array[i] |= (byte)(ordinal & 0xff)
		ordinal >>= 8
	}
}

// Convert an ordinal to the respective IP address
func ipToUint64(ip []byte) (value uint64) {
	cip := types.GetMinimalIP(ip)
	for i := 0; i < len(cip); i++ {
		j := len(cip) - 1 - i
		value += uint64(cip[i]) << uint(j*8)
	}
	return value
}

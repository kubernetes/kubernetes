// Package ipamutils provides utility functions for ipam management
package ipamutils

import (
	"net"
	"sync"
)

var (
	// PredefinedBroadNetworks contains a list of 31 IPv4 private networks with host size 16 and 12
	// (172.17-31.x.x/16, 192.168.x.x/20) which do not overlap with the networks in `PredefinedGranularNetworks`
	PredefinedBroadNetworks []*net.IPNet
	// PredefinedGranularNetworks contains a list of 64K IPv4 private networks with host size 8
	// (10.x.x.x/24) which do not overlap with the networks in `PredefinedBroadNetworks`
	PredefinedGranularNetworks []*net.IPNet

	initNetworksOnce sync.Once
)

// InitNetworks initializes the pre-defined networks used by the built-in IP allocator
func InitNetworks() {
	initNetworksOnce.Do(func() {
		PredefinedBroadNetworks = initBroadPredefinedNetworks()
		PredefinedGranularNetworks = initGranularPredefinedNetworks()
	})
}

func initBroadPredefinedNetworks() []*net.IPNet {
	pl := make([]*net.IPNet, 0, 31)
	mask := []byte{255, 255, 0, 0}
	for i := 17; i < 32; i++ {
		pl = append(pl, &net.IPNet{IP: []byte{172, byte(i), 0, 0}, Mask: mask})
	}
	mask20 := []byte{255, 255, 240, 0}
	for i := 0; i < 16; i++ {
		pl = append(pl, &net.IPNet{IP: []byte{192, 168, byte(i << 4), 0}, Mask: mask20})
	}
	return pl
}

func initGranularPredefinedNetworks() []*net.IPNet {
	pl := make([]*net.IPNet, 0, 256*256)
	mask := []byte{255, 255, 255, 0}
	for i := 0; i < 256; i++ {
		for j := 0; j < 256; j++ {
			pl = append(pl, &net.IPNet{IP: []byte{10, byte(i), byte(j), 0}, Mask: mask})
		}
	}
	return pl
}

// Package null implements the null ipam driver. Null ipam driver satisfies ipamapi contract,
// but does not effectively reserve/allocate any address pool or address
package null

import (
	"fmt"
	"net"

	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/types"
)

var (
	defaultAS      = "null"
	defaultPool, _ = types.ParseCIDR("0.0.0.0/0")
	defaultPoolID  = fmt.Sprintf("%s/%s", defaultAS, defaultPool.String())
)

type allocator struct{}

func (a *allocator) GetDefaultAddressSpaces() (string, string, error) {
	return defaultAS, defaultAS, nil
}

func (a *allocator) RequestPool(addressSpace, pool, subPool string, options map[string]string, v6 bool) (string, *net.IPNet, map[string]string, error) {
	if addressSpace != defaultAS {
		return "", nil, nil, types.BadRequestErrorf("unknown address space: %s", addressSpace)
	}
	if pool != "" {
		return "", nil, nil, types.BadRequestErrorf("null ipam driver does not handle specific address pool requests")
	}
	if subPool != "" {
		return "", nil, nil, types.BadRequestErrorf("null ipam driver does not handle specific address subpool requests")
	}
	if v6 {
		return "", nil, nil, types.BadRequestErrorf("null ipam driver does not handle IPv6 address pool pool requests")
	}
	return defaultPoolID, defaultPool, nil, nil
}

func (a *allocator) ReleasePool(poolID string) error {
	return nil
}

func (a *allocator) RequestAddress(poolID string, ip net.IP, opts map[string]string) (*net.IPNet, map[string]string, error) {
	if poolID != defaultPoolID {
		return nil, nil, types.BadRequestErrorf("unknown pool id: %s", poolID)
	}
	return nil, nil, nil
}

func (a *allocator) ReleaseAddress(poolID string, ip net.IP) error {
	if poolID != defaultPoolID {
		return types.BadRequestErrorf("unknown pool id: %s", poolID)
	}
	return nil
}

func (a *allocator) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

func (a *allocator) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

func (a *allocator) IsBuiltIn() bool {
	return true
}

// Init registers a remote ipam when its plugin is activated
func Init(ic ipamapi.Callback, l, g interface{}) error {
	return ic.RegisterIpamDriver(ipamapi.NullIPAM, &allocator{})
}

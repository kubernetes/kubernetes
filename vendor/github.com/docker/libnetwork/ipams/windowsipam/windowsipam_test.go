package windowsipam

import (
	"net"
	"testing"

	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/netlabel"
	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func TestWindowsIPAM(t *testing.T) {
	a := &allocator{}
	requestPool, _ := types.ParseCIDR("192.168.0.0/16")
	requestAddress := net.ParseIP("192.168.1.1")

	pid, pool, _, err := a.RequestPool(localAddressSpace, "", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(defaultPool, pool) ||
		pid != pool.String() {
		t.Fatalf("Unexpected data returned. Expected %v : %s. Got: %v : %s", defaultPool, pid, pool, pool.String())
	}

	pid, pool, _, err = a.RequestPool(localAddressSpace, requestPool.String(), "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(requestPool, pool) ||
		pid != requestPool.String() {
		t.Fatalf("Unexpected data returned. Expected %v : %s. Got: %v : %s", requestPool, requestPool.String(), pool, pool.String())
	}

	_, _, _, err = a.RequestPool(localAddressSpace, requestPool.String(), requestPool.String(), nil, false)
	if err == nil {
		t.Fatal("Unexpected success for subpool request")
	}

	_, _, _, err = a.RequestPool(localAddressSpace, requestPool.String(), "", nil, true)
	if err == nil {
		t.Fatal("Unexpected success for v6 request")
	}

	err = a.ReleasePool(requestPool.String())
	if err != nil {
		t.Fatal(err)
	}

	ip, _, err := a.RequestAddress(requestPool.String(), nil, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}

	if ip != nil {
		t.Fatalf("Unexpected data returned. Expected %v . Got: %v ", requestPool, ip)
	}

	ip, _, err = a.RequestAddress(requestPool.String(), requestAddress, map[string]string{})
	if err != nil {
		t.Fatal(err)
	}

	if !ip.IP.Equal(requestAddress) {
		t.Fatalf("Unexpected data returned. Expected %v . Got: %v ", requestAddress, ip.IP)
	}

	requestOptions := map[string]string{}
	requestOptions[ipamapi.RequestAddressType] = netlabel.Gateway
	ip, _, err = a.RequestAddress(requestPool.String(), requestAddress, requestOptions)
	if err != nil {
		t.Fatal(err)
	}

	if !ip.IP.Equal(requestAddress) {
		t.Fatalf("Unexpected data returned. Expected %v . Got: %v ", requestAddress, ip.IP)
	}

	err = a.ReleaseAddress(requestPool.String(), requestAddress)
	if err != nil {
		t.Fatal(err)
	}
}

package driverapi

import (
	"encoding/json"
	"net"
	"testing"

	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

func TestIPDataMarshalling(t *testing.T) {
	i := &IPAMData{
		AddressSpace: "giallo",
		Pool:         &net.IPNet{IP: net.IP{10, 10, 10, 8}, Mask: net.IPMask{255, 255, 255, 0}},
		Gateway:      &net.IPNet{IP: net.IP{10, 10, 10, 254}, Mask: net.IPMask{255, 255, 255, 0}},
		AuxAddresses: map[string]*net.IPNet{
			"ip1": {IP: net.IP{10, 10, 10, 1}, Mask: net.IPMask{255, 255, 255, 0}},
			"ip2": {IP: net.IP{10, 10, 10, 2}, Mask: net.IPMask{255, 255, 255, 0}},
		},
	}

	b, err := json.Marshal(i)
	if err != nil {
		t.Fatal(err)
	}

	ii := &IPAMData{}
	err = json.Unmarshal(b, &ii)
	if err != nil {
		t.Fatal(err)
	}

	if i.AddressSpace != ii.AddressSpace || !types.CompareIPNet(i.Pool, ii.Pool) ||
		!types.CompareIPNet(i.Gateway, ii.Gateway) ||
		!compareAddresses(i.AuxAddresses, ii.AuxAddresses) {
		t.Fatalf("JSON marsh/unmarsh failed.\nOriginal:\n%s\nDecoded:\n%s", i, ii)
	}
}

func compareAddresses(a, b map[string]*net.IPNet) bool {
	if len(a) != len(b) {
		return false
	}
	if len(a) > 0 {
		for k := range a {
			if !types.CompareIPNet(a[k], b[k]) {
				return false
			}
		}
	}
	return true
}

func TestValidateAndIsV6(t *testing.T) {
	var err error

	i := &IPAMData{
		Pool:    &net.IPNet{IP: net.IP{10, 10, 10, 8}, Mask: net.IPMask{255, 255, 255, 0}},
		Gateway: &net.IPNet{IP: net.IP{10, 10, 10, 254}, Mask: net.IPMask{255, 255, 255, 0}},
		AuxAddresses: map[string]*net.IPNet{
			"ip1": {IP: net.IP{10, 10, 10, 1}, Mask: net.IPMask{255, 255, 255, 0}},
			"ip2": {IP: net.IP{10, 10, 10, 2}, Mask: net.IPMask{255, 255, 255, 0}},
		},
	}

	// Check ip version
	if i.IsV6() {
		t.Fatal("incorrect ip version returned")
	}
	orig := i.Pool
	if i.Pool, err = types.ParseCIDR("2001:db8::33/64"); err != nil {
		t.Fatal(err)
	}
	if !i.IsV6() {
		t.Fatal("incorrect ip version returned")
	}
	i.Pool = orig

	// valid ip data
	if err = i.Validate(); err != nil {
		t.Fatal(err)
	}

	// incongruent gw ver
	if i.Gateway, err = types.ParseCIDR("2001:db8::45/65"); err != nil {
		t.Fatal(err)
	}
	if err = i.Validate(); err == nil {
		t.Fatal("expected error but succeeded")
	}
	i.Gateway = nil

	// incongruent secondary ip ver
	if i.AuxAddresses["ip2"], err = types.ParseCIDR("2001:db8::44/80"); err != nil {
		t.Fatal(err)
	}
	if err = i.Validate(); err == nil {
		t.Fatal("expected error but succeeded")
	}
	delete(i.AuxAddresses, "ip2")

	// gw outside pool
	if i.Gateway, err = types.ParseCIDR("10.10.15.254/24"); err != nil {
		t.Fatal(err)
	}
	if err = i.Validate(); err == nil {
		t.Fatal("expected error but succeeded")
	}
	i.Gateway = nil

	// sec ip outside of pool
	if i.AuxAddresses["ip1"], err = types.ParseCIDR("10.10.2.1/24"); err != nil {
		t.Fatal(err)
	}
	if err = i.Validate(); err == nil {
		t.Fatal("expected error but succeeded")
	}
}

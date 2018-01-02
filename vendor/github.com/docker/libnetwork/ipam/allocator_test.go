package ipam

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/docker/libkv/store"
	"github.com/docker/libkv/store/boltdb"
	"github.com/docker/libnetwork/bitseq"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/ipamapi"
	"github.com/docker/libnetwork/ipamutils"
	_ "github.com/docker/libnetwork/testutils"
	"github.com/docker/libnetwork/types"
)

const (
	defaultPrefix = "/tmp/libnetwork/test/ipam"
)

func init() {
	boltdb.Register()
}

// OptionBoltdbWithRandomDBFile function returns a random dir for local store backend
func randomLocalStore() (datastore.DataStore, error) {
	tmp, err := ioutil.TempFile("", "libnetwork-")
	if err != nil {
		return nil, fmt.Errorf("Error creating temp file: %v", err)
	}
	if err := tmp.Close(); err != nil {
		return nil, fmt.Errorf("Error closing temp file: %v", err)
	}
	return datastore.NewDataStore(datastore.LocalScope, &datastore.ScopeCfg{
		Client: datastore.ScopeClientCfg{
			Provider: "boltdb",
			Address:  defaultPrefix + tmp.Name(),
			Config: &store.Config{
				Bucket:            "libnetwork",
				ConnectionTimeout: 3 * time.Second,
			},
		},
	})
}

func getAllocator() (*Allocator, error) {
	ipamutils.InitNetworks()
	ds, err := randomLocalStore()
	if err != nil {
		return nil, err
	}
	a, err := NewAllocator(ds, nil)
	if err != nil {
		return nil, err
	}
	return a, nil
}

func TestInt2IP2IntConversion(t *testing.T) {
	for i := uint64(0); i < 256*256*256; i++ {
		var array [4]byte // new array at each cycle
		addIntToIP(array[:], i)
		j := ipToUint64(array[:])
		if j != i {
			t.Fatalf("Failed to convert ordinal %d to IP % x and back to ordinal. Got %d", i, array, j)
		}
	}
}

func TestGetAddressVersion(t *testing.T) {
	if v4 != getAddressVersion(net.ParseIP("172.28.30.112")) {
		t.Fatal("Failed to detect IPv4 version")
	}
	if v4 != getAddressVersion(net.ParseIP("0.0.0.1")) {
		t.Fatal("Failed to detect IPv4 version")
	}
	if v6 != getAddressVersion(net.ParseIP("ff01::1")) {
		t.Fatal("Failed to detect IPv6 version")
	}
	if v6 != getAddressVersion(net.ParseIP("2001:db8::76:51")) {
		t.Fatal("Failed to detect IPv6 version")
	}
}

func TestKeyString(t *testing.T) {
	k := &SubnetKey{AddressSpace: "default", Subnet: "172.27.0.0/16"}
	expected := "default/172.27.0.0/16"
	if expected != k.String() {
		t.Fatalf("Unexpected key string: %s", k.String())
	}

	k2 := &SubnetKey{}
	err := k2.FromString(expected)
	if err != nil {
		t.Fatal(err)
	}
	if k2.AddressSpace != k.AddressSpace || k2.Subnet != k.Subnet {
		t.Fatalf("SubnetKey.FromString() failed. Expected %v. Got %v", k, k2)
	}

	expected = fmt.Sprintf("%s/%s", expected, "172.27.3.0/24")
	k.ChildSubnet = "172.27.3.0/24"
	if expected != k.String() {
		t.Fatalf("Unexpected key string: %s", k.String())
	}

	err = k2.FromString(expected)
	if err != nil {
		t.Fatal(err)
	}
	if k2.AddressSpace != k.AddressSpace || k2.Subnet != k.Subnet || k2.ChildSubnet != k.ChildSubnet {
		t.Fatalf("SubnetKey.FromString() failed. Expected %v. Got %v", k, k2)
	}
}

func TestPoolDataMarshal(t *testing.T) {
	_, nw, err := net.ParseCIDR("172.28.30.1/24")
	if err != nil {
		t.Fatal(err)
	}

	p := &PoolData{
		ParentKey: SubnetKey{AddressSpace: "Blue", Subnet: "172.28.0.0/16"},
		Pool:      nw,
		Range:     &AddressRange{Sub: &net.IPNet{IP: net.IP{172, 28, 20, 0}, Mask: net.IPMask{255, 255, 255, 0}}, Start: 0, End: 255},
		RefCount:  4,
	}

	ba, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	var q PoolData
	err = json.Unmarshal(ba, &q)
	if err != nil {
		t.Fatal(err)
	}

	if p.ParentKey != q.ParentKey || !types.CompareIPNet(p.Range.Sub, q.Range.Sub) ||
		p.Range.Start != q.Range.Start || p.Range.End != q.Range.End || p.RefCount != q.RefCount ||
		!types.CompareIPNet(p.Pool, q.Pool) {
		t.Fatalf("\n%#v\n%#v", p, &q)
	}

	p = &PoolData{
		ParentKey: SubnetKey{AddressSpace: "Blue", Subnet: "172.28.0.0/16"},
		Pool:      nw,
		RefCount:  4,
	}

	ba, err = json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	err = json.Unmarshal(ba, &q)
	if err != nil {
		t.Fatal(err)
	}

	if q.Range != nil {
		t.Fatal("Unexpected Range")
	}
}

func TestSubnetsMarshal(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	pid0, _, _, err := a.RequestPool(localAddressSpace, "192.168.0.0/16", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	pid1, _, _, err := a.RequestPool(localAddressSpace, "192.169.0.0/16", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	_, _, err = a.RequestAddress(pid0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	ba := cfg.Value()
	if err := cfg.SetValue(ba); err != nil {
		t.Fatal(err)
	}

	expIP := &net.IPNet{IP: net.IP{192, 168, 0, 2}, Mask: net.IPMask{255, 255, 0, 0}}
	ip, _, err := a.RequestAddress(pid0, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(expIP, ip) {
		t.Fatalf("Got unexpected ip after pool config restore: %s", ip)
	}

	expIP = &net.IPNet{IP: net.IP{192, 169, 0, 1}, Mask: net.IPMask{255, 255, 0, 0}}
	ip, _, err = a.RequestAddress(pid1, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(expIP, ip) {
		t.Fatalf("Got unexpected ip after pool config restore: %s", ip)
	}
}

func TestAddSubnets(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	a.addrSpaces["abc"] = a.addrSpaces[localAddressSpace]

	pid0, _, _, err := a.RequestPool(localAddressSpace, "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding subnet")
	}

	pid1, _, _, err := a.RequestPool("abc", "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatalf("Unexpected failure in adding overlapping subnets to different address spaces: %v", err)
	}

	if pid0 == pid1 {
		t.Fatal("returned same pool id for same subnets in different namespaces")
	}

	pid, _, _, err := a.RequestPool("abc", "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatalf("Unexpected failure requesting existing subnet: %v", err)
	}
	if pid != pid1 {
		t.Fatal("returned different pool id for same subnet requests")
	}

	_, _, _, err = a.RequestPool("abc", "10.128.0.0/9", "", nil, false)
	if err == nil {
		t.Fatal("Expected failure on adding overlapping base subnet")
	}

	pid2, _, _, err := a.RequestPool("abc", "10.0.0.0/8", "10.128.0.0/9", nil, false)
	if err != nil {
		t.Fatalf("Unexpected failure on adding sub pool: %v", err)
	}
	pid3, _, _, err := a.RequestPool("abc", "10.0.0.0/8", "10.128.0.0/9", nil, false)
	if err != nil {
		t.Fatalf("Unexpected failure on adding overlapping sub pool: %v", err)
	}
	if pid2 != pid3 {
		t.Fatal("returned different pool id for same sub pool requests")
	}

	_, _, _, err = a.RequestPool(localAddressSpace, "10.20.2.0/24", "", nil, false)
	if err == nil {
		t.Fatal("Failed to detect overlapping subnets")
	}

	_, _, _, err = a.RequestPool(localAddressSpace, "10.128.0.0/9", "", nil, false)
	if err == nil {
		t.Fatal("Failed to detect overlapping subnets")
	}

	_, _, _, err = a.RequestPool(localAddressSpace, "1003:1:2:3:4:5:6::/112", "", nil, false)
	if err != nil {
		t.Fatalf("Failed to add v6 subnet: %s", err.Error())
	}

	_, _, _, err = a.RequestPool(localAddressSpace, "1003:1:2:3::/64", "", nil, false)
	if err == nil {
		t.Fatal("Failed to detect overlapping v6 subnet")
	}
}

func TestAddReleasePoolID(t *testing.T) {
	var k0, k1, k2 SubnetKey

	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}

	aSpace, err := a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	pid0, _, _, err := a.RequestPool(localAddressSpace, "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding pool")
	}
	if err := k0.FromString(pid0); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets := aSpace.subnets

	if subnets[k0].RefCount != 1 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}

	pid1, _, _, err := a.RequestPool(localAddressSpace, "10.0.0.0/8", "10.0.0.0/16", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding sub pool")
	}
	if err := k1.FromString(pid1); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k1].RefCount != 1 {
		t.Fatalf("Unexpected ref count for %s: %d", k1, subnets[k1].RefCount)
	}

	pid2, _, _, err := a.RequestPool(localAddressSpace, "10.0.0.0/8", "10.0.0.0/16", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding sub pool")
	}
	if pid0 == pid1 || pid0 == pid2 || pid1 != pid2 {
		t.Fatalf("Incorrect poolIDs returned %s, %s, %s", pid0, pid1, pid2)
	}
	if err := k2.FromString(pid2); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k2].RefCount != 2 {
		t.Fatalf("Unexpected ref count for %s: %d", k2, subnets[k2].RefCount)
	}

	if subnets[k0].RefCount != 3 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}

	if err := a.ReleasePool(pid1); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k0].RefCount != 2 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}
	if err := a.ReleasePool(pid0); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k0].RefCount != 1 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}

	pid00, _, _, err := a.RequestPool(localAddressSpace, "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding pool")
	}
	if pid00 != pid0 {
		t.Fatal("main pool should still exist")
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k0].RefCount != 2 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}

	if err := a.ReleasePool(pid2); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k0].RefCount != 1 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}

	if err := a.ReleasePool(pid00); err != nil {
		t.Fatal(err)
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if bp, ok := subnets[k0]; ok {
		t.Fatalf("Base pool %s is still present: %v", k0, bp)
	}

	_, _, _, err = a.RequestPool(localAddressSpace, "10.0.0.0/8", "", nil, false)
	if err != nil {
		t.Fatal("Unexpected failure in adding pool")
	}

	aSpace, err = a.getAddrSpace(localAddressSpace)
	if err != nil {
		t.Fatal(err)
	}

	subnets = aSpace.subnets
	if subnets[k0].RefCount != 1 {
		t.Fatalf("Unexpected ref count for %s: %d", k0, subnets[k0].RefCount)
	}
}

func TestPredefinedPool(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}

	if _, err := a.getPredefinedPool("blue", false); err == nil {
		t.Fatal("Expected failure for non default addr space")
	}

	pid, nw, _, err := a.RequestPool(localAddressSpace, "", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	nw2, err := a.getPredefinedPool(localAddressSpace, false)
	if err != nil {
		t.Fatal(err)
	}
	if types.CompareIPNet(nw, nw2) {
		t.Fatalf("Unexpected default network returned: %s = %s", nw2, nw)
	}

	if err := a.ReleasePool(pid); err != nil {
		t.Fatal(err)
	}
}

func TestRemoveSubnet(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	a.addrSpaces["splane"] = &addrSpace{
		id:      dsConfigKey + "/" + "splane",
		ds:      a.addrSpaces[localAddressSpace].ds,
		alloc:   a.addrSpaces[localAddressSpace].alloc,
		scope:   a.addrSpaces[localAddressSpace].scope,
		subnets: map[SubnetKey]*PoolData{},
	}

	input := []struct {
		addrSpace string
		subnet    string
		v6        bool
	}{
		{localAddressSpace, "192.168.0.0/16", false},
		{localAddressSpace, "172.17.0.0/16", false},
		{localAddressSpace, "10.0.0.0/8", false},
		{localAddressSpace, "2001:db8:1:2:3:4:ffff::/112", false},
		{"splane", "172.17.0.0/16", false},
		{"splane", "10.0.0.0/8", false},
		{"splane", "2001:db8:1:2:3:4:5::/112", true},
		{"splane", "2001:db8:1:2:3:4:ffff::/112", true},
	}

	poolIDs := make([]string, len(input))

	for ind, i := range input {
		if poolIDs[ind], _, _, err = a.RequestPool(i.addrSpace, i.subnet, "", nil, i.v6); err != nil {
			t.Fatalf("Failed to apply input. Can't proceed: %s", err.Error())
		}
	}

	for ind, id := range poolIDs {
		if err := a.ReleasePool(id); err != nil {
			t.Fatalf("Failed to release poolID %s (%d)", id, ind)
		}
	}
}

func TestGetSameAddress(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	a.addrSpaces["giallo"] = &addrSpace{
		id:      dsConfigKey + "/" + "giallo",
		ds:      a.addrSpaces[localAddressSpace].ds,
		alloc:   a.addrSpaces[localAddressSpace].alloc,
		scope:   a.addrSpaces[localAddressSpace].scope,
		subnets: map[SubnetKey]*PoolData{},
	}

	pid, _, _, err := a.RequestPool("giallo", "192.168.100.0/24", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	ip := net.ParseIP("192.168.100.250")
	_, _, err = a.RequestAddress(pid, ip, nil)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = a.RequestAddress(pid, ip, nil)
	if err == nil {
		t.Fatal(err)
	}
}

func TestGetAddressSubPoolEqualPool(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	// Requesting a subpool of same size of the master pool should not cause any problem on ip allocation
	pid, _, _, err := a.RequestPool(localAddressSpace, "172.18.0.0/16", "172.18.0.0/16", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	_, _, err = a.RequestAddress(pid, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
}

func TestRequestReleaseAddressFromSubPool(t *testing.T) {
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	a.addrSpaces["rosso"] = &addrSpace{
		id:      dsConfigKey + "/" + "rosso",
		ds:      a.addrSpaces[localAddressSpace].ds,
		alloc:   a.addrSpaces[localAddressSpace].alloc,
		scope:   a.addrSpaces[localAddressSpace].scope,
		subnets: map[SubnetKey]*PoolData{},
	}

	poolID, _, _, err := a.RequestPool("rosso", "172.28.0.0/16", "172.28.30.0/24", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	var ip *net.IPNet
	expected := &net.IPNet{IP: net.IP{172, 28, 30, 255}, Mask: net.IPMask{255, 255, 0, 0}}
	for err == nil {
		var c *net.IPNet
		if c, _, err = a.RequestAddress(poolID, nil, nil); err == nil {
			ip = c
		}
	}
	if err != ipamapi.ErrNoAvailableIPs {
		t.Fatal(err)
	}
	if !types.CompareIPNet(expected, ip) {
		t.Fatalf("Unexpected last IP from subpool. Expected: %s. Got: %v.", expected, ip)
	}
	rp := &net.IPNet{IP: net.IP{172, 28, 30, 97}, Mask: net.IPMask{255, 255, 0, 0}}
	if err = a.ReleaseAddress(poolID, rp.IP); err != nil {
		t.Fatal(err)
	}
	if ip, _, err = a.RequestAddress(poolID, nil, nil); err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(rp, ip) {
		t.Fatalf("Unexpected IP from subpool. Expected: %s. Got: %v.", rp, ip)
	}

	_, _, _, err = a.RequestPool("rosso", "10.0.0.0/8", "10.0.0.0/16", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	poolID, _, _, err = a.RequestPool("rosso", "10.0.0.0/16", "10.0.0.0/24", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	expected = &net.IPNet{IP: net.IP{10, 0, 0, 255}, Mask: net.IPMask{255, 255, 0, 0}}
	for err == nil {
		var c *net.IPNet
		if c, _, err = a.RequestAddress(poolID, nil, nil); err == nil {
			ip = c
		}
	}
	if err != ipamapi.ErrNoAvailableIPs {
		t.Fatal(err)
	}
	if !types.CompareIPNet(expected, ip) {
		t.Fatalf("Unexpected last IP from subpool. Expected: %s. Got: %v.", expected, ip)
	}
	rp = &net.IPNet{IP: net.IP{10, 0, 0, 79}, Mask: net.IPMask{255, 255, 0, 0}}
	if err = a.ReleaseAddress(poolID, rp.IP); err != nil {
		t.Fatal(err)
	}
	if ip, _, err = a.RequestAddress(poolID, nil, nil); err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(rp, ip) {
		t.Fatalf("Unexpected IP from subpool. Expected: %s. Got: %v.", rp, ip)
	}

	// Request any addresses from subpool after explicit address request
	unoExp, _ := types.ParseCIDR("10.2.2.0/16")
	dueExp, _ := types.ParseCIDR("10.2.2.2/16")
	treExp, _ := types.ParseCIDR("10.2.2.1/16")
	if poolID, _, _, err = a.RequestPool("rosso", "10.2.0.0/16", "10.2.2.0/24", nil, false); err != nil {
		t.Fatal(err)
	}
	tre, _, err := a.RequestAddress(poolID, treExp.IP, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(tre, treExp) {
		t.Fatalf("Unexpected address: %v", tre)
	}

	uno, _, err := a.RequestAddress(poolID, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(uno, unoExp) {
		t.Fatalf("Unexpected address: %v", uno)
	}

	due, _, err := a.RequestAddress(poolID, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(due, dueExp) {
		t.Fatalf("Unexpected address: %v", due)
	}

	if err = a.ReleaseAddress(poolID, uno.IP); err != nil {
		t.Fatal(err)
	}
	uno, _, err = a.RequestAddress(poolID, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(uno, unoExp) {
		t.Fatalf("Unexpected address: %v", uno)
	}

	if err = a.ReleaseAddress(poolID, tre.IP); err != nil {
		t.Fatal(err)
	}
	tre, _, err = a.RequestAddress(poolID, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !types.CompareIPNet(tre, treExp) {
		t.Fatalf("Unexpected address: %v", tre)
	}
}

func TestGetAddress(t *testing.T) {
	input := []string{
		/*"10.0.0.0/8", "10.0.0.0/9", "10.0.0.0/10",*/ "10.0.0.0/11", "10.0.0.0/12", "10.0.0.0/13", "10.0.0.0/14",
		"10.0.0.0/15", "10.0.0.0/16", "10.0.0.0/17", "10.0.0.0/18", "10.0.0.0/19", "10.0.0.0/20", "10.0.0.0/21",
		"10.0.0.0/22", "10.0.0.0/23", "10.0.0.0/24", "10.0.0.0/25", "10.0.0.0/26", "10.0.0.0/27", "10.0.0.0/28",
		"10.0.0.0/29", "10.0.0.0/30", "10.0.0.0/31"}

	for _, subnet := range input {
		assertGetAddress(t, subnet)
	}
}

func TestRequestSyntaxCheck(t *testing.T) {
	var (
		pool    = "192.168.0.0/16"
		subPool = "192.168.0.0/24"
		as      = "green"
		err     error
	)

	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	a.addrSpaces[as] = &addrSpace{
		id:      dsConfigKey + "/" + as,
		ds:      a.addrSpaces[localAddressSpace].ds,
		alloc:   a.addrSpaces[localAddressSpace].alloc,
		scope:   a.addrSpaces[localAddressSpace].scope,
		subnets: map[SubnetKey]*PoolData{},
	}

	_, _, _, err = a.RequestPool("", pool, "", nil, false)
	if err == nil {
		t.Fatal("Failed to detect wrong request: empty address space")
	}

	_, _, _, err = a.RequestPool("", pool, subPool, nil, false)
	if err == nil {
		t.Fatal("Failed to detect wrong request: empty address space")
	}

	_, _, _, err = a.RequestPool(as, "", subPool, nil, false)
	if err == nil {
		t.Fatal("Failed to detect wrong request: subPool specified and no pool")
	}

	pid, _, _, err := a.RequestPool(as, pool, subPool, nil, false)
	if err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}

	_, _, err = a.RequestAddress("", nil, nil)
	if err == nil {
		t.Fatal("Failed to detect wrong request: no pool id specified")
	}

	ip := net.ParseIP("172.17.0.23")
	_, _, err = a.RequestAddress(pid, ip, nil)
	if err == nil {
		t.Fatal("Failed to detect wrong request: requested IP from different subnet")
	}

	ip = net.ParseIP("192.168.0.50")
	_, _, err = a.RequestAddress(pid, ip, nil)
	if err != nil {
		t.Fatalf("Unexpected failure: %v", err)
	}

	err = a.ReleaseAddress("", ip)
	if err == nil {
		t.Fatal("Failed to detect wrong request: no pool id specified")
	}

	err = a.ReleaseAddress(pid, nil)
	if err == nil {
		t.Fatal("Failed to detect wrong request: no pool id specified")
	}

	err = a.ReleaseAddress(pid, ip)
	if err != nil {
		t.Fatalf("Unexpected failure: %v: %s, %s", err, pid, ip)
	}
}

func TestRequest(t *testing.T) {
	// Request N addresses from different size subnets, verifying last request
	// returns expected address. Internal subnet host size is Allocator's default, 16
	input := []struct {
		subnet string
		numReq int
		lastIP string
	}{
		{"192.168.59.0/24", 254, "192.168.59.254"},
		{"192.168.240.0/20", 255, "192.168.240.255"},
		{"192.168.0.0/16", 255, "192.168.0.255"},
		{"192.168.0.0/16", 256, "192.168.1.0"},
		{"10.16.0.0/16", 255, "10.16.0.255"},
		{"10.128.0.0/12", 255, "10.128.0.255"},
		{"10.0.0.0/8", 256, "10.0.1.0"},

		{"192.168.128.0/18", 4*256 - 1, "192.168.131.255"},
		/*
			{"192.168.240.0/20", 16*256 - 2, "192.168.255.254"},

			{"192.168.0.0/16", 256*256 - 2, "192.168.255.254"},
			{"10.0.0.0/8", 2 * 256, "10.0.2.0"},
			{"10.0.0.0/8", 5 * 256, "10.0.5.0"},
			{"10.0.0.0/8", 100 * 256 * 254, "10.99.255.254"},
		*/
	}

	for _, d := range input {
		assertNRequests(t, d.subnet, d.numReq, d.lastIP)
	}
}

func TestRelease(t *testing.T) {
	var (
		subnet = "192.168.0.0/23"
	)

	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	pid, _, _, err := a.RequestPool(localAddressSpace, subnet, "", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	bm := a.addresses[SubnetKey{localAddressSpace, subnet, ""}]

	// Allocate all addresses
	for err != ipamapi.ErrNoAvailableIPs {
		_, _, err = a.RequestAddress(pid, nil, nil)
	}

	toRelease := []struct {
		address string
	}{
		{"192.168.0.1"},
		{"192.168.0.2"},
		{"192.168.0.3"},
		{"192.168.0.4"},
		{"192.168.0.5"},
		{"192.168.0.6"},
		{"192.168.0.7"},
		{"192.168.0.8"},
		{"192.168.0.9"},
		{"192.168.0.10"},
		{"192.168.0.30"},
		{"192.168.0.31"},
		{"192.168.1.32"},

		{"192.168.0.254"},
		{"192.168.1.1"},
		{"192.168.1.2"},

		{"192.168.1.3"},

		{"192.168.1.253"},
		{"192.168.1.254"},
	}

	// One by one, relase the address and request again. We should get the same IP
	for i, inp := range toRelease {
		ip0 := net.ParseIP(inp.address)
		a.ReleaseAddress(pid, ip0)
		bm = a.addresses[SubnetKey{localAddressSpace, subnet, ""}]
		if bm.Unselected() != 1 {
			t.Fatalf("Failed to update free address count after release. Expected %d, Found: %d", i+1, bm.Unselected())
		}

		nw, _, err := a.RequestAddress(pid, nil, nil)
		if err != nil {
			t.Fatalf("Failed to obtain the address: %s", err.Error())
		}
		ip := nw.IP
		if !ip0.Equal(ip) {
			t.Fatalf("Failed to obtain the same address. Expected: %s, Got: %s", ip0, ip)
		}
	}
}

func assertGetAddress(t *testing.T, subnet string) {
	var (
		err       error
		printTime = false
		a         = &Allocator{}
	)

	_, sub, _ := net.ParseCIDR(subnet)
	ones, bits := sub.Mask.Size()
	zeroes := bits - ones
	numAddresses := 1 << uint(zeroes)

	bm, err := bitseq.NewHandle("ipam_test", nil, "default/"+subnet, uint64(numAddresses))
	if err != nil {
		t.Fatal(err)
	}

	start := time.Now()
	run := 0
	for err != ipamapi.ErrNoAvailableIPs {
		_, err = a.getAddress(sub, bm, nil, nil)
		run++
	}
	if printTime {
		fmt.Printf("\nTaken %v, to allocate all addresses on %s. (nemAddresses: %d. Runs: %d)", time.Since(start), subnet, numAddresses, run)
	}
	if bm.Unselected() != 0 {
		t.Fatalf("Unexpected free count after reserving all addresses: %d", bm.Unselected())
	}
	/*
		if bm.Head.Block != expectedMax || bm.Head.Count != numBlocks {
			t.Fatalf("Failed to effectively reserve all addresses on %s. Expected (0x%x, %d) as first sequence. Found (0x%x,%d)",
				subnet, expectedMax, numBlocks, bm.Head.Block, bm.Head.Count)
		}
	*/
}

func assertNRequests(t *testing.T, subnet string, numReq int, lastExpectedIP string) {
	var (
		nw        *net.IPNet
		printTime = false
	)

	lastIP := net.ParseIP(lastExpectedIP)
	a, err := getAllocator()
	if err != nil {
		t.Fatal(err)
	}
	pid, _, _, err := a.RequestPool(localAddressSpace, subnet, "", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	i := 0
	start := time.Now()
	for ; i < numReq; i++ {
		nw, _, err = a.RequestAddress(pid, nil, nil)
	}
	if printTime {
		fmt.Printf("\nTaken %v, to allocate %d addresses on %s\n", time.Since(start), numReq, subnet)
	}

	if !lastIP.Equal(nw.IP) {
		t.Fatalf("Wrong last IP. Expected %s. Got: %s (err: %v, ind: %d)", lastExpectedIP, nw.IP.String(), err, i)
	}
}

func benchmarkRequest(b *testing.B, a *Allocator, subnet string) {
	pid, _, _, err := a.RequestPool(localAddressSpace, subnet, "", nil, false)
	for err != ipamapi.ErrNoAvailableIPs {
		_, _, err = a.RequestAddress(pid, nil, nil)
	}
}

func benchMarkRequest(subnet string, b *testing.B) {
	a, _ := getAllocator()
	for n := 0; n < b.N; n++ {
		benchmarkRequest(b, a, subnet)
	}
}

func BenchmarkRequest_24(b *testing.B) {
	a, _ := getAllocator()
	benchmarkRequest(b, a, "10.0.0.0/24")
}

func BenchmarkRequest_16(b *testing.B) {
	a, _ := getAllocator()
	benchmarkRequest(b, a, "10.0.0.0/16")
}

func BenchmarkRequest_8(b *testing.B) {
	a, _ := getAllocator()
	benchmarkRequest(b, a, "10.0.0.0/8")
}

func TestAllocateRandomDeallocate(t *testing.T) {
	testAllocateRandomDeallocate(t, "172.25.0.0/16", "", 384)
	testAllocateRandomDeallocate(t, "172.25.0.0/16", "172.25.252.0/22", 384)
}

func testAllocateRandomDeallocate(t *testing.T, pool, subPool string, num int) {
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}

	a, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}

	pid, _, _, err := a.RequestPool(localAddressSpace, pool, subPool, nil, false)
	if err != nil {
		t.Fatal(err)
	}

	// Allocate num ip addresses
	indices := make(map[int]*net.IPNet, num)
	allocated := make(map[string]bool, num)
	for i := 0; i < num; i++ {
		ip, _, err := a.RequestAddress(pid, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		ips := ip.String()
		if _, ok := allocated[ips]; ok {
			t.Fatalf("Address %s is already allocated", ips)
		}
		allocated[ips] = true
		indices[i] = ip
	}
	if len(indices) != len(allocated) || len(indices) != num {
		t.Fatalf("Unexpected number of allocated addresses: (%d,%d).", len(indices), len(allocated))
	}

	seed := time.Now().Unix()
	rand.Seed(seed)

	// Deallocate half of the allocated addresses following a random pattern
	pattern := rand.Perm(num)
	for i := 0; i < num/2; i++ {
		idx := pattern[i]
		ip := indices[idx]
		err := a.ReleaseAddress(pid, ip.IP)
		if err != nil {
			t.Fatalf("Unexpected failure on deallocation of %s: %v.\nSeed: %d.", ip, err, seed)
		}
		delete(indices, idx)
		delete(allocated, ip.String())
	}

	// Request a quarter of addresses
	for i := 0; i < num/2; i++ {
		ip, _, err := a.RequestAddress(pid, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		ips := ip.String()
		if _, ok := allocated[ips]; ok {
			t.Fatalf("\nAddress %s is already allocated.\nSeed: %d.", ips, seed)
		}
		allocated[ips] = true
	}
	if len(allocated) != num {
		t.Fatalf("Unexpected number of allocated addresses: %d.\nSeed: %d.", len(allocated), seed)
	}
}

func TestRetrieveFromStore(t *testing.T) {
	num := 200
	ds, err := randomLocalStore()
	if err != nil {
		t.Fatal(err)
	}
	a, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}
	pid, _, _, err := a.RequestPool(localAddressSpace, "172.25.0.0/16", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < num; i++ {
		if _, _, err := a.RequestAddress(pid, nil, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Restore
	a1, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}
	a1.refresh(localAddressSpace)
	db := a.DumpDatabase()
	db1 := a1.DumpDatabase()
	if db != db1 {
		t.Fatalf("Unexpected db change.\nExpected:%s\nGot:%s", db, db1)
	}
	checkDBEquality(a, a1, t)
	pid, _, _, err = a1.RequestPool(localAddressSpace, "172.25.0.0/16", "172.25.1.0/24", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < num/2; i++ {
		if _, _, err := a1.RequestAddress(pid, nil, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Restore
	a2, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}
	a2.refresh(localAddressSpace)
	checkDBEquality(a1, a2, t)
	pid, _, _, err = a2.RequestPool(localAddressSpace, "172.25.0.0/16", "172.25.2.0/24", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < num/2; i++ {
		if _, _, err := a2.RequestAddress(pid, nil, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Restore
	a3, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}
	a3.refresh(localAddressSpace)
	checkDBEquality(a2, a3, t)
	pid, _, _, err = a3.RequestPool(localAddressSpace, "172.26.0.0/16", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < num/2; i++ {
		if _, _, err := a3.RequestAddress(pid, nil, nil); err != nil {
			t.Fatal(err)
		}
	}

	// Restore
	a4, err := NewAllocator(ds, nil)
	if err != nil {
		t.Fatal(err)
	}
	a4.refresh(localAddressSpace)
	checkDBEquality(a3, a4, t)
}

func checkDBEquality(a1, a2 *Allocator, t *testing.T) {
	for k, cnf1 := range a1.addrSpaces[localAddressSpace].subnets {
		cnf2 := a2.addrSpaces[localAddressSpace].subnets[k]
		if cnf1.String() != cnf2.String() {
			t.Fatalf("%s\n%s", cnf1, cnf2)
		}
		if cnf1.Range == nil {
			a2.retrieveBitmask(k, cnf1.Pool)
		}
	}

	for k, bm1 := range a1.addresses {
		bm2 := a2.addresses[k]
		if bm1.String() != bm2.String() {
			t.Fatalf("%s\n%s", bm1, bm2)
		}
	}
}

const (
	numInstances = 5
	first        = 0
	last         = numInstances - 1
)

var (
	allocator *Allocator
	start     = make(chan struct{})
	done      = make(chan chan struct{}, numInstances-1)
	pools     = make([]*net.IPNet, numInstances)
)

func runParallelTests(t *testing.T, instance int) {
	var err error

	t.Parallel()

	pTest := flag.Lookup("test.parallel")
	if pTest == nil {
		t.Skip("Skipped because test.parallel flag not set;")
	}
	numParallel, err := strconv.Atoi(pTest.Value.String())
	if err != nil {
		t.Fatal(err)
	}
	if numParallel < numInstances {
		t.Skip("Skipped because t.parallel was less than ", numInstances)
	}

	// The first instance creates the allocator, gives the start
	// and finally checks the pools each instance was assigned
	if instance == first {
		allocator, err = getAllocator()
		if err != nil {
			t.Fatal(err)
		}
		close(start)
	}

	if instance != first {
		select {
		case <-start:
		}

		instDone := make(chan struct{})
		done <- instDone
		defer close(instDone)

		if instance == last {
			defer close(done)
		}
	}

	_, pools[instance], _, err = allocator.RequestPool(localAddressSpace, "", "", nil, false)
	if err != nil {
		t.Fatal(err)
	}

	if instance == first {
		for instDone := range done {
			select {
			case <-instDone:
			}
		}
		// Now check each instance got a different pool
		for i := 0; i < numInstances; i++ {
			for j := i + 1; j < numInstances; j++ {
				if types.CompareIPNet(pools[i], pools[j]) {
					t.Fatalf("Instance %d and %d were given the same predefined pool: %v", i, j, pools)
				}
			}
		}
	}
}

func TestParallelPredefinedRequest1(t *testing.T) {
	runParallelTests(t, 0)
}

func TestParallelPredefinedRequest2(t *testing.T) {
	runParallelTests(t, 1)
}

func TestParallelPredefinedRequest3(t *testing.T) {
	runParallelTests(t, 2)
}

func TestParallelPredefinedRequest4(t *testing.T) {
	runParallelTests(t, 3)
}

func TestParallelPredefinedRequest5(t *testing.T) {
	runParallelTests(t, 4)
}

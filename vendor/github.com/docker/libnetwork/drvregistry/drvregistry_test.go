package drvregistry

import (
	"flag"
	"sort"
	"testing"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/ipamapi"
	builtinIpam "github.com/docker/libnetwork/ipams/builtin"
	nullIpam "github.com/docker/libnetwork/ipams/null"
	remoteIpam "github.com/docker/libnetwork/ipams/remote"
	"github.com/stretchr/testify/assert"
)

var runningInContainer = flag.Bool("incontainer", false,
	"Indicates if the test is running in a container")

const mockDriverName = "mock-driver"

type mockDriver struct{}

var md = mockDriver{}

func mockDriverInit(reg driverapi.DriverCallback, opt map[string]interface{}) error {
	return reg.RegisterDriver(mockDriverName, &md, driverapi.Capability{DataScope: datastore.LocalScope})
}

func (m *mockDriver) CreateNetwork(nid string, options map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	return nil
}

func (m *mockDriver) DeleteNetwork(nid string) error {
	return nil
}

func (m *mockDriver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo, options map[string]interface{}) error {
	return nil
}

func (m *mockDriver) DeleteEndpoint(nid, eid string) error {
	return nil
}

func (m *mockDriver) EndpointOperInfo(nid, eid string) (map[string]interface{}, error) {
	return nil, nil
}

func (m *mockDriver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	return nil
}

func (m *mockDriver) Leave(nid, eid string) error {
	return nil
}

func (m *mockDriver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

func (m *mockDriver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

func (m *mockDriver) Type() string {
	return mockDriverName
}

func (m *mockDriver) IsBuiltIn() bool {
	return true
}

func (m *mockDriver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	return nil
}

func (m *mockDriver) RevokeExternalConnectivity(nid, eid string) error {
	return nil
}

func (m *mockDriver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	return nil, nil
}

func (m *mockDriver) NetworkFree(id string) error {
	return nil
}

func (m *mockDriver) EventNotify(etype driverapi.EventType, nid, tableName, key string, value []byte) {
}

func (m *mockDriver) DecodeTableEntry(tablename string, key string, value []byte) (string, map[string]string) {
	return "", nil
}

func getNew(t *testing.T) *DrvRegistry {
	reg, err := New(nil, nil, nil, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	err = initIPAMDrivers(reg, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	return reg
}

func initIPAMDrivers(r *DrvRegistry, lDs, gDs interface{}) error {
	for _, fn := range [](func(ipamapi.Callback, interface{}, interface{}) error){
		builtinIpam.Init,
		remoteIpam.Init,
		nullIpam.Init,
	} {
		if err := fn(r, lDs, gDs); err != nil {
			return err
		}
	}

	return nil
}
func TestNew(t *testing.T) {
	getNew(t)
}

func TestAddDriver(t *testing.T) {
	reg := getNew(t)

	err := reg.AddDriver(mockDriverName, mockDriverInit, nil)
	assert.NoError(t, err)
}

func TestAddDuplicateDriver(t *testing.T) {
	reg := getNew(t)

	err := reg.AddDriver(mockDriverName, mockDriverInit, nil)
	assert.NoError(t, err)

	// Try adding the same driver
	err = reg.AddDriver(mockDriverName, mockDriverInit, nil)
	assert.Error(t, err)
}

func TestIPAMDefaultAddressSpaces(t *testing.T) {
	reg := getNew(t)

	as1, as2, err := reg.IPAMDefaultAddressSpaces("default")
	assert.NoError(t, err)
	assert.NotEqual(t, as1, "")
	assert.NotEqual(t, as2, "")
}

func TestDriver(t *testing.T) {
	reg := getNew(t)

	err := reg.AddDriver(mockDriverName, mockDriverInit, nil)
	assert.NoError(t, err)

	d, cap := reg.Driver(mockDriverName)
	assert.NotEqual(t, d, nil)
	assert.NotEqual(t, cap, nil)
}

func TestIPAM(t *testing.T) {
	reg := getNew(t)

	i, cap := reg.IPAM("default")
	assert.NotEqual(t, i, nil)
	assert.NotEqual(t, cap, nil)
}

func TestWalkIPAMs(t *testing.T) {
	reg := getNew(t)

	ipams := make([]string, 0, 2)
	reg.WalkIPAMs(func(name string, driver ipamapi.Ipam, cap *ipamapi.Capability) bool {
		ipams = append(ipams, name)
		return false
	})

	sort.Strings(ipams)
	assert.Equal(t, ipams, []string{"default", "null"})
}

func TestWalkDrivers(t *testing.T) {
	reg := getNew(t)

	err := reg.AddDriver(mockDriverName, mockDriverInit, nil)
	assert.NoError(t, err)

	var driverName string
	reg.WalkDrivers(func(name string, driver driverapi.Driver, capability driverapi.Capability) bool {
		driverName = name
		return false
	})

	assert.Equal(t, driverName, mockDriverName)
}

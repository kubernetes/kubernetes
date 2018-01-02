package ivmanager

import (
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/types"
)

const networkType = "ipvlan"

type driver struct{}

// Init registers a new instance of ipvlan manager driver
func Init(dc driverapi.DriverCallback, config map[string]interface{}) error {
	c := driverapi.Capability{
		DataScope:         datastore.LocalScope,
		ConnectivityScope: datastore.GlobalScope,
	}
	return dc.RegisterDriver(networkType, &driver{}, c)
}

func (d *driver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	return nil, types.NotImplementedErrorf("not implemented")
}

func (d *driver) NetworkFree(id string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) CreateNetwork(id string, option map[string]interface{}, nInfo driverapi.NetworkInfo, ipV4Data, ipV6Data []driverapi.IPAMData) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) EventNotify(etype driverapi.EventType, nid, tableName, key string, value []byte) {
}

func (d *driver) DecodeTableEntry(tablename string, key string, value []byte) (string, map[string]string) {
	return "", nil
}

func (d *driver) DeleteNetwork(nid string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) CreateEndpoint(nid, eid string, ifInfo driverapi.InterfaceInfo, epOptions map[string]interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) DeleteEndpoint(nid, eid string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) EndpointOperInfo(nid, eid string) (map[string]interface{}, error) {
	return nil, types.NotImplementedErrorf("not implemented")
}

func (d *driver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) Leave(nid, eid string) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) Type() string {
	return networkType
}

func (d *driver) IsBuiltIn() bool {
	return true
}

func (d *driver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) RevokeExternalConnectivity(nid, eid string) error {
	return types.NotImplementedErrorf("not implemented")
}

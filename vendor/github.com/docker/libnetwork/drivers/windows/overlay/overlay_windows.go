package overlay

//go:generate protoc -I.:../../Godeps/_workspace/src/github.com/gogo/protobuf  --gogo_out=import_path=github.com/docker/libnetwork/drivers/overlay,Mgogoproto/gogo.proto=github.com/gogo/protobuf/gogoproto:. overlay.proto

import (
	"encoding/json"
	"net"
	"sync"

	"github.com/Microsoft/hcsshim"
	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	networkType  = "overlay"
	vethPrefix   = "veth"
	vethLen      = 7
	secureOption = "encrypted"
)

type driver struct {
	config     map[string]interface{}
	networks   networkTable
	store      datastore.DataStore
	localStore datastore.DataStore
	once       sync.Once
	joinOnce   sync.Once
	sync.Mutex
}

// Init registers a new instance of overlay driver
func Init(dc driverapi.DriverCallback, config map[string]interface{}) error {
	c := driverapi.Capability{
		DataScope:         datastore.GlobalScope,
		ConnectivityScope: datastore.GlobalScope,
	}

	d := &driver{
		networks: networkTable{},
		config:   config,
	}

	if data, ok := config[netlabel.GlobalKVClient]; ok {
		var err error
		dsc, ok := data.(discoverapi.DatastoreConfigData)
		if !ok {
			return types.InternalErrorf("incorrect data in datastore configuration: %v", data)
		}
		d.store, err = datastore.NewDataStoreFromConfig(dsc)
		if err != nil {
			return types.InternalErrorf("failed to initialize data store: %v", err)
		}
	}

	if data, ok := config[netlabel.LocalKVClient]; ok {
		var err error
		dsc, ok := data.(discoverapi.DatastoreConfigData)
		if !ok {
			return types.InternalErrorf("incorrect data in datastore configuration: %v", data)
		}
		d.localStore, err = datastore.NewDataStoreFromConfig(dsc)
		if err != nil {
			return types.InternalErrorf("failed to initialize local data store: %v", err)
		}
	}

	d.restoreHNSNetworks()

	return dc.RegisterDriver(networkType, d, c)
}

func (d *driver) restoreHNSNetworks() error {
	logrus.Infof("Restoring existing overlay networks from HNS into docker")

	hnsresponse, err := hcsshim.HNSListNetworkRequest("GET", "", "")
	if err != nil {
		return err
	}

	for _, v := range hnsresponse {
		if v.Type != networkType {
			continue
		}

		logrus.Infof("Restoring overlay network: %s", v.Name)
		n := d.convertToOverlayNetwork(&v)
		d.addNetwork(n)

		//
		// We assume that any network will be recreated on daemon restart
		// and therefore don't restore hns endpoints for now
		//
		//n.restoreNetworkEndpoints()
	}

	return nil
}

func (d *driver) convertToOverlayNetwork(v *hcsshim.HNSNetwork) *network {
	n := &network{
		id:              v.Name,
		hnsID:           v.Id,
		driver:          d,
		endpoints:       endpointTable{},
		subnets:         []*subnet{},
		providerAddress: v.ManagementIP,
	}

	for _, hnsSubnet := range v.Subnets {
		vsidPolicy := &hcsshim.VsidPolicy{}
		for _, policy := range hnsSubnet.Policies {
			if err := json.Unmarshal([]byte(policy), &vsidPolicy); err == nil && vsidPolicy.Type == "VSID" {
				break
			}
		}

		gwIP := net.ParseIP(hnsSubnet.GatewayAddress)
		localsubnet := &subnet{
			vni:  uint32(vsidPolicy.VSID),
			gwIP: &gwIP,
		}

		_, subnetIP, err := net.ParseCIDR(hnsSubnet.AddressPrefix)

		if err != nil {
			logrus.Errorf("Error parsing subnet address %s ", hnsSubnet.AddressPrefix)
			continue
		}

		localsubnet.subnetIP = subnetIP

		n.subnets = append(n.subnets, localsubnet)
	}

	return n
}

func (d *driver) Type() string {
	return networkType
}

func (d *driver) IsBuiltIn() bool {
	return true
}

// DiscoverNew is a notification for a new discovery event, such as a new node joining a cluster
func (d *driver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

// DiscoverDelete is a notification for a discovery delete event, such as a node leaving a cluster
func (d *driver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

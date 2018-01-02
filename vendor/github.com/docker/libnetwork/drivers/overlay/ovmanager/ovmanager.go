package ovmanager

import (
	"fmt"
	"net"
	"strconv"
	"strings"
	"sync"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/idm"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	networkType  = "overlay"
	vxlanIDStart = 4096
	vxlanIDEnd   = (1 << 24) - 1
)

type networkTable map[string]*network

type driver struct {
	config   map[string]interface{}
	networks networkTable
	store    datastore.DataStore
	vxlanIdm *idm.Idm
	sync.Mutex
}

type subnet struct {
	subnetIP *net.IPNet
	gwIP     *net.IPNet
	vni      uint32
}

type network struct {
	id      string
	driver  *driver
	subnets []*subnet
	sync.Mutex
}

// Init registers a new instance of overlay driver
func Init(dc driverapi.DriverCallback, config map[string]interface{}) error {
	var err error
	c := driverapi.Capability{
		DataScope:         datastore.GlobalScope,
		ConnectivityScope: datastore.GlobalScope,
	}

	d := &driver{
		networks: networkTable{},
		config:   config,
	}

	d.vxlanIdm, err = idm.New(nil, "vxlan-id", 0, vxlanIDEnd)
	if err != nil {
		return fmt.Errorf("failed to initialize vxlan id manager: %v", err)
	}

	return dc.RegisterDriver(networkType, d, c)
}

func (d *driver) NetworkAllocate(id string, option map[string]string, ipV4Data, ipV6Data []driverapi.IPAMData) (map[string]string, error) {
	if id == "" {
		return nil, fmt.Errorf("invalid network id for overlay network")
	}

	if ipV4Data == nil {
		return nil, fmt.Errorf("empty ipv4 data passed during overlay network creation")
	}

	n := &network{
		id:      id,
		driver:  d,
		subnets: []*subnet{},
	}

	opts := make(map[string]string)
	vxlanIDList := make([]uint32, 0, len(ipV4Data))
	for key, val := range option {
		if key == netlabel.OverlayVxlanIDList {
			logrus.Debugf("overlay network option: %s", val)
			valStrList := strings.Split(val, ",")
			for _, idStr := range valStrList {
				vni, err := strconv.Atoi(idStr)
				if err != nil {
					return nil, fmt.Errorf("invalid vxlan id value %q passed", idStr)
				}

				vxlanIDList = append(vxlanIDList, uint32(vni))
			}
		} else {
			opts[key] = val
		}
	}

	for i, ipd := range ipV4Data {
		s := &subnet{
			subnetIP: ipd.Pool,
			gwIP:     ipd.Gateway,
		}

		if len(vxlanIDList) > i {
			s.vni = vxlanIDList[i]
		}

		if err := n.obtainVxlanID(s); err != nil {
			n.releaseVxlanID()
			return nil, fmt.Errorf("could not obtain vxlan id for pool %s: %v", s.subnetIP, err)
		}

		n.subnets = append(n.subnets, s)
	}

	val := fmt.Sprintf("%d", n.subnets[0].vni)
	for _, s := range n.subnets[1:] {
		val = val + fmt.Sprintf(",%d", s.vni)
	}
	opts[netlabel.OverlayVxlanIDList] = val

	d.Lock()
	d.networks[id] = n
	d.Unlock()

	return opts, nil
}

func (d *driver) NetworkFree(id string) error {
	if id == "" {
		return fmt.Errorf("invalid network id passed while freeing overlay network")
	}

	d.Lock()
	n, ok := d.networks[id]
	d.Unlock()

	if !ok {
		return fmt.Errorf("overlay network with id %s not found", id)
	}

	// Release all vxlan IDs in one shot.
	n.releaseVxlanID()

	d.Lock()
	delete(d.networks, id)
	d.Unlock()

	return nil
}

func (n *network) obtainVxlanID(s *subnet) error {
	var (
		err error
		vni uint64
	)

	n.Lock()
	vni = uint64(s.vni)
	n.Unlock()

	if vni == 0 {
		vni, err = n.driver.vxlanIdm.GetIDInRange(vxlanIDStart, vxlanIDEnd)
		if err != nil {
			return err
		}

		n.Lock()
		s.vni = uint32(vni)
		n.Unlock()
		return nil
	}

	return n.driver.vxlanIdm.GetSpecificID(vni)
}

func (n *network) releaseVxlanID() {
	n.Lock()
	vnis := make([]uint32, 0, len(n.subnets))
	for _, s := range n.subnets {
		vnis = append(vnis, s.vni)
		s.vni = 0
	}
	n.Unlock()

	for _, vni := range vnis {
		n.driver.vxlanIdm.Release(uint64(vni))
	}
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

// Join method is invoked when a Sandbox is attached to an endpoint.
func (d *driver) Join(nid, eid string, sboxKey string, jinfo driverapi.JoinInfo, options map[string]interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

// Leave method is invoked when a Sandbox detaches from an endpoint.
func (d *driver) Leave(nid, eid string) error {
	return types.NotImplementedErrorf("not implemented")
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

func (d *driver) ProgramExternalConnectivity(nid, eid string, options map[string]interface{}) error {
	return types.NotImplementedErrorf("not implemented")
}

func (d *driver) RevokeExternalConnectivity(nid, eid string) error {
	return types.NotImplementedErrorf("not implemented")
}

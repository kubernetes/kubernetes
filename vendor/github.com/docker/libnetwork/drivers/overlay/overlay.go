package overlay

//go:generate protoc -I.:../../Godeps/_workspace/src/github.com/gogo/protobuf  --gogo_out=import_path=github.com/docker/libnetwork/drivers/overlay,Mgogoproto/gogo.proto=github.com/gogo/protobuf/gogoproto:. overlay.proto

import (
	"context"
	"fmt"
	"net"
	"sync"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/driverapi"
	"github.com/docker/libnetwork/idm"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/osl"
	"github.com/docker/libnetwork/types"
	"github.com/hashicorp/serf/serf"
	"github.com/sirupsen/logrus"
)

const (
	networkType  = "overlay"
	vethPrefix   = "veth"
	vethLen      = 7
	vxlanIDStart = 256
	vxlanIDEnd   = (1 << 24) - 1
	vxlanPort    = 4789
	vxlanEncap   = 50
	secureOption = "encrypted"
)

var initVxlanIdm = make(chan (bool), 1)

type driver struct {
	eventCh          chan serf.Event
	notifyCh         chan ovNotify
	exitCh           chan chan struct{}
	bindAddress      string
	advertiseAddress string
	neighIP          string
	config           map[string]interface{}
	peerDb           peerNetworkMap
	secMap           *encrMap
	serfInstance     *serf.Serf
	networks         networkTable
	store            datastore.DataStore
	localStore       datastore.DataStore
	vxlanIdm         *idm.Idm
	initOS           sync.Once
	joinOnce         sync.Once
	localJoinOnce    sync.Once
	keys             []*key
	peerOpCh         chan *peerOperation
	peerOpCancel     context.CancelFunc
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
		peerDb: peerNetworkMap{
			mp: map[string]*peerMap{},
		},
		secMap:   &encrMap{nodes: map[string][]*spi{}},
		config:   config,
		peerOpCh: make(chan *peerOperation),
	}

	// Launch the go routine for processing peer operations
	ctx, cancel := context.WithCancel(context.Background())
	d.peerOpCancel = cancel
	go d.peerOpRoutine(ctx, d.peerOpCh)

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

	if err := d.restoreEndpoints(); err != nil {
		logrus.Warnf("Failure during overlay endpoints restore: %v", err)
	}

	// If an error happened when the network join the sandbox during the endpoints restore
	// we should reset it now along with the once variable, so that subsequent endpoint joins
	// outside of the restore path can potentially fix the network join and succeed.
	for nid, n := range d.networks {
		if n.initErr != nil {
			logrus.Infof("resetting init error and once variable for network %s after unsuccessful endpoint restore: %v", nid, n.initErr)
			n.initErr = nil
			n.once = &sync.Once{}
		}
	}

	return dc.RegisterDriver(networkType, d, c)
}

// Endpoints are stored in the local store. Restore them and reconstruct the overlay sandbox
func (d *driver) restoreEndpoints() error {
	if d.localStore == nil {
		logrus.Warn("Cannot restore overlay endpoints because local datastore is missing")
		return nil
	}
	kvol, err := d.localStore.List(datastore.Key(overlayEndpointPrefix), &endpoint{})
	if err != nil && err != datastore.ErrKeyNotFound {
		return fmt.Errorf("failed to read overlay endpoint from store: %v", err)
	}

	if err == datastore.ErrKeyNotFound {
		return nil
	}
	for _, kvo := range kvol {
		ep := kvo.(*endpoint)
		n := d.network(ep.nid)
		if n == nil {
			logrus.Debugf("Network (%s) not found for restored endpoint (%s)", ep.nid[0:7], ep.id[0:7])
			logrus.Debugf("Deleting stale overlay endpoint (%s) from store", ep.id[0:7])
			if err := d.deleteEndpointFromStore(ep); err != nil {
				logrus.Debugf("Failed to delete stale overlay endpoint (%s) from store", ep.id[0:7])
			}
			continue
		}
		n.addEndpoint(ep)

		s := n.getSubnetforIP(ep.addr)
		if s == nil {
			return fmt.Errorf("could not find subnet for endpoint %s", ep.id)
		}

		if err := n.joinSandbox(true); err != nil {
			return fmt.Errorf("restore network sandbox failed: %v", err)
		}

		if err := n.joinSubnetSandbox(s, true); err != nil {
			return fmt.Errorf("restore subnet sandbox failed for %q: %v", s.subnetIP.String(), err)
		}

		Ifaces := make(map[string][]osl.IfaceOption)
		vethIfaceOption := make([]osl.IfaceOption, 1)
		vethIfaceOption = append(vethIfaceOption, n.sbox.InterfaceOptions().Master(s.brName))
		Ifaces["veth+veth"] = vethIfaceOption

		err := n.sbox.Restore(Ifaces, nil, nil, nil)
		if err != nil {
			return fmt.Errorf("failed to restore overlay sandbox: %v", err)
		}

		n.incEndpointCount()
		d.peerAdd(ep.nid, ep.id, ep.addr.IP, ep.addr.Mask, ep.mac, net.ParseIP(d.advertiseAddress), false, false, true)
	}
	return nil
}

// Fini cleans up the driver resources
func Fini(drv driverapi.Driver) {
	d := drv.(*driver)

	// Notify the peer go routine to return
	if d.peerOpCancel != nil {
		d.peerOpCancel()
	}

	if d.exitCh != nil {
		waitCh := make(chan struct{})

		d.exitCh <- waitCh

		<-waitCh
	}
}

func (d *driver) configure() error {

	// Apply OS specific kernel configs if needed
	d.initOS.Do(applyOStweaks)

	if d.store == nil {
		return nil
	}

	if d.vxlanIdm == nil {
		return d.initializeVxlanIdm()
	}

	return nil
}

func (d *driver) initializeVxlanIdm() error {
	var err error

	initVxlanIdm <- true
	defer func() { <-initVxlanIdm }()

	if d.vxlanIdm != nil {
		return nil
	}

	d.vxlanIdm, err = idm.New(d.store, "vxlan-id", vxlanIDStart, vxlanIDEnd)
	if err != nil {
		return fmt.Errorf("failed to initialize vxlan id manager: %v", err)
	}

	return nil
}

func (d *driver) Type() string {
	return networkType
}

func (d *driver) IsBuiltIn() bool {
	return true
}

func validateSelf(node string) error {
	advIP := net.ParseIP(node)
	if advIP == nil {
		return fmt.Errorf("invalid self address (%s)", node)
	}

	addrs, err := net.InterfaceAddrs()
	if err != nil {
		return fmt.Errorf("Unable to get interface addresses %v", err)
	}
	for _, addr := range addrs {
		ip, _, err := net.ParseCIDR(addr.String())
		if err == nil && ip.Equal(advIP) {
			return nil
		}
	}
	return fmt.Errorf("Multi-Host overlay networking requires cluster-advertise(%s) to be configured with a local ip-address that is reachable within the cluster", advIP.String())
}

func (d *driver) nodeJoin(advertiseAddress, bindAddress string, self bool) {
	if self && !d.isSerfAlive() {
		d.Lock()
		d.advertiseAddress = advertiseAddress
		d.bindAddress = bindAddress
		d.Unlock()

		// If containers are already running on this network update the
		// advertiseaddress in the peerDB
		d.localJoinOnce.Do(func() {
			d.peerDBUpdateSelf()
		})

		// If there is no cluster store there is no need to start serf.
		if d.store != nil {
			if err := validateSelf(advertiseAddress); err != nil {
				logrus.Warn(err.Error())
			}
			err := d.serfInit()
			if err != nil {
				logrus.Errorf("initializing serf instance failed: %v", err)
				d.Lock()
				d.advertiseAddress = ""
				d.bindAddress = ""
				d.Unlock()
				return
			}
		}
	}

	d.Lock()
	if !self {
		d.neighIP = advertiseAddress
	}
	neighIP := d.neighIP
	d.Unlock()

	if d.serfInstance != nil && neighIP != "" {
		var err error
		d.joinOnce.Do(func() {
			err = d.serfJoin(neighIP)
			if err == nil {
				d.pushLocalDb()
			}
		})
		if err != nil {
			logrus.Errorf("joining serf neighbor %s failed: %v", advertiseAddress, err)
			d.Lock()
			d.joinOnce = sync.Once{}
			d.Unlock()
			return
		}
	}
}

func (d *driver) pushLocalEndpointEvent(action, nid, eid string) {
	n := d.network(nid)
	if n == nil {
		logrus.Debugf("Error pushing local endpoint event for network %s", nid)
		return
	}
	ep := n.endpoint(eid)
	if ep == nil {
		logrus.Debugf("Error pushing local endpoint event for ep %s / %s", nid, eid)
		return
	}

	if !d.isSerfAlive() {
		return
	}
	d.notifyCh <- ovNotify{
		action: "join",
		nw:     n,
		ep:     ep,
	}
}

// DiscoverNew is a notification for a new discovery event, such as a new node joining a cluster
func (d *driver) DiscoverNew(dType discoverapi.DiscoveryType, data interface{}) error {
	var err error
	switch dType {
	case discoverapi.NodeDiscovery:
		nodeData, ok := data.(discoverapi.NodeDiscoveryData)
		if !ok || nodeData.Address == "" {
			return fmt.Errorf("invalid discovery data")
		}
		d.nodeJoin(nodeData.Address, nodeData.BindAddress, nodeData.Self)
	case discoverapi.DatastoreConfig:
		if d.store != nil {
			return types.ForbiddenErrorf("cannot accept datastore configuration: Overlay driver has a datastore configured already")
		}
		dsc, ok := data.(discoverapi.DatastoreConfigData)
		if !ok {
			return types.InternalErrorf("incorrect data in datastore configuration: %v", data)
		}
		d.store, err = datastore.NewDataStoreFromConfig(dsc)
		if err != nil {
			return types.InternalErrorf("failed to initialize data store: %v", err)
		}
	case discoverapi.EncryptionKeysConfig:
		encrData, ok := data.(discoverapi.DriverEncryptionConfig)
		if !ok {
			return fmt.Errorf("invalid encryption key notification data")
		}
		keys := make([]*key, 0, len(encrData.Keys))
		for i := 0; i < len(encrData.Keys); i++ {
			k := &key{
				value: encrData.Keys[i],
				tag:   uint32(encrData.Tags[i]),
			}
			keys = append(keys, k)
		}
		if err := d.setKeys(keys); err != nil {
			logrus.Warn(err)
		}
	case discoverapi.EncryptionKeysUpdate:
		var newKey, delKey, priKey *key
		encrData, ok := data.(discoverapi.DriverEncryptionUpdate)
		if !ok {
			return fmt.Errorf("invalid encryption key notification data")
		}
		if encrData.Key != nil {
			newKey = &key{
				value: encrData.Key,
				tag:   uint32(encrData.Tag),
			}
		}
		if encrData.Primary != nil {
			priKey = &key{
				value: encrData.Primary,
				tag:   uint32(encrData.PrimaryTag),
			}
		}
		if encrData.Prune != nil {
			delKey = &key{
				value: encrData.Prune,
				tag:   uint32(encrData.PruneTag),
			}
		}
		if err := d.updateKeys(newKey, priKey, delKey); err != nil {
			logrus.Warn(err)
		}
	default:
	}
	return nil
}

// DiscoverDelete is a notification for a discovery delete event, such as a node leaving a cluster
func (d *driver) DiscoverDelete(dType discoverapi.DiscoveryType, data interface{}) error {
	return nil
}

package bridge

import (
	"encoding/json"
	"fmt"
	"net"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/discoverapi"
	"github.com/docker/libnetwork/netlabel"
	"github.com/docker/libnetwork/types"
	"github.com/sirupsen/logrus"
)

const (
	// network config prefix was not specific enough.
	// To be backward compatible, need custom endpoint
	// prefix with different root
	bridgePrefix         = "bridge"
	bridgeEndpointPrefix = "bridge-endpoint"
)

func (d *driver) initStore(option map[string]interface{}) error {
	if data, ok := option[netlabel.LocalKVClient]; ok {
		var err error
		dsc, ok := data.(discoverapi.DatastoreConfigData)
		if !ok {
			return types.InternalErrorf("incorrect data in datastore configuration: %v", data)
		}
		d.store, err = datastore.NewDataStoreFromConfig(dsc)
		if err != nil {
			return types.InternalErrorf("bridge driver failed to initialize data store: %v", err)
		}

		err = d.populateNetworks()
		if err != nil {
			return err
		}

		err = d.populateEndpoints()
		if err != nil {
			return err
		}
	}

	return nil
}

func (d *driver) populateNetworks() error {
	kvol, err := d.store.List(datastore.Key(bridgePrefix), &networkConfiguration{})
	if err != nil && err != datastore.ErrKeyNotFound {
		return fmt.Errorf("failed to get bridge network configurations from store: %v", err)
	}

	// It's normal for network configuration state to be empty. Just return.
	if err == datastore.ErrKeyNotFound {
		return nil
	}

	for _, kvo := range kvol {
		ncfg := kvo.(*networkConfiguration)
		if err = d.createNetwork(ncfg); err != nil {
			logrus.Warnf("could not create bridge network for id %s bridge name %s while booting up from persistent state: %v", ncfg.ID, ncfg.BridgeName, err)
		}
		logrus.Debugf("Network (%s) restored", ncfg.ID[0:7])
	}

	return nil
}

func (d *driver) populateEndpoints() error {
	kvol, err := d.store.List(datastore.Key(bridgeEndpointPrefix), &bridgeEndpoint{})
	if err != nil && err != datastore.ErrKeyNotFound {
		return fmt.Errorf("failed to get bridge endpoints from store: %v", err)
	}

	if err == datastore.ErrKeyNotFound {
		return nil
	}

	for _, kvo := range kvol {
		ep := kvo.(*bridgeEndpoint)
		n, ok := d.networks[ep.nid]
		if !ok {
			logrus.Debugf("Network (%s) not found for restored bridge endpoint (%s)", ep.nid[0:7], ep.id[0:7])
			logrus.Debugf("Deleting stale bridge endpoint (%s) from store", ep.id[0:7])
			if err := d.storeDelete(ep); err != nil {
				logrus.Debugf("Failed to delete stale bridge endpoint (%s) from store", ep.id[0:7])
			}
			continue
		}
		n.endpoints[ep.id] = ep
		n.restorePortAllocations(ep)
		logrus.Debugf("Endpoint (%s) restored to network (%s)", ep.id[0:7], ep.nid[0:7])
	}

	return nil
}

func (d *driver) storeUpdate(kvObject datastore.KVObject) error {
	if d.store == nil {
		logrus.Warnf("bridge store not initialized. kv object %s is not added to the store", datastore.Key(kvObject.Key()...))
		return nil
	}

	if err := d.store.PutObjectAtomic(kvObject); err != nil {
		return fmt.Errorf("failed to update bridge store for object type %T: %v", kvObject, err)
	}

	return nil
}

func (d *driver) storeDelete(kvObject datastore.KVObject) error {
	if d.store == nil {
		logrus.Debugf("bridge store not initialized. kv object %s is not deleted from store", datastore.Key(kvObject.Key()...))
		return nil
	}

retry:
	if err := d.store.DeleteObjectAtomic(kvObject); err != nil {
		if err == datastore.ErrKeyModified {
			if err := d.store.GetObject(datastore.Key(kvObject.Key()...), kvObject); err != nil {
				return fmt.Errorf("could not update the kvobject to latest when trying to delete: %v", err)
			}
			goto retry
		}
		return err
	}

	return nil
}

func (ncfg *networkConfiguration) MarshalJSON() ([]byte, error) {
	nMap := make(map[string]interface{})
	nMap["ID"] = ncfg.ID
	nMap["BridgeName"] = ncfg.BridgeName
	nMap["EnableIPv6"] = ncfg.EnableIPv6
	nMap["EnableIPMasquerade"] = ncfg.EnableIPMasquerade
	nMap["EnableICC"] = ncfg.EnableICC
	nMap["Mtu"] = ncfg.Mtu
	nMap["Internal"] = ncfg.Internal
	nMap["DefaultBridge"] = ncfg.DefaultBridge
	nMap["DefaultBindingIP"] = ncfg.DefaultBindingIP.String()
	nMap["DefaultGatewayIPv4"] = ncfg.DefaultGatewayIPv4.String()
	nMap["DefaultGatewayIPv6"] = ncfg.DefaultGatewayIPv6.String()
	nMap["ContainerIfacePrefix"] = ncfg.ContainerIfacePrefix
	nMap["BridgeIfaceCreator"] = ncfg.BridgeIfaceCreator

	if ncfg.AddressIPv4 != nil {
		nMap["AddressIPv4"] = ncfg.AddressIPv4.String()
	}

	if ncfg.AddressIPv6 != nil {
		nMap["AddressIPv6"] = ncfg.AddressIPv6.String()
	}

	return json.Marshal(nMap)
}

func (ncfg *networkConfiguration) UnmarshalJSON(b []byte) error {
	var (
		err  error
		nMap map[string]interface{}
	)

	if err = json.Unmarshal(b, &nMap); err != nil {
		return err
	}

	if v, ok := nMap["AddressIPv4"]; ok {
		if ncfg.AddressIPv4, err = types.ParseCIDR(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode bridge network address IPv4 after json unmarshal: %s", v.(string))
		}
	}

	if v, ok := nMap["AddressIPv6"]; ok {
		if ncfg.AddressIPv6, err = types.ParseCIDR(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode bridge network address IPv6 after json unmarshal: %s", v.(string))
		}
	}

	if v, ok := nMap["ContainerIfacePrefix"]; ok {
		ncfg.ContainerIfacePrefix = v.(string)
	}

	ncfg.DefaultBridge = nMap["DefaultBridge"].(bool)
	ncfg.DefaultBindingIP = net.ParseIP(nMap["DefaultBindingIP"].(string))
	ncfg.DefaultGatewayIPv4 = net.ParseIP(nMap["DefaultGatewayIPv4"].(string))
	ncfg.DefaultGatewayIPv6 = net.ParseIP(nMap["DefaultGatewayIPv6"].(string))
	ncfg.ID = nMap["ID"].(string)
	ncfg.BridgeName = nMap["BridgeName"].(string)
	ncfg.EnableIPv6 = nMap["EnableIPv6"].(bool)
	ncfg.EnableIPMasquerade = nMap["EnableIPMasquerade"].(bool)
	ncfg.EnableICC = nMap["EnableICC"].(bool)
	ncfg.Mtu = int(nMap["Mtu"].(float64))
	if v, ok := nMap["Internal"]; ok {
		ncfg.Internal = v.(bool)
	}

	if v, ok := nMap["BridgeIfaceCreator"]; ok {
		ncfg.BridgeIfaceCreator = ifaceCreator(v.(float64))
	}

	return nil
}

func (ncfg *networkConfiguration) Key() []string {
	return []string{bridgePrefix, ncfg.ID}
}

func (ncfg *networkConfiguration) KeyPrefix() []string {
	return []string{bridgePrefix}
}

func (ncfg *networkConfiguration) Value() []byte {
	b, err := json.Marshal(ncfg)
	if err != nil {
		return nil
	}
	return b
}

func (ncfg *networkConfiguration) SetValue(value []byte) error {
	return json.Unmarshal(value, ncfg)
}

func (ncfg *networkConfiguration) Index() uint64 {
	return ncfg.dbIndex
}

func (ncfg *networkConfiguration) SetIndex(index uint64) {
	ncfg.dbIndex = index
	ncfg.dbExists = true
}

func (ncfg *networkConfiguration) Exists() bool {
	return ncfg.dbExists
}

func (ncfg *networkConfiguration) Skip() bool {
	return false
}

func (ncfg *networkConfiguration) New() datastore.KVObject {
	return &networkConfiguration{}
}

func (ncfg *networkConfiguration) CopyTo(o datastore.KVObject) error {
	dstNcfg := o.(*networkConfiguration)
	*dstNcfg = *ncfg
	return nil
}

func (ncfg *networkConfiguration) DataScope() string {
	return datastore.LocalScope
}

func (ep *bridgeEndpoint) MarshalJSON() ([]byte, error) {
	epMap := make(map[string]interface{})
	epMap["id"] = ep.id
	epMap["nid"] = ep.nid
	epMap["SrcName"] = ep.srcName
	epMap["MacAddress"] = ep.macAddress.String()
	epMap["Addr"] = ep.addr.String()
	if ep.addrv6 != nil {
		epMap["Addrv6"] = ep.addrv6.String()
	}
	epMap["Config"] = ep.config
	epMap["ContainerConfig"] = ep.containerConfig
	epMap["ExternalConnConfig"] = ep.extConnConfig
	epMap["PortMapping"] = ep.portMapping

	return json.Marshal(epMap)
}

func (ep *bridgeEndpoint) UnmarshalJSON(b []byte) error {
	var (
		err   error
		epMap map[string]interface{}
	)

	if err = json.Unmarshal(b, &epMap); err != nil {
		return fmt.Errorf("Failed to unmarshal to bridge endpoint: %v", err)
	}

	if v, ok := epMap["MacAddress"]; ok {
		if ep.macAddress, err = net.ParseMAC(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode bridge endpoint MAC address (%s) after json unmarshal: %v", v.(string), err)
		}
	}
	if v, ok := epMap["Addr"]; ok {
		if ep.addr, err = types.ParseCIDR(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode bridge endpoint IPv4 address (%s) after json unmarshal: %v", v.(string), err)
		}
	}
	if v, ok := epMap["Addrv6"]; ok {
		if ep.addrv6, err = types.ParseCIDR(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode bridge endpoint IPv6 address (%s) after json unmarshal: %v", v.(string), err)
		}
	}
	ep.id = epMap["id"].(string)
	ep.nid = epMap["nid"].(string)
	ep.srcName = epMap["SrcName"].(string)
	d, _ := json.Marshal(epMap["Config"])
	if err := json.Unmarshal(d, &ep.config); err != nil {
		logrus.Warnf("Failed to decode endpoint config %v", err)
	}
	d, _ = json.Marshal(epMap["ContainerConfig"])
	if err := json.Unmarshal(d, &ep.containerConfig); err != nil {
		logrus.Warnf("Failed to decode endpoint container config %v", err)
	}
	d, _ = json.Marshal(epMap["ExternalConnConfig"])
	if err := json.Unmarshal(d, &ep.extConnConfig); err != nil {
		logrus.Warnf("Failed to decode endpoint external connectivity configuration %v", err)
	}
	d, _ = json.Marshal(epMap["PortMapping"])
	if err := json.Unmarshal(d, &ep.portMapping); err != nil {
		logrus.Warnf("Failed to decode endpoint port mapping %v", err)
	}

	return nil
}

func (ep *bridgeEndpoint) Key() []string {
	return []string{bridgeEndpointPrefix, ep.id}
}

func (ep *bridgeEndpoint) KeyPrefix() []string {
	return []string{bridgeEndpointPrefix}
}

func (ep *bridgeEndpoint) Value() []byte {
	b, err := json.Marshal(ep)
	if err != nil {
		return nil
	}
	return b
}

func (ep *bridgeEndpoint) SetValue(value []byte) error {
	return json.Unmarshal(value, ep)
}

func (ep *bridgeEndpoint) Index() uint64 {
	return ep.dbIndex
}

func (ep *bridgeEndpoint) SetIndex(index uint64) {
	ep.dbIndex = index
	ep.dbExists = true
}

func (ep *bridgeEndpoint) Exists() bool {
	return ep.dbExists
}

func (ep *bridgeEndpoint) Skip() bool {
	return false
}

func (ep *bridgeEndpoint) New() datastore.KVObject {
	return &bridgeEndpoint{}
}

func (ep *bridgeEndpoint) CopyTo(o datastore.KVObject) error {
	dstEp := o.(*bridgeEndpoint)
	*dstEp = *ep
	return nil
}

func (ep *bridgeEndpoint) DataScope() string {
	return datastore.LocalScope
}

func (n *bridgeNetwork) restorePortAllocations(ep *bridgeEndpoint) {
	if ep.extConnConfig == nil ||
		ep.extConnConfig.ExposedPorts == nil ||
		ep.extConnConfig.PortBindings == nil {
		return
	}
	tmp := ep.extConnConfig.PortBindings
	ep.extConnConfig.PortBindings = ep.portMapping
	_, err := n.allocatePorts(ep, n.config.DefaultBindingIP, n.driver.config.EnableUserlandProxy)
	if err != nil {
		logrus.Warnf("Failed to reserve existing port mapping for endpoint %s:%v", ep.id[0:7], err)
	}
	ep.extConnConfig.PortBindings = tmp
}

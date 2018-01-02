// +build windows

package windows

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
	windowsPrefix         = "windows"
	windowsEndpointPrefix = "windows-endpoint"
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
			return types.InternalErrorf("windows driver failed to initialize data store: %v", err)
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
	kvol, err := d.store.List(datastore.Key(windowsPrefix), &networkConfiguration{Type: d.name})
	if err != nil && err != datastore.ErrKeyNotFound {
		return fmt.Errorf("failed to get windows network configurations from store: %v", err)
	}

	// It's normal for network configuration state to be empty. Just return.
	if err == datastore.ErrKeyNotFound {
		return nil
	}

	for _, kvo := range kvol {
		ncfg := kvo.(*networkConfiguration)
		if ncfg.Type != d.name {
			continue
		}
		if err = d.createNetwork(ncfg); err != nil {
			logrus.Warnf("could not create windows network for id %s hnsid %s while booting up from persistent state: %v", ncfg.ID, ncfg.HnsID, err)
		}
		logrus.Debugf("Network (%s) restored", ncfg.ID[0:7])
	}

	return nil
}

func (d *driver) populateEndpoints() error {
	kvol, err := d.store.List(datastore.Key(windowsEndpointPrefix), &hnsEndpoint{Type: d.name})
	if err != nil && err != datastore.ErrKeyNotFound {
		return fmt.Errorf("failed to get endpoints from store: %v", err)
	}

	if err == datastore.ErrKeyNotFound {
		return nil
	}

	for _, kvo := range kvol {
		ep := kvo.(*hnsEndpoint)
		if ep.Type != d.name {
			continue
		}
		n, ok := d.networks[ep.nid]
		if !ok {
			logrus.Debugf("Network (%s) not found for restored endpoint (%s)", ep.nid[0:7], ep.id[0:7])
			logrus.Debugf("Deleting stale endpoint (%s) from store", ep.id[0:7])
			if err := d.storeDelete(ep); err != nil {
				logrus.Debugf("Failed to delete stale endpoint (%s) from store", ep.id[0:7])
			}
			continue
		}
		n.endpoints[ep.id] = ep
		logrus.Debugf("Endpoint (%s) restored to network (%s)", ep.id[0:7], ep.nid[0:7])
	}

	return nil
}

func (d *driver) storeUpdate(kvObject datastore.KVObject) error {
	if d.store == nil {
		logrus.Warnf("store not initialized. kv object %s is not added to the store", datastore.Key(kvObject.Key()...))
		return nil
	}

	if err := d.store.PutObjectAtomic(kvObject); err != nil {
		return fmt.Errorf("failed to update store for object type %T: %v", kvObject, err)
	}

	return nil
}

func (d *driver) storeDelete(kvObject datastore.KVObject) error {
	if d.store == nil {
		logrus.Debugf("store not initialized. kv object %s is not deleted from store", datastore.Key(kvObject.Key()...))
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
	nMap["Type"] = ncfg.Type
	nMap["Name"] = ncfg.Name
	nMap["HnsID"] = ncfg.HnsID
	nMap["VLAN"] = ncfg.VLAN
	nMap["VSID"] = ncfg.VSID
	nMap["DNSServers"] = ncfg.DNSServers
	nMap["DNSSuffix"] = ncfg.DNSSuffix
	nMap["SourceMac"] = ncfg.SourceMac
	nMap["NetworkAdapterName"] = ncfg.NetworkAdapterName

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

	ncfg.ID = nMap["ID"].(string)
	ncfg.Type = nMap["Type"].(string)
	ncfg.Name = nMap["Name"].(string)
	ncfg.HnsID = nMap["HnsID"].(string)
	ncfg.VLAN = uint(nMap["VLAN"].(float64))
	ncfg.VSID = uint(nMap["VSID"].(float64))
	ncfg.DNSServers = nMap["DNSServers"].(string)
	ncfg.DNSSuffix = nMap["DNSSuffix"].(string)
	ncfg.SourceMac = nMap["SourceMac"].(string)
	ncfg.NetworkAdapterName = nMap["NetworkAdapterName"].(string)
	return nil
}

func (ncfg *networkConfiguration) Key() []string {
	return []string{windowsPrefix + ncfg.Type, ncfg.ID}
}

func (ncfg *networkConfiguration) KeyPrefix() []string {
	return []string{windowsPrefix + ncfg.Type}
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
	return &networkConfiguration{Type: ncfg.Type}
}

func (ncfg *networkConfiguration) CopyTo(o datastore.KVObject) error {
	dstNcfg := o.(*networkConfiguration)
	*dstNcfg = *ncfg
	return nil
}

func (ncfg *networkConfiguration) DataScope() string {
	return datastore.LocalScope
}

func (ep *hnsEndpoint) MarshalJSON() ([]byte, error) {
	epMap := make(map[string]interface{})
	epMap["id"] = ep.id
	epMap["nid"] = ep.nid
	epMap["Type"] = ep.Type
	epMap["profileID"] = ep.profileID
	epMap["MacAddress"] = ep.macAddress.String()
	if ep.addr.IP != nil {
		epMap["Addr"] = ep.addr.String()
	}
	if ep.gateway != nil {
		epMap["gateway"] = ep.gateway.String()
	}
	epMap["epOption"] = ep.epOption
	epMap["epConnectivity"] = ep.epConnectivity
	epMap["PortMapping"] = ep.portMapping

	return json.Marshal(epMap)
}

func (ep *hnsEndpoint) UnmarshalJSON(b []byte) error {
	var (
		err   error
		epMap map[string]interface{}
	)

	if err = json.Unmarshal(b, &epMap); err != nil {
		return fmt.Errorf("Failed to unmarshal to endpoint: %v", err)
	}
	if v, ok := epMap["MacAddress"]; ok {
		if ep.macAddress, err = net.ParseMAC(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode endpoint MAC address (%s) after json unmarshal: %v", v.(string), err)
		}
	}
	if v, ok := epMap["Addr"]; ok {
		if ep.addr, err = types.ParseCIDR(v.(string)); err != nil {
			return types.InternalErrorf("failed to decode endpoint IPv4 address (%s) after json unmarshal: %v", v.(string), err)
		}
	}
	if v, ok := epMap["gateway"]; ok {
		ep.gateway = net.ParseIP(v.(string))
	}
	ep.id = epMap["id"].(string)
	ep.Type = epMap["Type"].(string)
	ep.nid = epMap["nid"].(string)
	ep.profileID = epMap["profileID"].(string)
	d, _ := json.Marshal(epMap["epOption"])
	if err := json.Unmarshal(d, &ep.epOption); err != nil {
		logrus.Warnf("Failed to decode endpoint container config %v", err)
	}
	d, _ = json.Marshal(epMap["epConnectivity"])
	if err := json.Unmarshal(d, &ep.epConnectivity); err != nil {
		logrus.Warnf("Failed to decode endpoint external connectivity configuration %v", err)
	}
	d, _ = json.Marshal(epMap["PortMapping"])
	if err := json.Unmarshal(d, &ep.portMapping); err != nil {
		logrus.Warnf("Failed to decode endpoint port mapping %v", err)
	}

	return nil
}

func (ep *hnsEndpoint) Key() []string {
	return []string{windowsEndpointPrefix + ep.Type, ep.id}
}

func (ep *hnsEndpoint) KeyPrefix() []string {
	return []string{windowsEndpointPrefix + ep.Type}
}

func (ep *hnsEndpoint) Value() []byte {
	b, err := json.Marshal(ep)
	if err != nil {
		return nil
	}
	return b
}

func (ep *hnsEndpoint) SetValue(value []byte) error {
	return json.Unmarshal(value, ep)
}

func (ep *hnsEndpoint) Index() uint64 {
	return ep.dbIndex
}

func (ep *hnsEndpoint) SetIndex(index uint64) {
	ep.dbIndex = index
	ep.dbExists = true
}

func (ep *hnsEndpoint) Exists() bool {
	return ep.dbExists
}

func (ep *hnsEndpoint) Skip() bool {
	return false
}

func (ep *hnsEndpoint) New() datastore.KVObject {
	return &hnsEndpoint{Type: ep.Type}
}

func (ep *hnsEndpoint) CopyTo(o datastore.KVObject) error {
	dstEp := o.(*hnsEndpoint)
	*dstEp = *ep
	return nil
}

func (ep *hnsEndpoint) DataScope() string {
	return datastore.LocalScope
}

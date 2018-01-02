package libnetwork

import (
	"fmt"
	"strings"

	"github.com/docker/libkv/store/boltdb"
	"github.com/docker/libkv/store/consul"
	"github.com/docker/libkv/store/etcd"
	"github.com/docker/libkv/store/zookeeper"
	"github.com/docker/libnetwork/datastore"
	"github.com/sirupsen/logrus"
)

func registerKVStores() {
	consul.Register()
	zookeeper.Register()
	etcd.Register()
	boltdb.Register()
}

func (c *controller) initScopedStore(scope string, scfg *datastore.ScopeCfg) error {
	store, err := datastore.NewDataStore(scope, scfg)
	if err != nil {
		return err
	}
	c.Lock()
	c.stores = append(c.stores, store)
	c.Unlock()

	return nil
}

func (c *controller) initStores() error {
	registerKVStores()

	c.Lock()
	if c.cfg == nil {
		c.Unlock()
		return nil
	}
	scopeConfigs := c.cfg.Scopes
	c.stores = nil
	c.Unlock()

	for scope, scfg := range scopeConfigs {
		if err := c.initScopedStore(scope, scfg); err != nil {
			return err
		}
	}

	c.startWatch()
	return nil
}

func (c *controller) closeStores() {
	for _, store := range c.getStores() {
		store.Close()
	}
}

func (c *controller) getStore(scope string) datastore.DataStore {
	c.Lock()
	defer c.Unlock()

	for _, store := range c.stores {
		if store.Scope() == scope {
			return store
		}
	}

	return nil
}

func (c *controller) getStores() []datastore.DataStore {
	c.Lock()
	defer c.Unlock()

	return c.stores
}

func (c *controller) getNetworkFromStore(nid string) (*network, error) {
	for _, store := range c.getStores() {
		n := &network{id: nid, ctrlr: c}
		err := store.GetObject(datastore.Key(n.Key()...), n)
		// Continue searching in the next store if the key is not found in this store
		if err != nil {
			if err != datastore.ErrKeyNotFound {
				logrus.Debugf("could not find network %s: %v", nid, err)
			}
			continue
		}

		ec := &endpointCnt{n: n}
		err = store.GetObject(datastore.Key(ec.Key()...), ec)
		if err != nil && !n.inDelete {
			return nil, fmt.Errorf("could not find endpoint count for network %s: %v", n.Name(), err)
		}

		n.epCnt = ec
		if n.scope == "" {
			n.scope = store.Scope()
		}
		return n, nil
	}

	return nil, fmt.Errorf("network %s not found", nid)
}

func (c *controller) getNetworksForScope(scope string) ([]*network, error) {
	var nl []*network

	store := c.getStore(scope)
	if store == nil {
		return nil, nil
	}

	kvol, err := store.List(datastore.Key(datastore.NetworkKeyPrefix),
		&network{ctrlr: c})
	if err != nil && err != datastore.ErrKeyNotFound {
		return nil, fmt.Errorf("failed to get networks for scope %s: %v",
			scope, err)
	}

	for _, kvo := range kvol {
		n := kvo.(*network)
		n.ctrlr = c

		ec := &endpointCnt{n: n}
		err = store.GetObject(datastore.Key(ec.Key()...), ec)
		if err != nil && !n.inDelete {
			logrus.Warnf("Could not find endpoint count key %s for network %s while listing: %v", datastore.Key(ec.Key()...), n.Name(), err)
			continue
		}

		n.epCnt = ec
		if n.scope == "" {
			n.scope = scope
		}
		nl = append(nl, n)
	}

	return nl, nil
}

func (c *controller) getNetworksFromStore() ([]*network, error) {
	var nl []*network

	for _, store := range c.getStores() {
		kvol, err := store.List(datastore.Key(datastore.NetworkKeyPrefix),
			&network{ctrlr: c})
		// Continue searching in the next store if no keys found in this store
		if err != nil {
			if err != datastore.ErrKeyNotFound {
				logrus.Debugf("failed to get networks for scope %s: %v", store.Scope(), err)
			}
			continue
		}

		kvep, err := store.Map(datastore.Key(epCntKeyPrefix), &endpointCnt{})
		if err != nil {
			if err != datastore.ErrKeyNotFound {
				logrus.Warnf("failed to get endpoint_count map for scope %s: %v", store.Scope(), err)
			}
		}

		for _, kvo := range kvol {
			n := kvo.(*network)
			n.Lock()
			n.ctrlr = c
			ec := &endpointCnt{n: n}
			// Trim the leading & trailing "/" to make it consistent across all stores
			if val, ok := kvep[strings.Trim(datastore.Key(ec.Key()...), "/")]; ok {
				ec = val.(*endpointCnt)
				ec.n = n
				n.epCnt = ec
			}
			if n.scope == "" {
				n.scope = store.Scope()
			}
			n.Unlock()
			nl = append(nl, n)
		}
	}

	return nl, nil
}

func (n *network) getEndpointFromStore(eid string) (*endpoint, error) {
	var errors []string
	for _, store := range n.ctrlr.getStores() {
		ep := &endpoint{id: eid, network: n}
		err := store.GetObject(datastore.Key(ep.Key()...), ep)
		// Continue searching in the next store if the key is not found in this store
		if err != nil {
			if err != datastore.ErrKeyNotFound {
				errors = append(errors, fmt.Sprintf("{%s:%v}, ", store.Scope(), err))
				logrus.Debugf("could not find endpoint %s in %s: %v", eid, store.Scope(), err)
			}
			continue
		}
		return ep, nil
	}
	return nil, fmt.Errorf("could not find endpoint %s: %v", eid, errors)
}

func (n *network) getEndpointsFromStore() ([]*endpoint, error) {
	var epl []*endpoint

	tmp := endpoint{network: n}
	for _, store := range n.getController().getStores() {
		kvol, err := store.List(datastore.Key(tmp.KeyPrefix()...), &endpoint{network: n})
		// Continue searching in the next store if no keys found in this store
		if err != nil {
			if err != datastore.ErrKeyNotFound {
				logrus.Debugf("failed to get endpoints for network %s scope %s: %v",
					n.Name(), store.Scope(), err)
			}
			continue
		}

		for _, kvo := range kvol {
			ep := kvo.(*endpoint)
			epl = append(epl, ep)
		}
	}

	return epl, nil
}

func (c *controller) updateToStore(kvObject datastore.KVObject) error {
	cs := c.getStore(kvObject.DataScope())
	if cs == nil {
		return ErrDataStoreNotInitialized(kvObject.DataScope())
	}

	if err := cs.PutObjectAtomic(kvObject); err != nil {
		if err == datastore.ErrKeyModified {
			return err
		}
		return fmt.Errorf("failed to update store for object type %T: %v", kvObject, err)
	}

	return nil
}

func (c *controller) deleteFromStore(kvObject datastore.KVObject) error {
	cs := c.getStore(kvObject.DataScope())
	if cs == nil {
		return ErrDataStoreNotInitialized(kvObject.DataScope())
	}

retry:
	if err := cs.DeleteObjectAtomic(kvObject); err != nil {
		if err == datastore.ErrKeyModified {
			if err := cs.GetObject(datastore.Key(kvObject.Key()...), kvObject); err != nil {
				return fmt.Errorf("could not update the kvobject to latest when trying to delete: %v", err)
			}
			goto retry
		}
		return err
	}

	return nil
}

type netWatch struct {
	localEps  map[string]*endpoint
	remoteEps map[string]*endpoint
	stopCh    chan struct{}
}

func (c *controller) getLocalEps(nw *netWatch) []*endpoint {
	c.Lock()
	defer c.Unlock()

	var epl []*endpoint
	for _, ep := range nw.localEps {
		epl = append(epl, ep)
	}

	return epl
}

func (c *controller) watchSvcRecord(ep *endpoint) {
	c.watchCh <- ep
}

func (c *controller) unWatchSvcRecord(ep *endpoint) {
	c.unWatchCh <- ep
}

func (c *controller) networkWatchLoop(nw *netWatch, ep *endpoint, ecCh <-chan datastore.KVObject) {
	for {
		select {
		case <-nw.stopCh:
			return
		case o := <-ecCh:
			ec := o.(*endpointCnt)

			epl, err := ec.n.getEndpointsFromStore()
			if err != nil {
				break
			}

			c.Lock()
			var addEp []*endpoint

			delEpMap := make(map[string]*endpoint)
			renameEpMap := make(map[string]bool)
			for k, v := range nw.remoteEps {
				delEpMap[k] = v
			}

			for _, lEp := range epl {
				if _, ok := nw.localEps[lEp.ID()]; ok {
					continue
				}

				if ep, ok := nw.remoteEps[lEp.ID()]; ok {
					// On a container rename EP ID will remain
					// the same but the name will change. service
					// records should reflect the change.
					// Keep old EP entry in the delEpMap and add
					// EP from the store (which has the new name)
					// into the new list
					if lEp.name == ep.name {
						delete(delEpMap, lEp.ID())
						continue
					}
					renameEpMap[lEp.ID()] = true
				}
				nw.remoteEps[lEp.ID()] = lEp
				addEp = append(addEp, lEp)
			}

			// EPs whose name are to be deleted from the svc records
			// should also be removed from nw's remote EP list, except
			// the ones that are getting renamed.
			for _, lEp := range delEpMap {
				if !renameEpMap[lEp.ID()] {
					delete(nw.remoteEps, lEp.ID())
				}
			}
			c.Unlock()

			for _, lEp := range delEpMap {
				ep.getNetwork().updateSvcRecord(lEp, c.getLocalEps(nw), false)

			}
			for _, lEp := range addEp {
				ep.getNetwork().updateSvcRecord(lEp, c.getLocalEps(nw), true)
			}
		}
	}
}

func (c *controller) processEndpointCreate(nmap map[string]*netWatch, ep *endpoint) {
	n := ep.getNetwork()
	if !c.isDistributedControl() && n.Scope() == datastore.SwarmScope && n.driverIsMultihost() {
		return
	}

	c.Lock()
	nw, ok := nmap[n.ID()]
	c.Unlock()

	if ok {
		// Update the svc db for the local endpoint join right away
		n.updateSvcRecord(ep, c.getLocalEps(nw), true)

		c.Lock()
		nw.localEps[ep.ID()] = ep

		// If we had learned that from the kv store remove it
		// from remote ep list now that we know that this is
		// indeed a local endpoint
		delete(nw.remoteEps, ep.ID())
		c.Unlock()
		return
	}

	nw = &netWatch{
		localEps:  make(map[string]*endpoint),
		remoteEps: make(map[string]*endpoint),
	}

	// Update the svc db for the local endpoint join right away
	// Do this before adding this ep to localEps so that we don't
	// try to update this ep's container's svc records
	n.updateSvcRecord(ep, c.getLocalEps(nw), true)

	c.Lock()
	nw.localEps[ep.ID()] = ep
	nmap[n.ID()] = nw
	nw.stopCh = make(chan struct{})
	c.Unlock()

	store := c.getStore(n.DataScope())
	if store == nil {
		return
	}

	if !store.Watchable() {
		return
	}

	ch, err := store.Watch(n.getEpCnt(), nw.stopCh)
	if err != nil {
		logrus.Warnf("Error creating watch for network: %v", err)
		return
	}

	go c.networkWatchLoop(nw, ep, ch)
}

func (c *controller) processEndpointDelete(nmap map[string]*netWatch, ep *endpoint) {
	n := ep.getNetwork()
	if !c.isDistributedControl() && n.Scope() == datastore.SwarmScope && n.driverIsMultihost() {
		return
	}

	c.Lock()
	nw, ok := nmap[n.ID()]

	if ok {
		delete(nw.localEps, ep.ID())
		c.Unlock()

		// Update the svc db about local endpoint leave right away
		// Do this after we remove this ep from localEps so that we
		// don't try to remove this svc record from this ep's container.
		n.updateSvcRecord(ep, c.getLocalEps(nw), false)

		c.Lock()
		if len(nw.localEps) == 0 {
			close(nw.stopCh)

			// This is the last container going away for the network. Destroy
			// this network's svc db entry
			delete(c.svcRecords, n.ID())

			delete(nmap, n.ID())
		}
	}
	c.Unlock()
}

func (c *controller) watchLoop() {
	for {
		select {
		case ep := <-c.watchCh:
			c.processEndpointCreate(c.nmap, ep)
		case ep := <-c.unWatchCh:
			c.processEndpointDelete(c.nmap, ep)
		}
	}
}

func (c *controller) startWatch() {
	if c.watchCh != nil {
		return
	}
	c.watchCh = make(chan *endpoint)
	c.unWatchCh = make(chan *endpoint)
	c.nmap = make(map[string]*netWatch)

	go c.watchLoop()
}

func (c *controller) networkCleanup() {
	networks, err := c.getNetworksFromStore()
	if err != nil {
		logrus.Warnf("Could not retrieve networks from store(s) during network cleanup: %v", err)
		return
	}

	for _, n := range networks {
		if n.inDelete {
			logrus.Infof("Removing stale network %s (%s)", n.Name(), n.ID())
			if err := n.delete(true); err != nil {
				logrus.Debugf("Error while removing stale network: %v", err)
			}
		}
	}
}

var populateSpecial NetworkWalker = func(nw Network) bool {
	if n := nw.(*network); n.hasSpecialDriver() && !n.ConfigOnly() {
		if err := n.getController().addNetwork(n); err != nil {
			logrus.Warnf("Failed to populate network %q with driver %q", nw.Name(), nw.Type())
		}
	}
	return false
}

package libnetwork

import (
	"container/heap"
	"encoding/json"
	"sync"

	"github.com/docker/libnetwork/datastore"
	"github.com/docker/libnetwork/osl"
	"github.com/sirupsen/logrus"
)

const (
	sandboxPrefix = "sandbox"
)

type epState struct {
	Eid string
	Nid string
}

type sbState struct {
	ID         string
	Cid        string
	c          *controller
	dbIndex    uint64
	dbExists   bool
	Eps        []epState
	EpPriority map[string]int
	// external servers have to be persisted so that on restart of a live-restore
	// enabled daemon we get the external servers for the running containers.
	// We have two versions of ExtDNS to support upgrade & downgrade of the daemon
	// between >=1.14 and <1.14 versions.
	ExtDNS  []string
	ExtDNS2 []extDNSEntry
}

func (sbs *sbState) Key() []string {
	return []string{sandboxPrefix, sbs.ID}
}

func (sbs *sbState) KeyPrefix() []string {
	return []string{sandboxPrefix}
}

func (sbs *sbState) Value() []byte {
	b, err := json.Marshal(sbs)
	if err != nil {
		return nil
	}
	return b
}

func (sbs *sbState) SetValue(value []byte) error {
	return json.Unmarshal(value, sbs)
}

func (sbs *sbState) Index() uint64 {
	sbi, err := sbs.c.SandboxByID(sbs.ID)
	if err != nil {
		return sbs.dbIndex
	}

	sb := sbi.(*sandbox)
	maxIndex := sb.dbIndex
	if sbs.dbIndex > maxIndex {
		maxIndex = sbs.dbIndex
	}

	return maxIndex
}

func (sbs *sbState) SetIndex(index uint64) {
	sbs.dbIndex = index
	sbs.dbExists = true

	sbi, err := sbs.c.SandboxByID(sbs.ID)
	if err != nil {
		return
	}

	sb := sbi.(*sandbox)
	sb.dbIndex = index
	sb.dbExists = true
}

func (sbs *sbState) Exists() bool {
	if sbs.dbExists {
		return sbs.dbExists
	}

	sbi, err := sbs.c.SandboxByID(sbs.ID)
	if err != nil {
		return false
	}

	sb := sbi.(*sandbox)
	return sb.dbExists
}

func (sbs *sbState) Skip() bool {
	return false
}

func (sbs *sbState) New() datastore.KVObject {
	return &sbState{c: sbs.c}
}

func (sbs *sbState) CopyTo(o datastore.KVObject) error {
	dstSbs := o.(*sbState)
	dstSbs.c = sbs.c
	dstSbs.ID = sbs.ID
	dstSbs.Cid = sbs.Cid
	dstSbs.dbIndex = sbs.dbIndex
	dstSbs.dbExists = sbs.dbExists
	dstSbs.EpPriority = sbs.EpPriority

	dstSbs.Eps = append(dstSbs.Eps, sbs.Eps...)

	if len(sbs.ExtDNS2) > 0 {
		for _, dns := range sbs.ExtDNS2 {
			dstSbs.ExtDNS2 = append(dstSbs.ExtDNS2, dns)
			dstSbs.ExtDNS = append(dstSbs.ExtDNS, dns.IPStr)
		}
		return nil
	}
	for _, dns := range sbs.ExtDNS {
		dstSbs.ExtDNS = append(dstSbs.ExtDNS, dns)
		dstSbs.ExtDNS2 = append(dstSbs.ExtDNS2, extDNSEntry{IPStr: dns})
	}

	return nil
}

func (sbs *sbState) DataScope() string {
	return datastore.LocalScope
}

func (sb *sandbox) storeUpdate() error {
	sbs := &sbState{
		c:          sb.controller,
		ID:         sb.id,
		Cid:        sb.containerID,
		EpPriority: sb.epPriority,
		ExtDNS2:    sb.extDNS,
	}

	for _, ext := range sb.extDNS {
		sbs.ExtDNS = append(sbs.ExtDNS, ext.IPStr)
	}

retry:
	sbs.Eps = nil
	for _, ep := range sb.getConnectedEndpoints() {
		// If the endpoint is not persisted then do not add it to
		// the sandbox checkpoint
		if ep.Skip() {
			continue
		}

		eps := epState{
			Nid: ep.getNetwork().ID(),
			Eid: ep.ID(),
		}

		sbs.Eps = append(sbs.Eps, eps)
	}

	err := sb.controller.updateToStore(sbs)
	if err == datastore.ErrKeyModified {
		// When we get ErrKeyModified it is sufficient to just
		// go back and retry.  No need to get the object from
		// the store because we always regenerate the store
		// state from in memory sandbox state
		goto retry
	}

	return err
}

func (sb *sandbox) storeDelete() error {
	sbs := &sbState{
		c:        sb.controller,
		ID:       sb.id,
		Cid:      sb.containerID,
		dbIndex:  sb.dbIndex,
		dbExists: sb.dbExists,
	}

	return sb.controller.deleteFromStore(sbs)
}

func (c *controller) sandboxCleanup(activeSandboxes map[string]interface{}) {
	store := c.getStore(datastore.LocalScope)
	if store == nil {
		logrus.Error("Could not find local scope store while trying to cleanup sandboxes")
		return
	}

	kvol, err := store.List(datastore.Key(sandboxPrefix), &sbState{c: c})
	if err != nil && err != datastore.ErrKeyNotFound {
		logrus.Errorf("failed to get sandboxes for scope %s: %v", store.Scope(), err)
		return
	}

	// It's normal for no sandboxes to be found. Just bail out.
	if err == datastore.ErrKeyNotFound {
		return
	}

	for _, kvo := range kvol {
		sbs := kvo.(*sbState)

		sb := &sandbox{
			id:                 sbs.ID,
			controller:         sbs.c,
			containerID:        sbs.Cid,
			endpoints:          epHeap{},
			populatedEndpoints: map[string]struct{}{},
			dbIndex:            sbs.dbIndex,
			isStub:             true,
			dbExists:           true,
		}
		// If we are restoring from a older version extDNSEntry won't have the
		// HostLoopback field
		if len(sbs.ExtDNS2) > 0 {
			sb.extDNS = sbs.ExtDNS2
		} else {
			for _, dns := range sbs.ExtDNS {
				sb.extDNS = append(sb.extDNS, extDNSEntry{IPStr: dns})
			}
		}

		msg := " for cleanup"
		create := true
		isRestore := false
		if val, ok := activeSandboxes[sb.ID()]; ok {
			msg = ""
			sb.isStub = false
			isRestore = true
			opts := val.([]SandboxOption)
			sb.processOptions(opts...)
			sb.restorePath()
			create = !sb.config.useDefaultSandBox
			heap.Init(&sb.endpoints)
		}
		sb.osSbox, err = osl.NewSandbox(sb.Key(), create, isRestore)
		if err != nil {
			logrus.Errorf("failed to create osl sandbox while trying to restore sandbox %s%s: %v", sb.ID()[0:7], msg, err)
			continue
		}

		c.Lock()
		c.sandboxes[sb.id] = sb
		c.Unlock()

		for _, eps := range sbs.Eps {
			n, err := c.getNetworkFromStore(eps.Nid)
			var ep *endpoint
			if err != nil {
				logrus.Errorf("getNetworkFromStore for nid %s failed while trying to build sandbox for cleanup: %v", eps.Nid, err)
				n = &network{id: eps.Nid, ctrlr: c, drvOnce: &sync.Once{}, persist: true}
				ep = &endpoint{id: eps.Eid, network: n, sandboxID: sbs.ID}
			} else {
				ep, err = n.getEndpointFromStore(eps.Eid)
				if err != nil {
					logrus.Errorf("getEndpointFromStore for eid %s failed while trying to build sandbox for cleanup: %v", eps.Eid, err)
					ep = &endpoint{id: eps.Eid, network: n, sandboxID: sbs.ID}
				}
			}
			if _, ok := activeSandboxes[sb.ID()]; ok && err != nil {
				logrus.Errorf("failed to restore endpoint %s in %s for container %s due to %v", eps.Eid, eps.Nid, sb.ContainerID(), err)
				continue
			}
			heap.Push(&sb.endpoints, ep)
		}

		if _, ok := activeSandboxes[sb.ID()]; !ok {
			logrus.Infof("Removing stale sandbox %s (%s)", sb.id, sb.containerID)
			if err := sb.delete(true); err != nil {
				logrus.Errorf("Failed to delete sandbox %s while trying to cleanup: %v", sb.id, err)
			}
			continue
		}

		// reconstruct osl sandbox field
		if !sb.config.useDefaultSandBox {
			if err := sb.restoreOslSandbox(); err != nil {
				logrus.Errorf("failed to populate fields for osl sandbox %s", sb.ID())
				continue
			}
		} else {
			c.sboxOnce.Do(func() {
				c.defOsSbox = sb.osSbox
			})
		}

		for _, ep := range sb.endpoints {
			// Watch for service records
			if !c.isAgent() {
				c.watchSvcRecord(ep)
			}
		}
	}
}

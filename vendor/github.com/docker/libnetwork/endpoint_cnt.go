package libnetwork

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/docker/libnetwork/datastore"
)

type endpointCnt struct {
	n        *network
	Count    uint64
	dbIndex  uint64
	dbExists bool
	sync.Mutex
}

const epCntKeyPrefix = "endpoint_count"

func (ec *endpointCnt) Key() []string {
	ec.Lock()
	defer ec.Unlock()

	return []string{epCntKeyPrefix, ec.n.id}
}

func (ec *endpointCnt) KeyPrefix() []string {
	ec.Lock()
	defer ec.Unlock()

	return []string{epCntKeyPrefix, ec.n.id}
}

func (ec *endpointCnt) Value() []byte {
	ec.Lock()
	defer ec.Unlock()

	b, err := json.Marshal(ec)
	if err != nil {
		return nil
	}
	return b
}

func (ec *endpointCnt) SetValue(value []byte) error {
	ec.Lock()
	defer ec.Unlock()

	return json.Unmarshal(value, &ec)
}

func (ec *endpointCnt) Index() uint64 {
	ec.Lock()
	defer ec.Unlock()
	return ec.dbIndex
}

func (ec *endpointCnt) SetIndex(index uint64) {
	ec.Lock()
	ec.dbIndex = index
	ec.dbExists = true
	ec.Unlock()
}

func (ec *endpointCnt) Exists() bool {
	ec.Lock()
	defer ec.Unlock()
	return ec.dbExists
}

func (ec *endpointCnt) Skip() bool {
	ec.Lock()
	defer ec.Unlock()
	return !ec.n.persist
}

func (ec *endpointCnt) New() datastore.KVObject {
	ec.Lock()
	defer ec.Unlock()

	return &endpointCnt{
		n: ec.n,
	}
}

func (ec *endpointCnt) CopyTo(o datastore.KVObject) error {
	ec.Lock()
	defer ec.Unlock()

	dstEc := o.(*endpointCnt)
	dstEc.n = ec.n
	dstEc.Count = ec.Count
	dstEc.dbExists = ec.dbExists
	dstEc.dbIndex = ec.dbIndex

	return nil
}

func (ec *endpointCnt) DataScope() string {
	return ec.n.DataScope()
}

func (ec *endpointCnt) EndpointCnt() uint64 {
	ec.Lock()
	defer ec.Unlock()

	return ec.Count
}

func (ec *endpointCnt) updateStore() error {
	store := ec.n.getController().getStore(ec.DataScope())
	if store == nil {
		return fmt.Errorf("store not found for scope %s on endpoint count update", ec.DataScope())
	}
	// make a copy of count and n to avoid being overwritten by store.GetObject
	count := ec.EndpointCnt()
	n := ec.n
	for {
		if err := ec.n.getController().updateToStore(ec); err == nil || err != datastore.ErrKeyModified {
			return err
		}
		if err := store.GetObject(datastore.Key(ec.Key()...), ec); err != nil {
			return fmt.Errorf("could not update the kvobject to latest on endpoint count update: %v", err)
		}
		ec.Lock()
		ec.Count = count
		ec.n = n
		ec.Unlock()
	}
}

func (ec *endpointCnt) setCnt(cnt uint64) error {
	ec.Lock()
	ec.Count = cnt
	ec.Unlock()
	return ec.updateStore()
}

func (ec *endpointCnt) atomicIncDecEpCnt(inc bool) error {
retry:
	ec.Lock()
	if inc {
		ec.Count++
	} else {
		if ec.Count > 0 {
			ec.Count--
		}
	}
	ec.Unlock()

	store := ec.n.getController().getStore(ec.DataScope())
	if store == nil {
		return fmt.Errorf("store not found for scope %s", ec.DataScope())
	}

	if err := ec.n.getController().updateToStore(ec); err != nil {
		if err == datastore.ErrKeyModified {
			if err := store.GetObject(datastore.Key(ec.Key()...), ec); err != nil {
				return fmt.Errorf("could not update the kvobject to latest when trying to atomic add endpoint count: %v", err)
			}

			goto retry
		}

		return err
	}

	return nil
}

func (ec *endpointCnt) IncEndpointCnt() error {
	return ec.atomicIncDecEpCnt(true)
}

func (ec *endpointCnt) DecEndpointCnt() error {
	return ec.atomicIncDecEpCnt(false)
}

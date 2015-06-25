package scheduler

import (
	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/upid"
	"sync"
)

type cachedOffer struct {
	offer    *mesos.Offer
	slavePid *upid.UPID
}

func newCachedOffer(offer *mesos.Offer, slavePid *upid.UPID) *cachedOffer {
	return &cachedOffer{offer: offer, slavePid: slavePid}
}

// schedCache a managed cache with backing maps to store offeres
// and tasked slaves.
type schedCache struct {
	lock           sync.RWMutex
	savedOffers    map[string]*cachedOffer // current offers key:OfferID
	savedSlavePids map[string]*upid.UPID   // Current saved slaves, key:slaveId
}

func newSchedCache() *schedCache {
	return &schedCache{
		savedOffers:    make(map[string]*cachedOffer),
		savedSlavePids: make(map[string]*upid.UPID),
	}
}

// putOffer stores an offer and the slavePID associated with offer.
func (cache *schedCache) putOffer(offer *mesos.Offer, pid *upid.UPID) {
	if offer == nil || pid == nil {
		log.V(3).Infoln("WARN: Offer not cached. The offer or pid cannot be nil")
		return
	}
	log.V(3).Infoln("Caching offer ", offer.Id.GetValue(), " with slavePID ", pid.String())
	cache.lock.Lock()
	cache.savedOffers[offer.Id.GetValue()] = &cachedOffer{offer: offer, slavePid: pid}
	cache.lock.Unlock()
}

// getOffer returns cached offer
func (cache *schedCache) getOffer(offerId *mesos.OfferID) *cachedOffer {
	if offerId == nil {
		log.V(3).Infoln("WARN: OfferId == nil, returning nil")
		return nil
	}
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	return cache.savedOffers[offerId.GetValue()]
}

// containsOff test cache for offer(offerId)
func (cache *schedCache) containsOffer(offerId *mesos.OfferID) bool {
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	_, ok := cache.savedOffers[offerId.GetValue()]
	return ok
}

func (cache *schedCache) removeOffer(offerId *mesos.OfferID) {
	cache.lock.Lock()
	delete(cache.savedOffers, offerId.GetValue())
	cache.lock.Unlock()
}

func (cache *schedCache) putSlavePid(slaveId *mesos.SlaveID, pid *upid.UPID) {
	cache.lock.Lock()
	cache.savedSlavePids[slaveId.GetValue()] = pid
	cache.lock.Unlock()
}

func (cache *schedCache) getSlavePid(slaveId *mesos.SlaveID) *upid.UPID {
	if slaveId == nil {
		log.V(3).Infoln("SlaveId == nil, returning empty UPID")
		return nil
	}
	return cache.savedSlavePids[slaveId.GetValue()]
}

func (cache *schedCache) containsSlavePid(slaveId *mesos.SlaveID) bool {
	cache.lock.RLock()
	defer cache.lock.RUnlock()
	_, ok := cache.savedSlavePids[slaveId.GetValue()]
	return ok
}

func (cache *schedCache) removeSlavePid(slaveId *mesos.SlaveID) {
	cache.lock.Lock()
	delete(cache.savedSlavePids, slaveId.GetValue())
	cache.lock.Unlock()
}

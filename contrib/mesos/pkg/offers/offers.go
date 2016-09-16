/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package offers

import (
	"fmt"
	"reflect"
	"sync"
	"sync/atomic"
	"time"

	log "github.com/golang/glog"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers/metrics"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	"k8s.io/kubernetes/contrib/mesos/pkg/queue"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	offerListenerMaxAge      = 12              // max number of times we'll attempt to fit an offer to a listener before requiring them to re-register themselves
	offerIdCacheTTL          = 1 * time.Second // determines expiration of cached offer ids, used in listener notification
	deferredDeclineTtlFactor = 2               // this factor, multiplied by the offer ttl, determines how long to wait before attempting to decline previously claimed offers that were subsequently deleted, then released. see offerStorage.Delete
	notifyListenersDelay     = 0               // delay between offer listener notification attempts
)

type Filter func(*mesos.Offer) bool

type Registry interface {
	// Initialize the instance, spawning necessary housekeeping go routines.
	Init(<-chan struct{})

	// Add offers to this registry, rejecting those that are deemed incompatible.
	Add([]*mesos.Offer)

	// Listen for arriving offers that are acceptable to the filter, sending
	// a signal on (by closing) the returned channel. A listener will only
	// ever be notified once, if at all.
	Listen(id string, f Filter) <-chan struct{}

	// invoked when offers are rescinded or expired
	Delete(string, metrics.OfferDeclinedReason)

	// when true, returns the offer that's registered for the given ID
	Get(offerId string) (Perishable, bool)

	// iterate through non-expired offers in this registry
	Walk(Walker) error

	// invalidate one or all (when offerId="") offers; offers are not declined,
	// but are simply flagged as expired in the offer history
	Invalidate(offerId string)

	// invalidate all offers associated with the slave identified by slaveId.
	InvalidateForSlave(slaveId string)
}

// callback that is invoked during a walk through a series of live offers,
// returning with stop=true (or err != nil) if the walk should stop prematurely.
type Walker func(offer Perishable) (stop bool, err error)

type RegistryConfig struct {
	DeclineOffer  func(offerId string) <-chan error // tell Mesos that we're declining the offer
	Compat        func(*mesos.Offer) bool           // returns true if offer is compatible; incompatible offers are declined
	TTL           time.Duration                     // determines a perishable offer's expiration deadline: now+ttl
	LingerTTL     time.Duration                     // if zero, offers will not linger in the FIFO past their expiration deadline
	ListenerDelay time.Duration                     // specifies the sleep time between offer listener notifications
}

type offerStorage struct {
	RegistryConfig
	offers    *cache.FIFO       // collection of Perishable, both live and expired
	listeners *queue.DelayFIFO  // collection of *offerListener
	delayed   *queue.DelayQueue // deadline-oriented offer-event queue
	slaves    *slaveStorage     // slave to offer mappings
}

type liveOffer struct {
	*mesos.Offer
	expiration time.Time
	acquired   int32 // 1 = acquired, 0 = free
}

type expiredOffer struct {
	offerSpec
	deadline time.Time
}

// subset of mesos.OfferInfo useful for recordkeeping
type offerSpec struct {
	id       string
	hostname string
}

// offers that may perish (all of them?) implement this interface.
// callers may expect to access these funcs concurrently so implementations
// must provide their own form of synchronization around mutable state.
type Perishable interface {
	// returns true if this offer has expired
	HasExpired() bool
	// if not yet expired, return mesos offer details; otherwise nil
	Details() *mesos.Offer
	// mark this offer as acquired, returning true if it was previously unacquired. thread-safe.
	Acquire() bool
	// mark this offer as un-acquired. thread-safe.
	Release()
	// expire or delete this offer from storage
	age(s *offerStorage)
	// return a unique identifier for this offer
	Id() string
	// return the slave host for this offer
	Host() string
	addTo(*queue.DelayQueue)
}

func (e *expiredOffer) addTo(q *queue.DelayQueue) {
	q.Add(e)
}

func (e *expiredOffer) Id() string {
	return e.id
}

func (e *expiredOffer) Host() string {
	return e.hostname
}

func (e *expiredOffer) HasExpired() bool {
	return true
}

func (e *expiredOffer) Details() *mesos.Offer {
	return nil
}

func (e *expiredOffer) Acquire() bool {
	return false
}

func (e *expiredOffer) Release() {}

func (e *expiredOffer) age(s *offerStorage) {
	log.V(3).Infof("Delete lingering offer: %v", e.id)
	s.offers.Delete(e)
	s.slaves.deleteOffer(e.id)
}

// return the time left to linger
func (e *expiredOffer) GetDelay() time.Duration {
	return e.deadline.Sub(time.Now())
}

func (to *liveOffer) HasExpired() bool {
	return time.Now().After(to.expiration)
}

func (to *liveOffer) Details() *mesos.Offer {
	return to.Offer
}

func (to *liveOffer) Acquire() (acquired bool) {
	if acquired = atomic.CompareAndSwapInt32(&to.acquired, 0, 1); acquired {
		metrics.OffersAcquired.WithLabelValues(to.Host()).Inc()
	}
	return
}

func (to *liveOffer) Release() {
	if released := atomic.CompareAndSwapInt32(&to.acquired, 1, 0); released {
		metrics.OffersReleased.WithLabelValues(to.Host()).Inc()
	}
}

func (to *liveOffer) age(s *offerStorage) {
	s.Delete(to.Id(), metrics.OfferExpired)
}

func (to *liveOffer) Id() string {
	return to.Offer.Id.GetValue()
}

func (to *liveOffer) Host() string {
	return to.Offer.GetHostname()
}

func (to *liveOffer) addTo(q *queue.DelayQueue) {
	q.Add(to)
}

// return the time remaining before the offer expires
func (to *liveOffer) GetDelay() time.Duration {
	return to.expiration.Sub(time.Now())
}

func CreateRegistry(c RegistryConfig) Registry {
	metrics.Register()
	return &offerStorage{
		RegistryConfig: c,
		offers: cache.NewFIFO(cache.KeyFunc(func(v interface{}) (string, error) {
			if perishable, ok := v.(Perishable); !ok {
				return "", fmt.Errorf("expected perishable offer, not '%+v'", v)
			} else {
				return perishable.Id(), nil
			}
		})),
		listeners: queue.NewDelayFIFO(),
		delayed:   queue.NewDelayQueue(),
		slaves:    newSlaveStorage(),
	}
}

func (s *offerStorage) declineOffer(offerId, hostname string, reason metrics.OfferDeclinedReason) {
	//TODO(jdef) might be nice to spec an abort chan here
	runtime.Signal(proc.OnError(s.DeclineOffer(offerId), func(err error) {
		log.Warningf("decline failed for offer id %v: %v", offerId, err)
	}, nil)).Then(func() {
		metrics.OffersDeclined.WithLabelValues(hostname, string(reason)).Inc()
	})
}

func (s *offerStorage) Add(offers []*mesos.Offer) {
	now := time.Now()
	for _, offer := range offers {
		if !s.Compat(offer) {
			//TODO(jdef) would be nice to batch these up
			offerId := offer.Id.GetValue()
			log.V(3).Infof("Declining incompatible offer %v", offerId)
			s.declineOffer(offerId, offer.GetHostname(), metrics.OfferCompat)
			continue
		}
		timed := &liveOffer{
			Offer:      offer,
			expiration: now.Add(s.TTL),
			acquired:   0,
		}
		log.V(3).Infof("Receiving offer %v", timed.Id())
		s.offers.Add(timed)
		s.delayed.Add(timed)
		s.slaves.add(offer.SlaveId.GetValue(), timed.Id())
		metrics.OffersReceived.WithLabelValues(timed.Host()).Inc()
	}
}

// delete an offer from storage, implicitly expires the offer
func (s *offerStorage) Delete(offerId string, reason metrics.OfferDeclinedReason) {
	if offer, ok := s.Get(offerId); ok {
		log.V(3).Infof("Deleting offer %v", offerId)
		// attempt to block others from consuming the offer. if it's already been
		// claimed and is not yet lingering then don't decline it - just mark it as
		// expired in the history: allow a prior claimant to attempt to launch with it
		notYetClaimed := offer.Acquire()
		if offer.Details() != nil {
			if notYetClaimed {
				log.V(3).Infof("Declining offer %v", offerId)
				s.declineOffer(offerId, offer.Host(), reason)
			} else {
				// some pod has acquired this and may attempt to launch a task with it
				// failed schedule/launch attempts are required to Release() any claims on the offer

				// TODO(jdef): not sure what a good value is here. the goal is to provide a
				// launchTasks (driver) operation enough time to complete so that we don't end
				// up declining an offer that we're actually attempting to use.
				time.AfterFunc(deferredDeclineTtlFactor*s.TTL, func() {
					// at this point the offer is in one of five states:
					// a) permanently deleted: expired due to timeout
					// b) permanently deleted: expired due to having been rescinded
					// c) lingering: expired due to timeout
					// d) lingering: expired due to having been rescinded
					// e) claimed: task launched and it using resources from this offer
					// we want to **avoid** declining an offer that's claimed: attempt to acquire
					if offer.Acquire() {
						// previously claimed offer was released, perhaps due to a launch
						// failure, so we should attempt to decline
						log.V(3).Infof("attempting to decline (previously claimed) offer %v", offerId)
						s.declineOffer(offerId, offer.Host(), reason)
					}
				})
			}
		}
		s.expireOffer(offer)
	} // else, ignore offers not in the history
}

func (s *offerStorage) InvalidateForSlave(slaveId string) {
	offerIds := s.slaves.deleteSlave(slaveId)
	for oid := range offerIds {
		s.invalidateOne(oid)
	}
}

// if offerId == "" then expire all known, live offers, otherwise only the offer indicated
func (s *offerStorage) Invalidate(offerId string) {
	if offerId != "" {
		s.invalidateOne(offerId)
		return
	}
	obj := s.offers.List()
	for _, o := range obj {
		offer, ok := o.(Perishable)
		if !ok {
			log.Errorf("Expected perishable offer, not %v", o)
			continue
		}
		offer.Acquire() // attempt to block others from using it
		s.expireOffer(offer)
		// don't decline, we already know that it's an invalid offer
	}
}

func (s *offerStorage) invalidateOne(offerId string) {
	if offer, ok := s.Get(offerId); ok {
		offer.Acquire() // attempt to block others from using it
		s.expireOffer(offer)
		// don't decline, we already know that it's an invalid offer
	}
}

// Walk the collection of offers. The walk stops either as indicated by the
// Walker or when the end of the offer list is reached. Expired offers are
// never passed to a Walker.
func (s *offerStorage) Walk(w Walker) error {
	for _, v := range s.offers.List() {
		offer, ok := v.(Perishable)
		if !ok {
			// offer disappeared...
			continue
		}
		if offer.HasExpired() {
			// never pass expired offers to walkers
			continue
		}
		if stop, err := w(offer); err != nil {
			return err
		} else if stop {
			return nil
		}
	}
	return nil
}

func Expired(offerId, hostname string, ttl time.Duration) *expiredOffer {
	return &expiredOffer{offerSpec{id: offerId, hostname: hostname}, time.Now().Add(ttl)}
}

func (s *offerStorage) expireOffer(offer Perishable) {
	// the offer may or may not be expired due to TTL so check for details
	// since that's a more reliable determinant of lingering status
	if details := offer.Details(); details != nil {
		// recently expired, should linger
		offerId := details.Id.GetValue()
		log.V(3).Infof("Expiring offer %v", offerId)
		if s.LingerTTL > 0 {
			log.V(3).Infof("offer will linger: %v", offerId)
			expired := Expired(offerId, offer.Host(), s.LingerTTL)
			s.offers.Update(expired)
			s.delayed.Add(expired)
		} else {
			log.V(3).Infof("Permanently deleting offer %v", offerId)
			s.offers.Delete(offerId)
			s.slaves.deleteOffer(offerId)
		}
	} // else, it's still lingering...
}

func (s *offerStorage) Get(id string) (Perishable, bool) {
	if obj, ok, _ := s.offers.GetByKey(id); !ok {
		return nil, false
	} else {
		to, ok := obj.(Perishable)
		if !ok {
			log.Errorf("invalid offer object in fifo '%v'", obj)
		}
		return to, ok
	}
}

type offerListener struct {
	id         string
	accepts    Filter
	notify     chan<- struct{}
	age        int
	deadline   time.Time
	sawVersion uint64
}

func (l *offerListener) GetUID() string {
	return l.id
}

func (l *offerListener) Deadline() (time.Time, bool) {
	return l.deadline, true
}

// register a listener for new offers, whom we'll notify upon receiving such.
// notification is delivered in the form of closing the channel, nothing is ever sent.
func (s *offerStorage) Listen(id string, f Filter) <-chan struct{} {
	if f == nil {
		return nil
	}
	ch := make(chan struct{})
	listen := &offerListener{
		id:       id,
		accepts:  f,
		notify:   ch,
		deadline: time.Now().Add(s.ListenerDelay),
	}
	log.V(3).Infof("Registering offer listener %s", listen.id)
	s.listeners.Offer(listen, queue.ReplaceExisting)
	return ch
}

func (s *offerStorage) ageOffers() {
	offer, ok := s.delayed.Pop().(Perishable)
	if !ok {
		log.Errorf("Expected Perishable, not %v", offer)
		return
	}
	if details := offer.Details(); details != nil && !offer.HasExpired() {
		// live offer has not expired yet: timed out early
		// FWIW: early timeouts are more frequent when GOMAXPROCS is > 1
		offer.addTo(s.delayed)
	} else {
		offer.age(s)
	}
}

func (s *offerStorage) nextListener() *offerListener {
	obj := s.listeners.Pop(queue.WithoutCancel())
	if listen, ok := obj.(*offerListener); !ok {
		//programming error
		panic(fmt.Sprintf("unexpected listener object %v", obj))
	} else {
		return listen
	}
}

// notify listeners if we find an acceptable offer for them. listeners
// are garbage collected after a certain age (see offerListenerMaxAge).
// ids lists offer IDs that are retrievable from offer storage.
func (s *offerStorage) notifyListeners(ids func() (sets.String, uint64)) {
	listener := s.nextListener() // blocking

	offerIds, version := ids()
	if listener.sawVersion == version {
		// no changes to offer list, avoid growing older - just wait for new offers to arrive
		listener.deadline = time.Now().Add(s.ListenerDelay)
		s.listeners.Offer(listener, queue.KeepExisting)
		return
	}
	listener.sawVersion = version

	// notify if we find an acceptable offer
	for id := range offerIds {
		if offer, ok := s.Get(id); !ok || offer.HasExpired() {
			continue
		} else if listener.accepts(offer.Details()) {
			log.V(3).Infof("Notifying offer listener %s", listener.id)
			close(listener.notify)
			return
		}
	}

	// no interesting offers found, re-queue the listener
	listener.age++
	if listener.age < offerListenerMaxAge {
		listener.deadline = time.Now().Add(s.ListenerDelay)
		s.listeners.Offer(listener, queue.KeepExisting)
	} else {
		// garbage collection is as simple as not re-adding the listener to the queue
		log.V(3).Infof("garbage collecting offer listener %s", listener.id)
	}
}

func (s *offerStorage) Init(done <-chan struct{}) {
	// zero delay, reap offers as soon as they expire
	go runtime.Until(s.ageOffers, 0, done)

	// cached offer ids for the purposes of listener notification
	idCache := &stringsCache{
		refill: func() sets.String {
			result := sets.NewString()
			for _, v := range s.offers.List() {
				if offer, ok := v.(Perishable); ok {
					result.Insert(offer.Id())
				}
			}
			return result
		},
		ttl: offerIdCacheTTL,
	}

	go runtime.Until(func() { s.notifyListeners(idCache.Strings) }, notifyListenersDelay, done)
}

type stringsCache struct {
	expiresAt time.Time
	cached    sets.String
	ttl       time.Duration
	refill    func() sets.String
	version   uint64
}

// not thread-safe
func (c *stringsCache) Strings() (sets.String, uint64) {
	now := time.Now()
	if c.expiresAt.Before(now) {
		old := c.cached
		c.cached = c.refill()
		c.expiresAt = now.Add(c.ttl)
		if !reflect.DeepEqual(old, c.cached) {
			c.version++
		}
	}
	return c.cached, c.version
}

type slaveStorage struct {
	sync.Mutex
	index map[string]string // map offerId to slaveId
}

func newSlaveStorage() *slaveStorage {
	return &slaveStorage{
		index: make(map[string]string),
	}
}

// create a mapping between a slave and an offer
func (self *slaveStorage) add(slaveId, offerId string) {
	self.Lock()
	defer self.Unlock()
	self.index[offerId] = slaveId
}

// delete the slave-offer mappings for slaveId, returns the IDs of the offers that were unmapped
func (self *slaveStorage) deleteSlave(slaveId string) sets.String {
	offerIds := sets.NewString()
	self.Lock()
	defer self.Unlock()
	for oid, sid := range self.index {
		if sid == slaveId {
			offerIds.Insert(oid)
			delete(self.index, oid)
		}
	}
	return offerIds
}

// delete the slave-offer mappings for offerId
func (self *slaveStorage) deleteOffer(offerId string) {
	self.Lock()
	defer self.Unlock()
	delete(self.index, offerId)
}

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
	"errors"
	"sync/atomic"
	"testing"
	"time"

	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
)

func TestExpiredOffer(t *testing.T) {
	t.Parallel()

	ttl := 2 * time.Second
	o := Expired("test", "testhost", ttl)

	if o.Id() != "test" {
		t.Error("expiredOffer does not return its Id")
	}
	if o.Host() != "testhost" {
		t.Error("expiredOffer does not return its hostname")
	}
	if o.HasExpired() != true {
		t.Error("expiredOffer is not expired")
	}
	if o.Details() != nil {
		t.Error("expiredOffer does not return nil Details")
	}
	if o.Acquire() != false {
		t.Error("expiredOffer must not be able to be acquired")
	}
	if delay := o.GetDelay(); !(0 < delay && delay <= ttl) {
		t.Error("expiredOffer does not return a valid deadline")
	}
} // TestExpiredOffer

func TestTimedOffer(t *testing.T) {
	t.Parallel()

	ttl := 2 * time.Second
	now := time.Now()
	o := &liveOffer{nil, now.Add(ttl), 0}

	if o.HasExpired() {
		t.Errorf("offer ttl was %v and should not have expired yet", ttl)
	}
	if !o.Acquire() {
		t.Fatal("1st acquisition of offer failed")
	}
	o.Release()
	if !o.Acquire() {
		t.Fatal("2nd acquisition of offer failed")
	}
	if o.Acquire() {
		t.Fatal("3rd acquisition of offer passed but prior claim was not released")
	}
	o.Release()
	if !o.Acquire() {
		t.Fatal("4th acquisition of offer failed")
	}
	o.Release()
	time.Sleep(ttl)
	if !o.HasExpired() {
		t.Fatal("offer not expired after ttl passed")
	}
	if !o.Acquire() {
		t.Fatal("5th acquisition of offer failed; should not be tied to expiration")
	}
	if o.Acquire() {
		t.Fatal("6th acquisition of offer succeeded; should already be acquired")
	}
} // TestTimedOffer

func TestOfferStorage(t *testing.T) {
	ttl := time.Second / 4
	var declinedNum int32
	getDeclinedNum := func() int32 { return atomic.LoadInt32(&declinedNum) }
	config := RegistryConfig{
		DeclineOffer: func(offerId string) <-chan error {
			atomic.AddInt32(&declinedNum, 1)
			return proc.ErrorChan(nil)
		},
		Compat: func(o *mesos.Offer) bool {
			return o.Hostname == nil || *o.Hostname != "incompatiblehost"
		},
		TTL:       ttl,
		LingerTTL: 2 * ttl,
	}
	storage := CreateRegistry(config)

	done := make(chan struct{})
	storage.Init(done)

	// Add offer
	id := util.NewOfferID("foo")
	o := &mesos.Offer{Id: id}
	storage.Add([]*mesos.Offer{o})

	// Added offer should be in the storage
	if obj, ok := storage.Get(id.GetValue()); obj == nil || !ok {
		t.Error("offer not added")
	}
	if obj, _ := storage.Get(id.GetValue()); obj.Details() != o {
		t.Error("added offer differs from returned offer")
	}

	// Not-added offer is not in storage
	if obj, ok := storage.Get("bar"); obj != nil || ok {
		t.Error("offer bar should not exist in storage")
	}

	// Deleted offer lingers in storage, is acquired and declined
	offer, _ := storage.Get(id.GetValue())
	declinedNumBefore := getDeclinedNum()
	storage.Delete(id.GetValue(), "deleted for test")
	if obj, _ := storage.Get(id.GetValue()); obj == nil {
		t.Error("deleted offer is not lingering")
	}
	if obj, _ := storage.Get(id.GetValue()); !obj.HasExpired() {
		t.Error("deleted offer is no expired")
	}
	if ok := offer.Acquire(); ok {
		t.Error("deleted offer can be acquired")
	}
	if getDeclinedNum() <= declinedNumBefore {
		t.Error("deleted offer was not declined")
	}

	// Acquired offer is only declined after 2*ttl
	id = util.NewOfferID("foo2")
	o = &mesos.Offer{Id: id}
	storage.Add([]*mesos.Offer{o})
	offer, _ = storage.Get(id.GetValue())
	declinedNumBefore = getDeclinedNum()
	offer.Acquire()
	storage.Delete(id.GetValue(), "deleted for test")
	if getDeclinedNum() > declinedNumBefore {
		t.Error("acquired offer is declined")
	}

	offer.Release()
	time.Sleep(3 * ttl)
	if getDeclinedNum() <= declinedNumBefore {
		t.Error("released offer is not declined after 2*ttl")
	}

	// Added offer should be expired after ttl, but lingering
	id = util.NewOfferID("foo3")
	o = &mesos.Offer{Id: id}
	storage.Add([]*mesos.Offer{o})

	time.Sleep(2 * ttl)
	obj, ok := storage.Get(id.GetValue())
	if obj == nil || !ok {
		t.Error("offer not lingering after ttl")
	}
	if !obj.HasExpired() {
		t.Error("offer is not expired after ttl")
	}

	// Should be deleted when waiting longer than LingerTTL
	time.Sleep(2 * ttl)
	if obj, ok := storage.Get(id.GetValue()); obj != nil || ok {
		t.Error("offer not deleted after LingerTTL")
	}

	// Incompatible offer is declined
	id = util.NewOfferID("foo4")
	incompatibleHostname := "incompatiblehost"
	o = &mesos.Offer{Id: id, Hostname: &incompatibleHostname}
	declinedNumBefore = getDeclinedNum()
	storage.Add([]*mesos.Offer{o})
	if obj, ok := storage.Get(id.GetValue()); obj != nil || ok {
		t.Error("incompatible offer not rejected")
	}
	if getDeclinedNum() <= declinedNumBefore {
		t.Error("incompatible offer is not declined")
	}

	// Invalidated offer are not declined, but expired
	id = util.NewOfferID("foo5")
	o = &mesos.Offer{Id: id}
	storage.Add([]*mesos.Offer{o})
	offer, _ = storage.Get(id.GetValue())
	declinedNumBefore = getDeclinedNum()
	storage.Invalidate(id.GetValue())
	if obj, _ := storage.Get(id.GetValue()); !obj.HasExpired() {
		t.Error("invalidated offer is not expired")
	}
	if getDeclinedNum() > declinedNumBefore {
		t.Error("invalidated offer is declined")
	}
	if ok := offer.Acquire(); ok {
		t.Error("invalidated offer can be acquired")
	}

	// Invalidate "" will invalidate all offers
	id = util.NewOfferID("foo6")
	o = &mesos.Offer{Id: id}
	storage.Add([]*mesos.Offer{o})
	id2 := util.NewOfferID("foo7")
	o2 := &mesos.Offer{Id: id2}
	storage.Add([]*mesos.Offer{o2})
	storage.Invalidate("")
	if obj, _ := storage.Get(id.GetValue()); !obj.HasExpired() {
		t.Error("invalidated offer is not expired")
	}
	if obj2, _ := storage.Get(id2.GetValue()); !obj2.HasExpired() {
		t.Error("invalidated offer is not expired")
	}

	// InvalidateForSlave invalides all offers for that slave, but only those
	id = util.NewOfferID("foo8")
	slaveId := util.NewSlaveID("test-slave")
	o = &mesos.Offer{Id: id, SlaveId: slaveId}
	storage.Add([]*mesos.Offer{o})
	id2 = util.NewOfferID("foo9")
	o2 = &mesos.Offer{Id: id2}
	storage.Add([]*mesos.Offer{o2})
	storage.InvalidateForSlave(slaveId.GetValue())
	if obj, _ := storage.Get(id.GetValue()); !obj.HasExpired() {
		t.Error("invalidated offer for test-slave is not expired")
	}
	if obj2, _ := storage.Get(id2.GetValue()); obj2.HasExpired() {
		t.Error("invalidated offer another slave is expired")
	}

	close(done)
} // TestOfferStorage

func TestListen(t *testing.T) {
	ttl := time.Second / 4
	config := RegistryConfig{
		DeclineOffer: func(offerId string) <-chan error {
			return proc.ErrorChan(nil)
		},
		Compat: func(o *mesos.Offer) bool {
			return true
		},
		TTL:           ttl,
		ListenerDelay: ttl / 2,
	}
	storage := CreateRegistry(config)

	done := make(chan struct{})
	storage.Init(done)

	// Create two listeners with a hostname filter
	hostname1 := "hostname1"
	hostname2 := "hostname2"
	listener1 := storage.Listen("listener1", func(offer *mesos.Offer) bool {
		return offer.GetHostname() == hostname1
	})
	listener2 := storage.Listen("listener2", func(offer *mesos.Offer) bool {
		return offer.GetHostname() == hostname2
	})

	// Add hostname1 offer
	id := util.NewOfferID("foo")
	o := &mesos.Offer{Id: id, Hostname: &hostname1}
	storage.Add([]*mesos.Offer{o})

	// listener1 is notified by closing channel
	select {
	case _, more := <-listener1:
		if more {
			t.Error("listener1 is not closed")
		}
	}

	// listener2 is not notified within ttl
	select {
	case <-listener2:
		t.Error("listener2 is notified")
	case <-time.After(ttl):
	}

	close(done)
} // TestListen

func TestWalk(t *testing.T) {
	t.Parallel()
	config := RegistryConfig{
		DeclineOffer: func(offerId string) <-chan error {
			return proc.ErrorChan(nil)
		},
		TTL:           0 * time.Second,
		LingerTTL:     0 * time.Second,
		ListenerDelay: 0 * time.Second,
	}
	storage := CreateRegistry(config)
	acceptedOfferId := ""
	walked := 0
	walker1 := func(p Perishable) (bool, error) {
		walked++
		if p.Acquire() {
			acceptedOfferId = p.Details().Id.GetValue()
			return true, nil
		}
		return false, nil
	}
	// sanity check
	err := storage.Walk(walker1)
	if err != nil {
		t.Fatalf("received impossible error %v", err)
	}
	if walked != 0 {
		t.Fatal("walked empty storage")
	}
	if acceptedOfferId != "" {
		t.Fatal("somehow found an offer when registry was empty")
	}
	impl, ok := storage.(*offerStorage)
	if !ok {
		t.Fatal("unexpected offer storage impl")
	}
	// single offer
	ttl := 2 * time.Second
	now := time.Now()
	o := &liveOffer{&mesos.Offer{Id: util.NewOfferID("foo")}, now.Add(ttl), 0}

	impl.offers.Add(o)
	err = storage.Walk(walker1)
	if err != nil {
		t.Fatalf("received impossible error %v", err)
	}
	if walked != 1 {
		t.Fatalf("walk count %d", walked)
	}
	if acceptedOfferId != "foo" {
		t.Fatalf("found offer %v", acceptedOfferId)
	}

	acceptedOfferId = ""
	err = storage.Walk(walker1)
	if err != nil {
		t.Fatalf("received impossible error %v", err)
	}
	if walked != 2 {
		t.Fatalf("walk count %d", walked)
	}
	if acceptedOfferId != "" {
		t.Fatalf("found offer %v", acceptedOfferId)
	}

	walker2 := func(p Perishable) (bool, error) {
		walked++
		return true, nil
	}
	err = storage.Walk(walker2)
	if err != nil {
		t.Fatalf("received impossible error %v", err)
	}
	if walked != 3 {
		t.Fatalf("walk count %d", walked)
	}
	if acceptedOfferId != "" {
		t.Fatalf("found offer %v", acceptedOfferId)
	}

	walker3 := func(p Perishable) (bool, error) {
		walked++
		return true, errors.New("baz")
	}
	err = storage.Walk(walker3)
	if err == nil {
		t.Fatal("expected error")
	}
	if walked != 4 {
		t.Fatalf("walk count %d", walked)
	}
}

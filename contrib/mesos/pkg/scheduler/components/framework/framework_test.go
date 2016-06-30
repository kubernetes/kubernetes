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

package framework

import (
	"reflect"
	"testing"

	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler"
	schedcfg "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/cache"
)

//get number of non-expired offers from  offer registry
func getNumberOffers(os offers.Registry) int {
	//walk offers and check it is stored in registry
	walked := 0
	walker1 := func(p offers.Perishable) (bool, error) {
		walked++
		return false, nil

	}
	os.Walk(walker1)
	return walked
}

type mockRegistrator struct {
	store cache.Store
}

func (r *mockRegistrator) Run(terminate <-chan struct{}) error {
	return nil
}

func (r *mockRegistrator) Register(hostName string, labels map[string]string) (bool, error) {
	obj, _, err := r.store.GetByKey(hostName)
	if err != nil {
		return false, err
	}
	if obj == nil {
		return true, r.store.Add(&api.Node{
			ObjectMeta: api.ObjectMeta{
				Name:   hostName,
				Labels: labels,
			},
			Spec: api.NodeSpec{
				ExternalID: hostName,
			},
			Status: api.NodeStatus{
				Phase: api.NodePending,
			},
		})
	} else {
		n := obj.(*api.Node)
		if reflect.DeepEqual(n.Labels, labels) {
			return false, nil
		}
		n.Labels = labels
		return true, r.store.Update(n)
	}
}

func mockScheduler() scheduler.Scheduler {
	mockScheduler := &scheduler.MockScheduler{}
	reg := podtask.NewInMemoryRegistry()
	mockScheduler.On("Tasks").Return(reg)
	return mockScheduler
}

//test adding of ressource offer, should be added to offer registry and slaves
func TestResourceOffer_Add(t *testing.T) {
	assert := assert.New(t)

	registrator := &mockRegistrator{cache.NewStore(cache.MetaNamespaceKeyFunc)}
	testFramework := &framework{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			DeclineOffer: func(offerId string) <-chan error {
				return proc.ErrorChan(nil)
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaveHostNames:  newSlaveRegistry(),
		nodeRegistrator: registrator,
		sched:           mockScheduler(),
	}

	hostname := "h1"
	offerID1 := util.NewOfferID("test1")
	offer1 := &mesos.Offer{Id: offerID1, Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testFramework.ResourceOffers(nil, offers1)
	assert.Equal(1, len(registrator.store.List()))

	assert.Equal(1, getNumberOffers(testFramework.offers))
	//check slave hostname
	assert.Equal(1, len(testFramework.slaveHostNames.SlaveIDs()))

	//add another offer
	hostname2 := "h2"
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers2 := []*mesos.Offer{offer2}
	testFramework.ResourceOffers(nil, offers2)

	//check it is stored in registry
	assert.Equal(2, getNumberOffers(testFramework.offers))

	//check slave hostnames
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))
}

//test adding of ressource offer, should be added to offer registry and slavesf
func TestResourceOffer_Add_Rescind(t *testing.T) {
	assert := assert.New(t)

	testFramework := &framework{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			DeclineOffer: func(offerId string) <-chan error {
				return proc.ErrorChan(nil)
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaveHostNames: newSlaveRegistry(),
		sched:          mockScheduler(),
	}

	hostname := "h1"
	offerID1 := util.NewOfferID("test1")
	offer1 := &mesos.Offer{Id: offerID1, Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testFramework.ResourceOffers(nil, offers1)

	assert.Equal(1, getNumberOffers(testFramework.offers))

	//check slave hostname
	assert.Equal(1, len(testFramework.slaveHostNames.SlaveIDs()))

	//add another offer
	hostname2 := "h2"
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers2 := []*mesos.Offer{offer2}
	testFramework.ResourceOffers(nil, offers2)

	assert.Equal(2, getNumberOffers(testFramework.offers))

	//check slave hostnames
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))

	//next whether offers can be rescinded
	testFramework.OfferRescinded(nil, offerID1)
	assert.Equal(1, getNumberOffers(testFramework.offers))

	//next whether offers can be rescinded
	testFramework.OfferRescinded(nil, util.NewOfferID("test2"))
	//walk offers again and check it is removed from registry
	assert.Equal(0, getNumberOffers(testFramework.offers))

	//remove non existing ID
	testFramework.OfferRescinded(nil, util.NewOfferID("notExist"))
}

//test that when a slave is lost we remove all offers
func TestSlave_Lost(t *testing.T) {
	assert := assert.New(t)

	//
	testFramework := &framework{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaveHostNames: newSlaveRegistry(),
		sched:          mockScheduler(),
	}

	hostname := "h1"
	offer1 := &mesos.Offer{Id: util.NewOfferID("test1"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testFramework.ResourceOffers(nil, offers1)
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers2 := []*mesos.Offer{offer2}
	testFramework.ResourceOffers(nil, offers2)

	//add another offer from different slaveID
	hostname2 := "h2"
	offer3 := &mesos.Offer{Id: util.NewOfferID("test3"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers3 := []*mesos.Offer{offer3}
	testFramework.ResourceOffers(nil, offers3)

	//test precondition
	assert.Equal(3, getNumberOffers(testFramework.offers))
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))

	//remove first slave
	testFramework.SlaveLost(nil, util.NewSlaveID(hostname))

	//offers should be removed
	assert.Equal(1, getNumberOffers(testFramework.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))

	//remove second slave
	testFramework.SlaveLost(nil, util.NewSlaveID(hostname2))

	//offers should be removed
	assert.Equal(0, getNumberOffers(testFramework.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))

	//try to remove non existing slave
	testFramework.SlaveLost(nil, util.NewSlaveID("notExist"))

}

//test when we loose connection to master we invalidate all cached offers
func TestDisconnect(t *testing.T) {
	assert := assert.New(t)

	//
	testFramework := &framework{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaveHostNames: newSlaveRegistry(),
		sched:          mockScheduler(),
	}

	hostname := "h1"
	offer1 := &mesos.Offer{Id: util.NewOfferID("test1"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testFramework.ResourceOffers(nil, offers1)
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers2 := []*mesos.Offer{offer2}
	testFramework.ResourceOffers(nil, offers2)

	//add another offer from different slaveID
	hostname2 := "h2"
	offer3 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers3 := []*mesos.Offer{offer3}
	testFramework.ResourceOffers(nil, offers3)

	//disconnect
	testFramework.Disconnected(nil)

	//all offers should be removed
	assert.Equal(0, getNumberOffers(testFramework.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testFramework.slaveHostNames.SlaveIDs()))
}

//test we can handle different status updates, TODO check state transitions
func TestStatus_Update(t *testing.T) {

	mockdriver := MockSchedulerDriver{}
	// setup expectations
	mockdriver.On("KillTask", util.NewTaskID("test-task-001")).Return(mesos.Status_DRIVER_RUNNING, nil)

	testFramework := &framework{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaveHostNames: newSlaveRegistry(),
		driver:         &mockdriver,
		sched:          mockScheduler(),
	}

	taskStatus_task_starting := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_RUNNING,
	)
	testFramework.StatusUpdate(testFramework.driver, taskStatus_task_starting)

	taskStatus_task_running := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_RUNNING,
	)
	testFramework.StatusUpdate(testFramework.driver, taskStatus_task_running)

	taskStatus_task_failed := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_FAILED,
	)
	testFramework.StatusUpdate(testFramework.driver, taskStatus_task_failed)

	//assert that mock was invoked
	mockdriver.AssertExpectations(t)
}

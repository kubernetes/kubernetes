/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package scheduler

import (
	"testing"

	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/contrib/mesos/pkg/offers"
	"k8s.io/kubernetes/contrib/mesos/pkg/proc"
	schedcfg "k8s.io/kubernetes/contrib/mesos/pkg/scheduler/config"
	"k8s.io/kubernetes/contrib/mesos/pkg/scheduler/podtask"
)

// Check that same slave is only added once.
func TestSlaveStorage_checkAndAdd(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveStorage()
	assert.Equal(0, len(slaveStorage.slaves))

	slaveId := "slave1"
	slaveHostname := "slave1Hostname"
	slaveStorage.checkAndAdd(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.getSlaveIds()))

	slaveStorage.checkAndAdd(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.getSlaveIds()))
}

// Check that getSlave returns notExist for nonexisting slave.
func TestSlaveStorage_getSlave(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveStorage()
	assert.Equal(0, len(slaveStorage.slaves))

	slaveId := "slave1"
	slaveHostname := "slave1Hostname"

	_, exists := slaveStorage.getSlave(slaveId)
	assert.Equal(false, exists)

	slaveStorage.checkAndAdd(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.getSlaveIds()))

	_, exists = slaveStorage.getSlave(slaveId)
	assert.Equal(true, exists)
}

// Check that getSlaveIds returns array with all slaveIds.
func TestSlaveStorage_getSlaveIds(t *testing.T) {
	assert := assert.New(t)

	slaveStorage := newSlaveStorage()
	assert.Equal(0, len(slaveStorage.slaves))

	slaveId := "1"
	slaveHostname := "hn1"
	slaveStorage.checkAndAdd(slaveId, slaveHostname)
	assert.Equal(1, len(slaveStorage.getSlaveIds()))

	slaveId = "2"
	slaveHostname = "hn2"
	slaveStorage.checkAndAdd(slaveId, slaveHostname)
	assert.Equal(2, len(slaveStorage.getSlaveIds()))

	slaveIds := slaveStorage.getSlaveIds()

	slaveIdsMap := make(map[string]bool, len(slaveIds))
	for _, s := range slaveIds {
		slaveIdsMap[s] = true
	}

	_, ok := slaveIdsMap["1"]
	assert.Equal(ok, true)

	_, ok = slaveIdsMap["2"]
	assert.Equal(ok, true)

}

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

//test adding of ressource offer, should be added to offer registry and slavesf
func TestResourceOffer_Add(t *testing.T) {
	assert := assert.New(t)

	testScheduler := &KubernetesScheduler{
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
		slaves: newSlaveStorage(),
	}

	hostname := "h1"
	offerID1 := util.NewOfferID("test1")
	offer1 := &mesos.Offer{Id: offerID1, Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testScheduler.ResourceOffers(nil, offers1)

	assert.Equal(1, getNumberOffers(testScheduler.offers))
	//check slave hostname
	assert.Equal(1, len(testScheduler.slaves.getSlaveIds()))

	//add another offer
	hostname2 := "h2"
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers2 := []*mesos.Offer{offer2}
	testScheduler.ResourceOffers(nil, offers2)

	//check it is stored in registry
	assert.Equal(2, getNumberOffers(testScheduler.offers))

	//check slave hostnames
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))
}

//test adding of ressource offer, should be added to offer registry and slavesf
func TestResourceOffer_Add_Rescind(t *testing.T) {
	assert := assert.New(t)

	testScheduler := &KubernetesScheduler{
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
		slaves: newSlaveStorage(),
	}

	hostname := "h1"
	offerID1 := util.NewOfferID("test1")
	offer1 := &mesos.Offer{Id: offerID1, Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testScheduler.ResourceOffers(nil, offers1)

	assert.Equal(1, getNumberOffers(testScheduler.offers))

	//check slave hostname
	assert.Equal(1, len(testScheduler.slaves.getSlaveIds()))

	//add another offer
	hostname2 := "h2"
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers2 := []*mesos.Offer{offer2}
	testScheduler.ResourceOffers(nil, offers2)

	assert.Equal(2, getNumberOffers(testScheduler.offers))

	//check slave hostnames
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))

	//next whether offers can be rescinded
	testScheduler.OfferRescinded(nil, offerID1)
	assert.Equal(1, getNumberOffers(testScheduler.offers))

	//next whether offers can be rescinded
	testScheduler.OfferRescinded(nil, util.NewOfferID("test2"))
	//walk offers again and check it is removed from registry
	assert.Equal(0, getNumberOffers(testScheduler.offers))

	//remove non existing ID
	testScheduler.OfferRescinded(nil, util.NewOfferID("notExist"))
}

//test that when a slave is lost we remove all offers
func TestSlave_Lost(t *testing.T) {
	assert := assert.New(t)

	//
	testScheduler := &KubernetesScheduler{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaves: newSlaveStorage(),
	}

	hostname := "h1"
	offer1 := &mesos.Offer{Id: util.NewOfferID("test1"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testScheduler.ResourceOffers(nil, offers1)
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers2 := []*mesos.Offer{offer2}
	testScheduler.ResourceOffers(nil, offers2)

	//add another offer from different slaveID
	hostname2 := "h2"
	offer3 := &mesos.Offer{Id: util.NewOfferID("test3"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers3 := []*mesos.Offer{offer3}
	testScheduler.ResourceOffers(nil, offers3)

	//test precondition
	assert.Equal(3, getNumberOffers(testScheduler.offers))
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))

	//remove first slave
	testScheduler.SlaveLost(nil, util.NewSlaveID(hostname))

	//offers should be removed
	assert.Equal(1, getNumberOffers(testScheduler.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))

	//remove second slave
	testScheduler.SlaveLost(nil, util.NewSlaveID(hostname2))

	//offers should be removed
	assert.Equal(0, getNumberOffers(testScheduler.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))

	//try to remove non existing slave
	testScheduler.SlaveLost(nil, util.NewSlaveID("notExist"))

}

//test when we loose connection to master we invalidate all cached offers
func TestDisconnect(t *testing.T) {
	assert := assert.New(t)

	//
	testScheduler := &KubernetesScheduler{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaves: newSlaveStorage(),
	}

	hostname := "h1"
	offer1 := &mesos.Offer{Id: util.NewOfferID("test1"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers1 := []*mesos.Offer{offer1}
	testScheduler.ResourceOffers(nil, offers1)
	offer2 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname, SlaveId: util.NewSlaveID(hostname)}
	offers2 := []*mesos.Offer{offer2}
	testScheduler.ResourceOffers(nil, offers2)

	//add another offer from different slaveID
	hostname2 := "h2"
	offer3 := &mesos.Offer{Id: util.NewOfferID("test2"), Hostname: &hostname2, SlaveId: util.NewSlaveID(hostname2)}
	offers3 := []*mesos.Offer{offer3}
	testScheduler.ResourceOffers(nil, offers3)

	//disconnect
	testScheduler.Disconnected(nil)

	//all offers should be removed
	assert.Equal(0, getNumberOffers(testScheduler.offers))
	//slave hostnames should still be all present
	assert.Equal(2, len(testScheduler.slaves.getSlaveIds()))
}

//test we can handle different status updates, TODO check state transitions
func TestStatus_Update(t *testing.T) {

	mockdriver := MockSchedulerDriver{}
	// setup expectations
	mockdriver.On("KillTask", util.NewTaskID("test-task-001")).Return(mesos.Status_DRIVER_RUNNING, nil)

	testScheduler := &KubernetesScheduler{
		offers: offers.CreateRegistry(offers.RegistryConfig{
			Compat: func(o *mesos.Offer) bool {
				return true
			},
			// remember expired offers so that we can tell if a previously scheduler offer relies on one
			LingerTTL:     schedcfg.DefaultOfferLingerTTL,
			TTL:           schedcfg.DefaultOfferTTL,
			ListenerDelay: schedcfg.DefaultListenerDelay,
		}),
		slaves:       newSlaveStorage(),
		driver:       &mockdriver,
		taskRegistry: podtask.NewInMemoryRegistry(),
	}

	taskStatus_task_starting := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_RUNNING,
	)
	testScheduler.StatusUpdate(testScheduler.driver, taskStatus_task_starting)

	taskStatus_task_running := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_RUNNING,
	)
	testScheduler.StatusUpdate(testScheduler.driver, taskStatus_task_running)

	taskStatus_task_failed := util.NewTaskStatus(
		util.NewTaskID("test-task-001"),
		mesos.TaskState_TASK_FAILED,
	)
	testScheduler.StatusUpdate(testScheduler.driver, taskStatus_task_failed)

	//assert that mock was invoked
	mockdriver.AssertExpectations(t)
}

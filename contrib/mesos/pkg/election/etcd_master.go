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

package election

import (
	"fmt"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/golang/glog"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/watch"
)

// Master is used to announce the current elected master.
type Master string

// IsAnAPIObject is used solely so we can work with the watch package.
// TODO(k8s): Either fix watch so this isn't necessary, or make this a real API Object.
// TODO(k8s): when it becomes clear how this package will be used, move these declarations to
// to the proper place.
func (Master) IsAnAPIObject() {}

// NewEtcdMasterElector returns an implementation of election.MasterElector backed by etcd.
func NewEtcdMasterElector(h tools.EtcdClient) MasterElector {
	return &etcdMasterElector{etcd: h}
}

type empty struct{}

// internal implementation struct
type etcdMasterElector struct {
	etcd   tools.EtcdClient
	done   chan empty
	events chan watch.Event
}

// Elect implements the election.MasterElector interface.
func (e *etcdMasterElector) Elect(path, id string) watch.Interface {
	e.done = make(chan empty)
	e.events = make(chan watch.Event)
	go util.Until(func() { e.run(path, id) }, time.Second*5, util.NeverStop)
	return e
}

func (e *etcdMasterElector) run(path, id string) {
	masters := make(chan string)
	errors := make(chan error)
	go e.master(path, id, 30, masters, errors, e.done) // TODO(jdef) extract constant
	for {
		select {
		case m := <-masters:
			e.events <- watch.Event{
				Type:   watch.Modified,
				Object: Master(m),
			}
		case e := <-errors:
			glog.Errorf("error in election: %v", e)
		}
	}
}

// ResultChan implements the watch.Interface interface.
func (e *etcdMasterElector) ResultChan() <-chan watch.Event {
	return e.events
}

// extendMaster attempts to extend ownership of a master lock for TTL seconds.
// returns "", nil if extension failed
// returns id, nil if extension succeeded
// returns "", err if an error occurred
func (e *etcdMasterElector) extendMaster(path, id string, ttl uint64, res *etcd.Response) (string, error) {
	// If it matches the passed in id, extend the lease by writing a new entry.
	// Uses compare and swap, so that if we TTL out in the meantime, the write will fail.
	// We don't handle the TTL delete w/o a write case here, it's handled in the next loop
	// iteration.
	_, err := e.etcd.CompareAndSwap(path, id, ttl, "", res.Node.ModifiedIndex)
	if err != nil && !etcdstorage.IsEtcdTestFailed(err) {
		return "", err
	}
	if err != nil && etcdstorage.IsEtcdTestFailed(err) {
		return "", nil
	}
	return id, nil
}

// becomeMaster attempts to become the master for this lock.
// returns "", nil if the attempt failed
// returns id, nil if the attempt succeeded
// returns "", err if an error occurred
func (e *etcdMasterElector) becomeMaster(path, id string, ttl uint64) (string, error) {
	_, err := e.etcd.Create(path, id, ttl)
	if err != nil && !etcdstorage.IsEtcdNodeExist(err) {
		// unexpected error
		return "", err
	}
	if err != nil && etcdstorage.IsEtcdNodeExist(err) {
		return "", nil
	}
	return id, nil
}

// handleMaster performs one loop of master locking.
// on success it returns <master>, nil
// on error it returns "", err
// in situations where you should try again due to concurrent state changes (e.g. another actor simultaneously acquiring the lock)
// it returns "", nil
func (e *etcdMasterElector) handleMaster(path, id string, ttl uint64) (string, error) {
	res, err := e.etcd.Get(path, false, false)

	// Unexpected error, bail out
	if err != nil && !etcdstorage.IsEtcdNotFound(err) {
		return "", err
	}

	// There is no master, try to become the master.
	if err != nil && etcdstorage.IsEtcdNotFound(err) {
		return e.becomeMaster(path, id, ttl)
	}

	// This should never happen.
	if res.Node == nil {
		return "", fmt.Errorf("unexpected response: %#v", res)
	}

	// We're not the master, just return the current value
	if res.Node.Value != id {
		return res.Node.Value, nil
	}

	// We are the master, try to extend out lease
	return e.extendMaster(path, id, ttl, res)
}

// master provices a distributed master election lock, maintains lock until failure, or someone sends something in the done channel.
// The basic algorithm is:
// while !done
//   Get the current master
//   If there is no current master
//      Try to become the master
//   Otherwise
//      If we are the master, extend the lease
//      If the master is different than the last time through the loop, report the master
//   Sleep 80% of TTL
func (e *etcdMasterElector) master(path, id string, ttl uint64, masters chan<- string, errors chan<- error, done <-chan empty) {
	lastMaster := ""
	for {
		master, err := e.handleMaster(path, id, ttl)
		if err != nil {
			errors <- err
		} else if len(master) == 0 {
			continue
		} else if master != lastMaster {
			lastMaster = master
			masters <- master
		}
		// TODO(k8s): Add Watch here, skip the polling for faster reactions
		// If done is closed, break out.
		select {
		case <-done:
			return
		case <-time.After(time.Duration((ttl*8)/10) * time.Second):
		}
	}
}

// ResultChan implements the watch.Interface interface
func (e *etcdMasterElector) Stop() {
	close(e.done)
}

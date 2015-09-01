/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package lb

import (
	"time"

	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/client/unversioned/cache"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

// taskQueue manages a work queue through an independent worker that
// invokes the given sync function for every work item inserted.
type taskQueue struct {
	queue *workqueue.Type
	sync  func(string)
}

func (t *taskQueue) run(period time.Duration, stopCh <-chan struct{}) {
	util.Until(t.worker, period, stopCh)
}

// enqueue enqueues ns/name of the given api object in the task queue.
func (t *taskQueue) enqueue(obj interface{}) {
	key, err := keyFunc(obj)
	if err != nil {
		glog.Infof("Couldn't get key for object %+v: %v", obj, err)
		return
	}
	t.queue.Add(key)
}

func (t *taskQueue) requeue(key string, err error) {
	glog.Errorf("Requeuing %v, err %v", key, err)
	t.queue.Add(key)
}

// worker processes work in the queue through sync.
func (t *taskQueue) worker() {
	for {
		key, _ := t.queue.Get()
		glog.Infof("Syncing %v", key)
		t.sync(key.(string))
		t.queue.Done(key)
	}
}

// NewTaskQueue creates a new task queue with the given sync function.
// The sync function is called for every element inserted into the queue.
func NewTaskQueue(syncFn func(string)) *taskQueue {
	return &taskQueue{
		queue: workqueue.New(),
		sync:  syncFn,
	}
}

// poolStore is used as a cache for cluster resource pools.
type poolStore struct {
	cache.ThreadSafeStore
}

// Returns a read only copy of the k:v pairs in the store.
// Caller beware: Violates traditional snapshot guarantees.
func (p *poolStore) snapshot() map[string]interface{} {
	snap := map[string]interface{}{}
	for _, key := range p.ListKeys() {
		if item, ok := p.Get(key); ok {
			snap[key] = item
		}
	}
	return snap
}

func newPoolStore() *poolStore {
	return &poolStore{
		cache.NewThreadSafeStore(cache.Indexers{}, cache.Indices{})}
}

// compareLinks returns true if the 2 self links are equal.
func compareLinks(l1, l2 string) bool {
	// TODO: These can be partial links
	return l1 == l2
}

// runServer is a debug method.
// Eg invocation add: curl http://localhost:8082 -X POST -d '{"foo.bar.com":{"/test/*": "svcx"}}'
// Eg invocation del: curl http://localhost:8082?type=del -X POST -d '{"foo.bar.com":{"/test/*": "svcx"}}'
func RunTestController(lbName string) {
	client := client.NewOrDie(
		&client.Config{
			Host:    "http://127.0.0.1:8001",
			Version: "v1",
		})

	lbc, err := NewLoadBalancerController(client, lbName)
	if err != nil {
		glog.Fatalf("Unable to create loadbalancer %v", err)
	}
	glog.Infof("Starting loadbalancer")
	lbc.Run(util.NeverStop)
}

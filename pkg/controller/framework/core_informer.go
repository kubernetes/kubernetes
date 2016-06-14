/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
)

type coreInformer struct {
	indexer    cache.Indexer
	controller *Controller
	handlers   []ResourceEventHandler
	started    bool
	// mu protects data structures of coreInformer
	mu sync.Mutex
}

func newCoreInformer(lw cache.ListerWatcher, objType runtime.Object, resyncPeriod time.Duration, indexers cache.Indexers) *coreInformer {
	indexer := cache.NewIndexer(DeletionHandlingMetaNamespaceKeyFunc, indexers)
	informer := &coreInformer{
		indexer: indexer,
	}

	cfg := &Config{
		Queue:            cache.NewDeltaFIFO(cache.MetaNamespaceKeyFunc, nil, indexer),
		ListerWatcher:    lw,
		ObjectType:       objType,
		FullResyncPeriod: resyncPeriod,
		RetryOnError:     false,
		Process:          informer.Process,
	}
	informer.controller = New(cfg)
	return informer
}

func (c *coreInformer) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	c.mu.Lock()
	c.started = true
	c.mu.Unlock()

	c.controller.Run(stopCh)
}

func (c *coreInformer) Process(obj interface{}) error {
	// from oldest to newest
	for _, d := range obj.(cache.Deltas) {
		switch d.Type {
		case cache.Sync, cache.Added, cache.Updated:
			if old, exists, err := c.indexer.Get(d.Object); err == nil && exists {
				if err := c.indexer.Update(d.Object); err != nil {
					return err
				}
				c.OnUpdate(old, d.Object)
			} else {
				if err := c.indexer.Add(d.Object); err != nil {
					return err
				}
				c.OnAdd(d.Object)
			}
		case cache.Deleted:
			if err := c.indexer.Delete(d.Object); err != nil {
				return err
			}
			c.OnDelete(d.Object)
		}
	}
	return nil
}

func (c *coreInformer) addEventHandler(handler ResourceEventHandler) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.started {
		return fmt.Errorf("informer has already started")
	}

	c.handlers = append(c.handlers, handler)
	return nil
}

func (c *coreInformer) OnAdd(newObj interface{}) {
	for _, handler := range c.handlers {
		handler.OnAdd(newObj)
	}
}

func (c *coreInformer) OnUpdate(oldObj, newObj interface{}) {
	for _, handler := range c.handlers {
		handler.OnUpdate(oldObj, newObj)
	}
}

func (c *coreInformer) OnDelete(newObj interface{}) {
	for _, handler := range c.handlers {
		handler.OnDelete(newObj)
	}
}

func (c *coreInformer) HasSynced() bool {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.controller == nil {
		return false
	}
	return c.controller.HasSynced()
}

func (c *coreInformer) GetIndexer() cache.Indexer {
	return c.indexer
}

func (c *coreInformer) AddIndexers(indexers cache.Indexers) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.started {
		return fmt.Errorf("informer has already started")
	}

	return c.indexer.AddIndexers(indexers)
}

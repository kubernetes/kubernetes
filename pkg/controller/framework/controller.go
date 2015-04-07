/*
Copyright 2015 Google Inc. All rights reserved.

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
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/cache"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// Config contains all the settings for a Controller.
type Config struct {
	// The queue for your objects; either a cache.FIFO or
	// a cache.DeltaFIFO. Your Process() function should accept
	// the output of this Oueue's Pop() method.
	cache.Queue

	// Something that can list and watch your objects.
	cache.ListerWatcher

	// Something that can process your objects.
	Process ProcessFunc

	// The type of your objects.
	ObjectType runtime.Object

	// Reprocess everything at least this often.
	// Note that if it takes longer for you to clear the queue than this
	// period, you will end up processing items in the order determined
	// by cache.FIFO.Replace(). Currently, this is random. If this is a
	// problem, we can change that replacement policy to append new
	// things to the end of the queue instead of replacing the entire
	// queue.
	FullResyncPeriod time.Duration

	// If true, when Process() returns an error, re-enqueue the object.
	// TODO: add interface to let you inject a delay/backoff or drop
	//       the object completely if desired. Pass the object in
	//       question to this interface as a parameter.
	RetryOnError bool
}

// ProcessFunc processes a single object.
type ProcessFunc func(obj interface{}) error

// Controller is a generic controller framework.
type Controller struct {
	config Config
}

// New makes a new Controller from the given Config.
func New(c *Config) *Controller {
	ctlr := &Controller{
		config: *c,
	}
	return ctlr
}

// Run begins processing items, and will continue until a value is sent down stopCh.
// It's an error to call Run more than once.
// Run does not block.
func (c *Controller) Run(stopCh <-chan struct{}) {
	cache.NewReflector(
		c.config.ListerWatcher,
		c.config.ObjectType,
		c.config.Queue,
		c.config.FullResyncPeriod,
	).RunUntil(stopCh)

	go util.Until(c.processLoop, time.Second, stopCh)
}

// processLoop drains the work queue.
// TODO: Consider doing the processing in parallel. This will require a little thought
// to make sure that we don't end up processing the same object multiple times
// concurrently.
func (c *Controller) processLoop() {
	for {
		obj := c.config.Queue.Pop()
		err := c.config.Process(obj)
		if err != nil {
			if c.config.RetryOnError {
				// This is the safe way to re-enqueue.
				c.config.Queue.AddIfNotPresent(obj)
			}
		}
	}
}

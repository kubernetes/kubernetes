/*
Copyright 2019 The logr Authors.

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

package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/go-logr/logr"
)

// This application demonstrates the usage of logger.
// It's a simple reconciliation loop that pretends to
// receive notifications about updates from a some API
// server, make some changes, and then submit updates of
// its own.

// This uses object-based logging.  It's also possible
// (but a bit trickier) to use file-level "base" loggers.

var objectMap = map[string]Object{
	"obj1": Object{
		Name: "obj1",
		Kind: "one",
		Details: 33,
	},
	"obj2": Object{
		Name: "obj2",
		Kind: "two",
		Details: "hi",
	},
	"obj3": Object{
		Name: "obj3",
		Kind: "one",
		Details: 1,
	},
}

type Object struct {
	Name string
	Kind string
	Details interface{}
}

type Client struct {
	objects map[string]Object
	log logr.Logger
}

func (c *Client) Get(key string) (Object, error) {
	c.log.V(1).Info("fetching object", "key", key)
	obj, ok := c.objects[key]
	if !ok {
		return Object{}, fmt.Errorf("no object %s exists", key)
	}
	c.log.V(1).Info("pretending to deserialize object", "key", key, "json", "[insert real json here]")
	return obj, nil
}

func (c *Client) Save(obj Object) error {
	c.log.V(1).Info("saving object", "key", obj.Name, "object", obj)
	if rand.Intn(2) == 0 {
		return fmt.Errorf("couldn't save to %s", obj.Name)
	}
	c.log.V(1).Info("pretending to post object", "key", obj.Name, "url", "https://fake.test")
	return nil
}

func (c *Client) WatchNext() string {
	time.Sleep(2*time.Second)

	keyInd := rand.Intn(len(c.objects))

	currInd := 0
	for key := range c.objects {
		if currInd == keyInd {
			return key
		}
		currInd++
	}

	c.log.Info("watch ended")
	return ""
}

type Controller struct {
	log logr.Logger
	expectedKind string
	client *Client
}

func (c *Controller) Run() {
	c.log.Info("starting reconciliation")

	for key := c.client.WatchNext(); key != ""; key = c.client.WatchNext() {
		// we can make more specific loggers if we always want to attach a particular named value
		log := c.log.WithValues("key", key)

		// fetch our object
		obj, err := c.client.Get(key)
		if err != nil {
			log.Error(err, "unable to reconcile object")
			continue
		}

		// make sure it's as expected
		if obj.Kind != c.expectedKind {
			log.Error(nil, "got object that wasn't expected kind", "actual-kind", obj.Kind, "object", obj)
			continue
		}

		// always log the object with log messages
		log = log.WithValues("object", obj)
		log.V(1).Info("reconciling object for key")

		// Do some complicated updates updates
		obj.Details = obj.Details.(int) * 2

		// actually save the updates
		log.V(1).Info("updating object", "details", obj.Details)
		if err := c.client.Save(obj); err != nil {
			log.Error(err, "unable to reconcile object")
		}
	}

	c.log.Info("stopping reconciliation")
}

func NewController(log logr.Logger, objectKind string) *Controller {
	ctrlLogger := log.WithName("controller").WithName(objectKind)
	client := &Client{
		log: ctrlLogger.WithName("client"),
		objects: objectMap,
	}
	return &Controller{
		log: ctrlLogger,
		expectedKind: objectKind,
		client: client,
	}
}

func main() {
	// use a fake implementation just for demonstration purposes
	log := NewTabLogger()

	// update objects with the "one" kind
	ctrl := NewController(log, "one")

	ctrl.Run()
}

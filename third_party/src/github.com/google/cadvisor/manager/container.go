// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Per-container manager.

package manager

import (
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/google/cadvisor/container"
	"github.com/google/cadvisor/info"
	"github.com/google/cadvisor/storage"
)

// Internal mirror of the external data structure.
type containerStat struct {
	Timestamp time.Time
	Data      *info.ContainerStats
}
type containerInfo struct {
	info.ContainerReference
	Subcontainers []info.ContainerReference
	Spec          *info.ContainerSpec
}

type containerData struct {
	handler       container.ContainerHandler
	info          containerInfo
	storageDriver storage.StorageDriver
	lock          sync.Mutex

	// Tells the container to stop.
	stop chan bool
}

func (c *containerData) Start() error {
	// Force the first update.
	c.housekeepingTick()
	log.Printf("Start housekeeping for container %q\n", c.info.Name)

	go c.housekeeping()
	return nil
}

func (c *containerData) Stop() error {
	c.stop <- true
	return nil
}

func (c *containerData) GetInfo() (*containerInfo, error) {
	// TODO(vmarmol): Consider caching this.
	// Get spec and subcontainers.
	err := c.updateSpec()
	if err != nil {
		return nil, err
	}
	err = c.updateSubcontainers()
	if err != nil {
		return nil, err
	}

	// Make a copy of the info for the user.
	c.lock.Lock()
	defer c.lock.Unlock()
	ret := c.info
	return &ret, nil
}

func NewContainerData(containerName string, driver storage.StorageDriver) (*containerData, error) {
	if driver == nil {
		return nil, fmt.Errorf("nil storage driver")
	}
	cont := &containerData{}
	handler, err := container.NewContainerHandler(containerName)
	if err != nil {
		return nil, err
	}
	cont.handler = handler
	ref, err := handler.ContainerReference()
	if err != nil {
		return nil, err
	}
	cont.info.Name = ref.Name
	cont.info.Aliases = ref.Aliases
	cont.storageDriver = driver
	cont.stop = make(chan bool, 1)

	return cont, nil
}

func (c *containerData) housekeeping() {
	// Housekeep every second.
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-c.stop:
			// Stop housekeeping when signaled.
			return
		case <-ticker.C:
			start := time.Now()
			c.housekeepingTick()

			// Log if housekeeping took longer than 120ms.
			duration := time.Since(start)
			if duration >= 120*time.Millisecond {
				log.Printf("Housekeeping(%s) took %s", c.info.Name, duration)
			}
		}
	}
}

func (c *containerData) housekeepingTick() {
	err := c.updateStats()
	if err != nil {
		log.Printf("Failed to update stats for container \"%s\": %s", c.info.Name, err)
	}
}

func (c *containerData) updateSpec() error {
	spec, err := c.handler.GetSpec()
	if err != nil {
		return err
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.info.Spec = spec
	return nil
}

func (c *containerData) updateStats() error {
	stats, err := c.handler.GetStats()
	if err != nil {
		return err
	}
	if stats == nil {
		return nil
	}
	ref, err := c.handler.ContainerReference()
	if err != nil {
		return err
	}
	err = c.storageDriver.AddStats(ref, stats)
	if err != nil {
		return err
	}
	return nil
}

func (c *containerData) updateSubcontainers() error {
	subcontainers, err := c.handler.ListContainers(container.LIST_SELF)
	if err != nil {
		return err
	}
	c.lock.Lock()
	defer c.lock.Unlock()
	c.info.Subcontainers = subcontainers
	return nil
}

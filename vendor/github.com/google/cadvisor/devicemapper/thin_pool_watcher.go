// Copyright 2016 Google Inc. All Rights Reserved.
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

package devicemapper

import (
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"k8s.io/klog/v2"
)

// usageCache is a typed wrapper around atomic.Value that eliminates the need
// for type assertions at every call site. It stores device ID strings mapped
// to usage values (uint64).
type usageCache struct {
	v atomic.Value
}

// Load retrieves the current cache map.
func (c *usageCache) Load() map[string]uint64 {
	return c.v.Load().(map[string]uint64)
}

// Store saves a new cache map.
func (c *usageCache) Store(m map[string]uint64) {
	c.v.Store(m)
}

// ThinPoolWatcher maintains a cache of device name -> usage stats for a
// devicemapper thin-pool using thin_ls.
type ThinPoolWatcher struct {
	poolName       string
	metadataDevice string
	cache          usageCache
	period         time.Duration
	stopChan       chan struct{}
	dmsetup        DmsetupClient
	thinLsClient   thinLsClient
}

// NewThinPoolWatcher returns a new ThinPoolWatcher for the given devicemapper
// thin pool name and metadata device or an error.
func NewThinPoolWatcher(poolName, metadataDevice string) (*ThinPoolWatcher, error) {
	thinLsClient, err := newThinLsClient()
	if err != nil {
		return nil, fmt.Errorf("encountered error creating thin_ls client: %v", err)
	}

	w := &ThinPoolWatcher{
		poolName:       poolName,
		metadataDevice: metadataDevice,
		period:         15 * time.Second,
		stopChan:       make(chan struct{}),
		dmsetup:        NewDmsetupClient(),
		thinLsClient:   thinLsClient,
	}
	w.cache.Store(map[string]uint64{})
	return w, nil
}

// Start starts the ThinPoolWatcher.
func (w *ThinPoolWatcher) Start() {
	err := w.Refresh()
	if err != nil {
		klog.Errorf("encountered error refreshing thin pool watcher: %v", err)
	}

	for {
		select {
		case <-w.stopChan:
			return
		case <-time.After(w.period):
			start := time.Now()
			err = w.Refresh()
			if err != nil {
				klog.Errorf("encountered error refreshing thin pool watcher: %v", err)
			}

			// print latency for refresh
			duration := time.Since(start)
			klog.V(5).Infof("thin_ls(%d) took %s", start.Unix(), duration)
		}
	}
}

// Stop stops the ThinPoolWatcher.
func (w *ThinPoolWatcher) Stop() {
	close(w.stopChan)
}

// GetUsage gets the cached usage value of the given device.
func (w *ThinPoolWatcher) GetUsage(deviceID string) (uint64, error) {
	cache := w.cache.Load()
	v, ok := cache[deviceID]
	if !ok {
		return 0, fmt.Errorf("no cached value for usage of device %v", deviceID)
	}
	return v, nil
}

const (
	reserveMetadataMessage = "reserve_metadata_snap"
	releaseMetadataMessage = "release_metadata_snap"
)

// Refresh performs a `thin_ls` of the pool being watched and refreshes the
// cached data with the result.
func (w *ThinPoolWatcher) Refresh() error {
	currentlyReserved, err := w.checkReservation(w.poolName)
	if err != nil {
		err = fmt.Errorf("error determining whether snapshot is reserved: %v", err)
		return err
	}

	if currentlyReserved {
		klog.V(5).Infof("metadata for %v is currently reserved; releasing", w.poolName)
		_, err = w.dmsetup.Message(w.poolName, 0, releaseMetadataMessage)
		if err != nil {
			err = fmt.Errorf("error releasing metadata snapshot for %v: %v", w.poolName, err)
			return err
		}
	}

	klog.V(5).Infof("reserving metadata snapshot for thin-pool %v", w.poolName)
	// NOTE: "0" in the call below is for the 'sector' argument to 'dmsetup
	// message'.  It's not needed for thin pools.
	if output, err := w.dmsetup.Message(w.poolName, 0, reserveMetadataMessage); err != nil {
		err = fmt.Errorf("error reserving metadata for thin-pool %v: %v output: %v", w.poolName, err, string(output))
		return err
	}
	klog.V(5).Infof("reserved metadata snapshot for thin-pool %v", w.poolName)

	defer func() {
		klog.V(5).Infof("releasing metadata snapshot for thin-pool %v", w.poolName)
		_, err := w.dmsetup.Message(w.poolName, 0, releaseMetadataMessage)
		if err != nil {
			klog.Warningf("Unable to release metadata snapshot for thin-pool %v: %s", w.poolName, err)
		}
	}()

	klog.V(5).Infof("running thin_ls on metadata device %v", w.metadataDevice)
	newCache, err := w.thinLsClient.ThinLs(w.metadataDevice)
	if err != nil {
		err = fmt.Errorf("error performing thin_ls on metadata device %v: %v", w.metadataDevice, err)
		return err
	}

	w.cache.Store(newCache)
	return nil
}

const (
	thinPoolDmsetupStatusHeldMetadataRoot = 6
	thinPoolDmsetupStatusMinFields        = thinPoolDmsetupStatusHeldMetadataRoot + 1
)

// checkReservation checks to see whether the thin device is currently holding
// userspace metadata.
func (w *ThinPoolWatcher) checkReservation(poolName string) (bool, error) {
	klog.V(5).Infof("checking whether the thin-pool is holding a metadata snapshot")
	output, err := w.dmsetup.Status(poolName)
	if err != nil {
		return false, err
	}

	// we care about the field at fields[thinPoolDmsetupStatusHeldMetadataRoot],
	// so make sure we get enough fields
	fields := strings.Fields(string(output))
	if len(fields) < thinPoolDmsetupStatusMinFields {
		return false, fmt.Errorf("unexpected output of dmsetup status command; expected at least %d fields, got %v; output: %v", thinPoolDmsetupStatusMinFields, len(fields), string(output))
	}

	heldMetadataRoot := fields[thinPoolDmsetupStatusHeldMetadataRoot]
	currentlyReserved := heldMetadataRoot != "-"
	return currentlyReserved, nil
}

// Copyright 2015 Google Inc. All Rights Reserved.
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

// Handler for Docker containers.
package docker

import (
	"sync"
	"time"

	"github.com/golang/glog"
	"github.com/google/cadvisor/fs"
)

type fsHandler interface {
	start()
	usage() uint64
	stop()
}

type realFsHandler struct {
	sync.RWMutex
	lastUpdate  time.Time
	usageBytes  uint64
	period      time.Duration
	storageDirs []string
	fsInfo      fs.FsInfo
	// Tells the container to stop.
	stopChan chan struct{}
}

const longDu = time.Second

var _ fsHandler = &realFsHandler{}

func newFsHandler(period time.Duration, storageDirs []string, fsInfo fs.FsInfo) fsHandler {
	return &realFsHandler{
		lastUpdate:  time.Time{},
		usageBytes:  0,
		period:      period,
		storageDirs: storageDirs,
		fsInfo:      fsInfo,
		stopChan:    make(chan struct{}, 1),
	}
}

func (fh *realFsHandler) needsUpdate() bool {
	return time.Now().After(fh.lastUpdate.Add(fh.period))
}

func (fh *realFsHandler) update() error {
	var usage uint64
	for _, dir := range fh.storageDirs {
		// TODO(Vishh): Add support for external mounts.
		dirUsage, err := fh.fsInfo.GetDirUsage(dir)
		if err != nil {
			return err
		}
		usage += dirUsage
	}
	fh.Lock()
	defer fh.Unlock()
	fh.lastUpdate = time.Now()
	fh.usageBytes = usage
	return nil
}

func (fh *realFsHandler) trackUsage() {
	for {
		start := time.Now()
		if _, ok := <-fh.stopChan; !ok {
			return
		}
		if err := fh.update(); err != nil {
			glog.V(2).Infof("failed to collect filesystem stats - %v", err)
		}
		duration := time.Since(start)
		if duration > longDu {
			glog.V(3).Infof("`du` on following dirs took %v: %v", duration, fh.storageDirs)
		}
		next := start.Add(fh.period)
		time.Sleep(next.Sub(time.Now()))
	}
}

func (fh *realFsHandler) start() {
	go fh.trackUsage()
}

func (fh *realFsHandler) stop() {
	close(fh.stopChan)
}

func (fh *realFsHandler) usage() uint64 {
	fh.RLock()
	defer fh.RUnlock()
	return fh.usageBytes
}

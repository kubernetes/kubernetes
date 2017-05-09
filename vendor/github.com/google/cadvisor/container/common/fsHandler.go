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
package common

import (
	"sync"
	"time"

	"github.com/google/cadvisor/fs"

	"github.com/golang/glog"
)

type FsHandler interface {
	Start()
	Usage() (baseUsageBytes uint64, totalUsageBytes uint64)
	Stop()
}

type realFsHandler struct {
	sync.RWMutex
	lastUpdate     time.Time
	usageBytes     uint64
	baseUsageBytes uint64
	period         time.Duration
	minPeriod      time.Duration
	rootfs         string
	extraDir       string
	fsInfo         fs.FsInfo
	// Tells the container to stop.
	stopChan chan struct{}
}

const (
	longDu             = time.Second
	duTimeout          = time.Minute
	maxDuBackoffFactor = 20
)

var _ FsHandler = &realFsHandler{}

func NewFsHandler(period time.Duration, rootfs, extraDir string, fsInfo fs.FsInfo) FsHandler {
	return &realFsHandler{
		lastUpdate:     time.Time{},
		usageBytes:     0,
		baseUsageBytes: 0,
		period:         period,
		minPeriod:      period,
		rootfs:         rootfs,
		extraDir:       extraDir,
		fsInfo:         fsInfo,
		stopChan:       make(chan struct{}, 1),
	}
}

func (fh *realFsHandler) update() error {
	var (
		baseUsage, extraDirUsage uint64
		err                      error
	)
	// TODO(vishh): Add support for external mounts.
	if fh.rootfs != "" {
		baseUsage, err = fh.fsInfo.GetDirUsage(fh.rootfs, duTimeout)
		if err != nil {
			return err
		}
	}

	if fh.extraDir != "" {
		extraDirUsage, err = fh.fsInfo.GetDirUsage(fh.extraDir, duTimeout)
		if err != nil {
			return err
		}
	}

	fh.Lock()
	defer fh.Unlock()
	fh.lastUpdate = time.Now()
	fh.usageBytes = baseUsage + extraDirUsage
	fh.baseUsageBytes = baseUsage
	return nil
}

func (fh *realFsHandler) trackUsage() {
	fh.update()
	for {
		select {
		case <-fh.stopChan:
			return
		case <-time.After(fh.period):
			start := time.Now()
			if err := fh.update(); err != nil {
				glog.Errorf("failed to collect filesystem stats - %v", err)
				fh.period = fh.period * 2
				if fh.period > maxDuBackoffFactor*fh.minPeriod {
					fh.period = maxDuBackoffFactor * fh.minPeriod
				}
			} else {
				fh.period = fh.minPeriod
			}
			duration := time.Since(start)
			if duration > longDu {
				glog.V(2).Infof("`du` on following dirs took %v: %v", duration, []string{fh.rootfs, fh.extraDir})
			}
		}
	}
}

func (fh *realFsHandler) Start() {
	go fh.trackUsage()
}

func (fh *realFsHandler) Stop() {
	close(fh.stopChan)
}

func (fh *realFsHandler) Usage() (baseUsageBytes, totalUsageBytes uint64) {
	fh.RLock()
	defer fh.RUnlock()
	return fh.baseUsageBytes, fh.usageBytes
}

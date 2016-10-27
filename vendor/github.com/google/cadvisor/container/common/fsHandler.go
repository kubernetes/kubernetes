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
	"fmt"
	"sync"
	"time"

	"github.com/google/cadvisor/fs"

	"github.com/golang/glog"
)

type FsHandler interface {
	Start()
	Usage() FsUsage
	Stop()
}

type FsUsage struct {
	BaseUsageBytes  uint64
	TotalUsageBytes uint64
	InodeUsage      uint64
}

type realFsHandler struct {
	sync.RWMutex
	lastUpdate time.Time
	usage      FsUsage
	period     time.Duration
	minPeriod  time.Duration
	rootfs     string
	extraDir   string
	fsInfo     fs.FsInfo
	// Tells the container to stop.
	stopChan chan struct{}
}

const (
	longOp           = time.Second
	timeout          = 2 * time.Minute
	maxBackoffFactor = 20
)

const DefaultPeriod = time.Minute

var _ FsHandler = &realFsHandler{}

func NewFsHandler(period time.Duration, rootfs, extraDir string, fsInfo fs.FsInfo) FsHandler {
	return &realFsHandler{
		lastUpdate: time.Time{},
		usage:      FsUsage{},
		period:     period,
		minPeriod:  period,
		rootfs:     rootfs,
		extraDir:   extraDir,
		fsInfo:     fsInfo,
		stopChan:   make(chan struct{}, 1),
	}
}

func (fh *realFsHandler) update() error {
	var (
		baseUsage, extraDirUsage, inodeUsage    uint64
		rootDiskErr, rootInodeErr, extraDiskErr error
	)
	// TODO(vishh): Add support for external mounts.
	if fh.rootfs != "" {
		baseUsage, rootDiskErr = fh.fsInfo.GetDirDiskUsage(fh.rootfs, timeout)
		inodeUsage, rootInodeErr = fh.fsInfo.GetDirInodeUsage(fh.rootfs, timeout)
	}

	if fh.extraDir != "" {
		extraDirUsage, extraDiskErr = fh.fsInfo.GetDirDiskUsage(fh.extraDir, timeout)
	}

	// Wait to handle errors until after all operartions are run.
	// An error in one will not cause an early return, skipping others
	fh.Lock()
	defer fh.Unlock()
	fh.lastUpdate = time.Now()
	if rootDiskErr == nil && fh.rootfs != "" {
		fh.usage.InodeUsage = inodeUsage
	}
	if rootInodeErr == nil && fh.rootfs != "" {
		fh.usage.TotalUsageBytes = baseUsage + extraDirUsage
	}
	if extraDiskErr == nil && fh.extraDir != "" {
		fh.usage.BaseUsageBytes = baseUsage
	}
	// Combine errors into a single error to return
	if rootDiskErr != nil || rootInodeErr != nil || extraDiskErr != nil {
		return fmt.Errorf("rootDiskErr: %v, rootInodeErr: %v, extraDiskErr: %v", rootDiskErr, rootInodeErr, extraDiskErr)
	}
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
				if fh.period > maxBackoffFactor*fh.minPeriod {
					fh.period = maxBackoffFactor * fh.minPeriod
				}
			} else {
				fh.period = fh.minPeriod
			}
			duration := time.Since(start)
			if duration > longOp {
				glog.V(2).Infof("du and find on following dirs took %v: %v", duration, []string{fh.rootfs, fh.extraDir})
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

func (fh *realFsHandler) Usage() FsUsage {
	fh.RLock()
	defer fh.RUnlock()
	return fh.usage
}

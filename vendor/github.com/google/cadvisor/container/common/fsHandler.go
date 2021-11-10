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

	"k8s.io/klog/v2"
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
		rootUsage, extraUsage fs.UsageInfo
		rootErr, extraErr     error
	)
	// TODO(vishh): Add support for external mounts.
	if fh.rootfs != "" {
		rootUsage, rootErr = fh.fsInfo.GetDirUsage(fh.rootfs)
	}

	if fh.extraDir != "" {
		extraUsage, extraErr = fh.fsInfo.GetDirUsage(fh.extraDir)
	}

	// Wait to handle errors until after all operartions are run.
	// An error in one will not cause an early return, skipping others
	fh.Lock()
	defer fh.Unlock()
	fh.lastUpdate = time.Now()
	if fh.rootfs != "" && rootErr == nil {
		fh.usage.InodeUsage = rootUsage.Inodes
		fh.usage.BaseUsageBytes = rootUsage.Bytes
		fh.usage.TotalUsageBytes = rootUsage.Bytes
	}
	if fh.extraDir != "" && extraErr == nil {
		if fh.rootfs != "" {
			fh.usage.TotalUsageBytes += extraUsage.Bytes
		} else {
			// rootfs is empty, totalUsageBytes use extra usage bytes
			fh.usage.TotalUsageBytes = extraUsage.Bytes
		}
	}

	// Combine errors into a single error to return
	if rootErr != nil || extraErr != nil {
		return fmt.Errorf("rootDiskErr: %v, extraDiskErr: %v", rootErr, extraErr)
	}
	return nil
}

func (fh *realFsHandler) trackUsage() {
	longOp := time.Second
	for {
		start := time.Now()
		if err := fh.update(); err != nil {
			klog.Errorf("failed to collect filesystem stats - %v", err)
			fh.period = fh.period * 2
			if fh.period > maxBackoffFactor*fh.minPeriod {
				fh.period = maxBackoffFactor * fh.minPeriod
			}
		} else {
			fh.period = fh.minPeriod
		}
		duration := time.Since(start)
		if duration > longOp {
			// adapt longOp time so that message doesn't continue to print
			// if the long duration is persistent either because of slow
			// disk or lots of containers.
			longOp = longOp + time.Second
			klog.V(2).Infof("fs: disk usage and inodes count on following dirs took %v: %v; will not log again for this container unless duration exceeds %v", duration, []string{fh.rootfs, fh.extraDir}, longOp)
		}
		select {
		case <-fh.stopChan:
			return
		case <-time.After(fh.period):
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

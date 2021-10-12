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

type FsUsageProvider interface {
	// Usage returns the fs usage
	Usage() (*FsUsage, error)
	// Targets returns where the fs usage metric is collected,it maybe a directory, a file or some
	// information about the snapshotter(for containerd)
	Targets() []string
}

type realFsHandler struct {
	sync.RWMutex
	lastUpdate    time.Time
	usage         FsUsage
	period        time.Duration
	minPeriod     time.Duration
	usageProvider FsUsageProvider
	// Tells the container to stop.
	stopChan chan struct{}
}

const (
	maxBackoffFactor = 20
)

const DefaultPeriod = time.Minute

var _ FsHandler = &realFsHandler{}

func NewFsHandler(period time.Duration, provider FsUsageProvider) FsHandler {
	return &realFsHandler{
		lastUpdate:    time.Time{},
		usage:         FsUsage{},
		period:        period,
		minPeriod:     period,
		usageProvider: provider,
		stopChan:      make(chan struct{}, 1),
	}
}

func (fh *realFsHandler) update() error {

	usage, err := fh.usageProvider.Usage()

	if err != nil {
		return err
	}

	fh.Lock()
	defer fh.Unlock()
	fh.lastUpdate = time.Now()

	fh.usage.InodeUsage = usage.InodeUsage
	fh.usage.BaseUsageBytes = usage.BaseUsageBytes
	fh.usage.TotalUsageBytes = usage.TotalUsageBytes

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
			klog.V(2).Infof(`fs: disk usage and inodes count on targets took %v: %v; `+
				`will not log again for this container unless duration exceeds %v`, duration, fh.usageProvider.Targets(), longOp)
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

type fsUsageProvider struct {
	fsInfo fs.FsInfo
	rootFs string
	// The directory consumed by the container but outside rootFs, e.g. directory of saving logs
	extraDir string
}

func NewGeneralFsUsageProvider(fsInfo fs.FsInfo, rootFs, extraDir string) FsUsageProvider {
	return &fsUsageProvider{
		fsInfo:   fsInfo,
		rootFs:   rootFs,
		extraDir: extraDir,
	}
}

func (f *fsUsageProvider) Targets() []string {
	return []string{f.rootFs, f.extraDir}
}

func (f *fsUsageProvider) Usage() (*FsUsage, error) {
	var (
		rootUsage, extraUsage fs.UsageInfo
		rootErr, extraErr     error
	)

	if f.rootFs != "" {
		rootUsage, rootErr = f.fsInfo.GetDirUsage(f.rootFs)
	}

	if f.extraDir != "" {
		extraUsage, extraErr = f.fsInfo.GetDirUsage(f.extraDir)
	}

	usage := &FsUsage{}

	if f.rootFs != "" && rootErr == nil {
		usage.InodeUsage = rootUsage.Inodes
		usage.BaseUsageBytes = rootUsage.Bytes
		usage.TotalUsageBytes = rootUsage.Bytes
	}

	if f.extraDir != "" && extraErr == nil {
		usage.TotalUsageBytes += extraUsage.Bytes
	}

	// Combine errors into a single error to return
	if rootErr != nil || extraErr != nil {
		return nil, fmt.Errorf("failed to obtain filesystem usage; rootDiskErr: %v, extraDiskErr: %v", rootErr, extraErr)
	}

	return usage, nil
}

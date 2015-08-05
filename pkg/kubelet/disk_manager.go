/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"fmt"
	"sync"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/cadvisor"
	"github.com/golang/glog"
	cadvisorApi "github.com/google/cadvisor/info/v2"
)

// Manages policy for diskspace management for disks holding docker images and root fs.

// Implementation is thread-safe.
type diskSpaceManager interface {
	// Checks the available disk space
	IsRootDiskSpaceAvailable() (bool, error)
	IsDockerDiskSpaceAvailable() (bool, error)
	// Always returns sufficient space till Unfreeze() is called.
	// This is a signal from caller that its internal initialization is done.
	Unfreeze()
}

type DiskSpacePolicy struct {
	// free disk space threshold for filesystem holding docker images.
	DockerFreeDiskMB int
	// free disk space threshold for root filesystem. Host volumes are created on root fs.
	RootFreeDiskMB int
}

type fsInfo struct {
	Usage     int64
	Capacity  int64
	Available int64
	Timestamp time.Time
}

type realDiskSpaceManager struct {
	cadvisor   cadvisor.Interface
	cachedInfo map[string]fsInfo // cache of filesystem info.
	lock       sync.Mutex        // protecting cachedInfo and frozen.
	policy     DiskSpacePolicy   // thresholds. Set at creation time.
	frozen     bool              // space checks always return ok when frozen is set. True on creation.
}

func (dm *realDiskSpaceManager) getFsInfo(fsType string, f func() (cadvisorApi.FsInfo, error)) (fsInfo, error) {
	dm.lock.Lock()
	defer dm.lock.Unlock()
	fsi := fsInfo{}
	if info, ok := dm.cachedInfo[fsType]; ok {
		timeLimit := time.Now().Add(-2 * time.Second)
		if info.Timestamp.After(timeLimit) {
			fsi = info
		}
	}
	if fsi.Timestamp.IsZero() {
		fs, err := f()
		if err != nil {
			return fsInfo{}, err
		}
		fsi.Timestamp = time.Now()
		fsi.Usage = int64(fs.Usage)
		fsi.Capacity = int64(fs.Capacity)
		fsi.Available = int64(fs.Available)
		dm.cachedInfo[fsType] = fsi
	}
	return fsi, nil
}

func (dm *realDiskSpaceManager) IsDockerDiskSpaceAvailable() (bool, error) {
	return dm.isSpaceAvailable("docker", dm.policy.DockerFreeDiskMB, dm.cadvisor.DockerImagesFsInfo)
}

func (dm *realDiskSpaceManager) IsRootDiskSpaceAvailable() (bool, error) {
	return dm.isSpaceAvailable("root", dm.policy.RootFreeDiskMB, dm.cadvisor.RootFsInfo)
}

func (dm *realDiskSpaceManager) isSpaceAvailable(fsType string, threshold int, f func() (cadvisorApi.FsInfo, error)) (bool, error) {
	if dm.frozen {
		return true, nil
	}
	fsInfo, err := dm.getFsInfo(fsType, f)
	if err != nil {
		return true, fmt.Errorf("failed to get fs info for %q: %v", fsType, err)
	}
	if fsInfo.Capacity == 0 {
		return true, fmt.Errorf("could not determine capacity for %q fs. Info: %+v", fsType, fsInfo)
	}
	if fsInfo.Available < 0 {
		return true, fmt.Errorf("wrong available space for %q: %+v", fsType, fsInfo)
	}

	const mb = int64(1024 * 1024)

	if fsInfo.Available < int64(threshold)*mb {
		glog.Infof("Running out of space on disk for %q: available %d MB, threshold %d MB", fsType, fsInfo.Available/mb, threshold)
		return false, nil
	}
	return true, nil
}

func (dm *realDiskSpaceManager) Unfreeze() {
	dm.lock.Lock()
	defer dm.lock.Unlock()
	dm.frozen = false
}

func validatePolicy(policy DiskSpacePolicy) error {
	if policy.DockerFreeDiskMB < 0 {
		return fmt.Errorf("free disk space should be non-negative. Invalid value %d for docker disk space threshold.", policy.DockerFreeDiskMB)
	}
	if policy.RootFreeDiskMB < 0 {
		return fmt.Errorf("free disk space should be non-negative. Invalid value %d for root disk space threshold.", policy.RootFreeDiskMB)
	}
	return nil
}

func newDiskSpaceManager(cadvisorInterface cadvisor.Interface, policy DiskSpacePolicy) (diskSpaceManager, error) {
	// validate policy
	err := validatePolicy(policy)
	if err != nil {
		return nil, err
	}

	dm := &realDiskSpaceManager{
		cadvisor:   cadvisorInterface,
		policy:     policy,
		cachedInfo: map[string]fsInfo{},
		frozen:     true,
	}

	return dm, nil
}

// Copyright 2015 The etcd Authors
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

package api

import (
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/go-semver/semver"
	"github.com/coreos/pkg/capnslog"
)

type Capability string

const (
	AuthCapability  Capability = "auth"
	V3rpcCapability Capability = "v3rpc"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd/etcdserver", "api")

	// capabilityMaps is a static map of version to capability map.
	// the base capabilities is the set of capability 2.0 supports.
	capabilityMaps = map[string]map[Capability]bool{
		"2.1.0": {AuthCapability: true},
		"2.2.0": {AuthCapability: true},
		"2.3.0": {AuthCapability: true},
		"3.0.0": {AuthCapability: true, V3rpcCapability: true},
	}

	// capLoopOnce ensures we only create one capability monitor goroutine
	capLoopOnce sync.Once

	enableMapMu sync.RWMutex
	// enabledMap points to a map in capabilityMaps
	enabledMap map[Capability]bool
)

func init() {
	enabledMap = make(map[Capability]bool)
}

// RunCapabilityLoop checks the cluster version every 500ms and updates
// the enabledMap when the cluster version increased.
func RunCapabilityLoop(s *etcdserver.EtcdServer) {
	go capLoopOnce.Do(func() { runCapabilityLoop(s) })
}

func runCapabilityLoop(s *etcdserver.EtcdServer) {
	stopped := s.StopNotify()

	var pv *semver.Version
	for {
		if v := s.ClusterVersion(); v != pv {
			if pv == nil || (v != nil && pv.LessThan(*v)) {
				pv = v
				enableMapMu.Lock()
				enabledMap = capabilityMaps[pv.String()]
				enableMapMu.Unlock()
				plog.Infof("enabled capabilities for version %s", pv)
			}
		}

		select {
		case <-stopped:
			return
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func IsCapabilityEnabled(c Capability) bool {
	enableMapMu.RLock()
	defer enableMapMu.RUnlock()
	if enabledMap == nil {
		return false
	}
	return enabledMap[c]
}

func EnableCapability(c Capability) {
	enableMapMu.Lock()
	defer enableMapMu.Unlock()
	enabledMap[c] = true
}

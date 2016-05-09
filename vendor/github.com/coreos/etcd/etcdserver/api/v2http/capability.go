// Copyright 2015 CoreOS, Inc.
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

package v2http

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	"github.com/coreos/go-semver/semver"
)

type capability string

const (
	authCapability capability = "auth"
)

var (
	// capabilityMaps is a static map of version to capability map.
	// the base capabilities is the set of capability 2.0 supports.
	capabilityMaps = map[string]map[capability]bool{
		"2.1.0": {authCapability: true},
		"2.2.0": {authCapability: true},
		"2.3.0": {authCapability: true},
	}

	enableMapMu sync.Mutex
	// enabledMap points to a map in capabilityMaps
	enabledMap map[capability]bool
)

// capabilityLoop checks the cluster version every 500ms and updates
// the enabledMap when the cluster version increased.
// capabilityLoop MUST be ran in a goroutine before checking capability
// or using capabilityHandler.
func capabilityLoop(s *etcdserver.EtcdServer) {
	stopped := s.StopNotify()

	var pv *semver.Version
	for {
		if v := s.ClusterVersion(); v != pv {
			if pv == nil {
				pv = v
			} else if v != nil && pv.LessThan(*v) {
				pv = v
			}
			enableMapMu.Lock()
			enabledMap = capabilityMaps[pv.String()]
			enableMapMu.Unlock()
			plog.Infof("enabled capabilities for version %s", pv)
		}

		select {
		case <-stopped:
			return
		case <-time.After(500 * time.Millisecond):
		}
	}
}

func isCapabilityEnabled(c capability) bool {
	enableMapMu.Lock()
	defer enableMapMu.Unlock()
	if enabledMap == nil {
		return false
	}
	return enabledMap[c]
}

func capabilityHandler(c capability, fn func(http.ResponseWriter, *http.Request)) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if !isCapabilityEnabled(c) {
			notCapable(w, r, c)
			return
		}
		fn(w, r)
	}
}

func notCapable(w http.ResponseWriter, r *http.Request, c capability) {
	herr := httptypes.NewHTTPError(http.StatusInternalServerError, fmt.Sprintf("Not capable of accessing %s feature during rolling upgrades.", c))
	if err := herr.WriteTo(w); err != nil {
		plog.Debugf("error writing HTTPError (%v) to %s", err, r.RemoteAddr)
	}
}

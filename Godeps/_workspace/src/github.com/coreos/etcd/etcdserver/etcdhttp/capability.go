package etcdhttp

import (
	"fmt"
	"net/http"
	"sync"
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/etcdhttp/httptypes"
	"github.com/coreos/go-semver/semver"
)

type capability string

const (
	authCapability capability = "auth"
)

var (
	// capabilityMap is a static map of version to capability map.
	// the base capabilities is the set of capability 2.0 supports.
	capabilityMaps = map[string]map[capability]bool{
		"2.1.0": {authCapability: true},
		"2.2.0": {authCapability: true},
	}

	enableMapMu sync.Mutex
	// enabled points to a map in cpapbilityMaps
	enabledMap map[capability]bool
)

// capabilityLoop checks the cluster version every 500ms and updates
// the enabledCapability when the cluster version increased.
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
			notCapable(w, c)
			return
		}
		fn(w, r)
	}
}

func notCapable(w http.ResponseWriter, c capability) {
	herr := httptypes.NewHTTPError(http.StatusInternalServerError, fmt.Sprintf("Not capable of accessing %s feature during rolling upgrades.", c))
	herr.WriteTo(w)
}

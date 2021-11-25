package userns

import (
	"sync"

	"github.com/opencontainers/runc/libcontainer/user"
)

var (
	inUserNS bool
	nsOnce   sync.Once
)

// runningInUserNS detects whether we are currently running in a user namespace.
// Originally copied from github.com/lxc/lxd/shared/util.go
func runningInUserNS() bool {
	nsOnce.Do(func() {
		uidmap, err := user.CurrentProcessUIDMap()
		if err != nil {
			// This kernel-provided file only exists if user namespaces are supported
			return
		}
		inUserNS = uidMapInUserNS(uidmap)
	})
	return inUserNS
}

func uidMapInUserNS(uidmap []user.IDMap) bool {
	/*
	 * We assume we are in the initial user namespace if we have a full
	 * range - 4294967295 uids starting at uid 0.
	 */
	if len(uidmap) == 1 && uidmap[0].ID == 0 && uidmap[0].ParentID == 0 && uidmap[0].Count == 4294967295 {
		return false
	}
	return true
}

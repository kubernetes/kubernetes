/*
Temporary API endpoint for libcontainer while the full API is finalized (api.go).
*/
package libcontainer

import (
	"github.com/docker/libcontainer/cgroups/fs"
	"github.com/docker/libcontainer/network"
)

// TODO(vmarmol): Complete Stats() in final libcontainer API and move users to that.
// DEPRECATED: The below portions are only to be used during the transition to the official API.
// Returns all available stats for the given container.
func GetStats(container *Config, state *State) (stats *ContainerStats, err error) {
	stats = &ContainerStats{}
	if stats.CgroupStats, err = fs.GetStats(state.CgroupPaths); err != nil {
		return stats, err
	}
	stats.NetworkStats, err = network.GetStats(&state.NetworkState)
	return stats, err
}

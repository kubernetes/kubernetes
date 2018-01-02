// +build !windows

package daemon

import (
	"fmt"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/container"
)

// Resolve Network SandboxID in case the container reuse another container's network stack
func (daemon *Daemon) getNetworkSandboxID(c *container.Container) (string, error) {
	curr := c
	for curr.HostConfig.NetworkMode.IsContainer() {
		containerID := curr.HostConfig.NetworkMode.ConnectedContainer()
		connected, err := daemon.GetContainer(containerID)
		if err != nil {
			return "", fmt.Errorf("Could not get container for %s", containerID)
		}
		curr = connected
	}
	return curr.NetworkSettings.SandboxID, nil
}

func (daemon *Daemon) getNetworkStats(c *container.Container) (map[string]types.NetworkStats, error) {
	sandboxID, err := daemon.getNetworkSandboxID(c)
	if err != nil {
		return nil, err
	}

	sb, err := daemon.netController.SandboxByID(sandboxID)
	if err != nil {
		return nil, err
	}

	lnstats, err := sb.Statistics()
	if err != nil {
		return nil, err
	}

	stats := make(map[string]types.NetworkStats)
	// Convert libnetwork nw stats into api stats
	for ifName, ifStats := range lnstats {
		stats[ifName] = types.NetworkStats{
			RxBytes:   ifStats.RxBytes,
			RxPackets: ifStats.RxPackets,
			RxErrors:  ifStats.RxErrors,
			RxDropped: ifStats.RxDropped,
			TxBytes:   ifStats.TxBytes,
			TxPackets: ifStats.TxPackets,
			TxErrors:  ifStats.TxErrors,
			TxDropped: ifStats.TxDropped,
		}
	}

	return stats, nil
}

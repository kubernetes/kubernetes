package daemon

import (
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/daemon/execdriver"
	"github.com/docker/libnetwork/sandbox"
	"github.com/opencontainers/runc/libcontainer"
)

type ContainerStatsConfig struct {
	Stream    bool
	OutStream io.Writer
	Stop      <-chan bool
}

func (daemon *Daemon) ContainerStats(name string, config *ContainerStatsConfig) error {
	updates, err := daemon.SubscribeToContainerStats(name)
	if err != nil {
		return err
	}

	if config.Stream {
		config.OutStream.Write(nil)
	}

	var preCpuStats types.CpuStats
	getStat := func(v interface{}) *types.Stats {
		update := v.(*execdriver.ResourceStats)
		// Retrieve the nw statistics from libnetwork and inject them in the Stats
		if nwStats, err := daemon.getNetworkStats(name); err == nil {
			update.Stats.Interfaces = nwStats
		}
		ss := convertStatsToAPITypes(update.Stats)
		ss.PreCpuStats = preCpuStats
		ss.MemoryStats.Limit = uint64(update.MemoryLimit)
		ss.Read = update.Read
		ss.CpuStats.SystemUsage = update.SystemUsage
		preCpuStats = ss.CpuStats
		return ss
	}

	enc := json.NewEncoder(config.OutStream)

	defer daemon.UnsubscribeToContainerStats(name, updates)

	noStreamFirstFrame := true
	for {
		select {
		case v, ok := <-updates:
			if !ok {
				return nil
			}

			s := getStat(v)
			if !config.Stream && noStreamFirstFrame {
				// prime the cpu stats so they aren't 0 in the final output
				noStreamFirstFrame = false
				continue
			}

			if err := enc.Encode(s); err != nil {
				return err
			}

			if !config.Stream {
				return nil
			}
		case <-config.Stop:
			return nil
		}
	}
}

func (daemon *Daemon) getNetworkStats(name string) ([]*libcontainer.NetworkInterface, error) {
	var list []*libcontainer.NetworkInterface

	c, err := daemon.Get(name)
	if err != nil {
		return list, err
	}

	nw, err := daemon.netController.NetworkByID(c.NetworkSettings.NetworkID)
	if err != nil {
		return list, err
	}
	ep, err := nw.EndpointByID(c.NetworkSettings.EndpointID)
	if err != nil {
		return list, err
	}

	stats, err := ep.Statistics()
	if err != nil {
		return list, err
	}

	// Convert libnetwork nw stats into libcontainer nw stats
	for ifName, ifStats := range stats {
		list = append(list, convertLnNetworkStats(ifName, ifStats))
	}

	return list, nil
}

func convertLnNetworkStats(name string, stats *sandbox.InterfaceStatistics) *libcontainer.NetworkInterface {
	n := &libcontainer.NetworkInterface{Name: name}
	n.RxBytes = stats.RxBytes
	n.RxPackets = stats.RxPackets
	n.RxErrors = stats.RxErrors
	n.RxDropped = stats.RxDropped
	n.TxBytes = stats.TxBytes
	n.TxPackets = stats.TxPackets
	n.TxErrors = stats.TxErrors
	n.TxDropped = stats.TxDropped
	return n
}

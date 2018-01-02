// +build linux

package cgroups

import (
	"strconv"

	"github.com/containerd/cgroups"
	metrics "github.com/docker/go-metrics"
	"github.com/prometheus/client_golang/prometheus"
)

var blkioMetrics = []*metric{
	{
		name:   "blkio_io_merged_recursive",
		help:   "The blkio io merged recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoMergedRecursive)
		},
	},
	{
		name:   "blkio_io_queued_recursive",
		help:   "The blkio io queued recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoQueuedRecursive)
		},
	},
	{
		name:   "blkio_io_service_bytes_recursive",
		help:   "The blkio io service bytes recursive",
		unit:   metrics.Bytes,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoServiceBytesRecursive)
		},
	},
	{
		name:   "blkio_io_service_time_recursive",
		help:   "The blkio io servie time recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoServiceTimeRecursive)
		},
	},
	{
		name:   "blkio_io_serviced_recursive",
		help:   "The blkio io servied recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoServicedRecursive)
		},
	},
	{
		name:   "blkio_io_time_recursive",
		help:   "The blkio io time recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.IoTimeRecursive)
		},
	},
	{
		name:   "blkio_sectors_recursive",
		help:   "The blkio sectors recursive",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"op", "device", "major", "minor"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Blkio == nil {
				return nil
			}
			return blkioValues(stats.Blkio.SectorsRecursive)
		},
	},
}

func blkioValues(l []*cgroups.BlkIOEntry) []value {
	var out []value
	for _, e := range l {
		out = append(out, value{
			v: float64(e.Value),
			l: []string{e.Op, e.Device, strconv.FormatUint(e.Major, 10), strconv.FormatUint(e.Minor, 10)},
		})
	}
	return out
}

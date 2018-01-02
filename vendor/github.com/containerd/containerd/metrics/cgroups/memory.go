// +build linux

package cgroups

import (
	"github.com/containerd/cgroups"
	metrics "github.com/docker/go-metrics"
	"github.com/prometheus/client_golang/prometheus"
)

var memoryMetrics = []*metric{
	{
		name: "memory_cache",
		help: "The cache amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Cache),
				},
			}
		},
	},
	{
		name: "memory_rss",
		help: "The rss amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.RSS),
				},
			}
		},
	},
	{
		name: "memory_rss_huge",
		help: "The rss_huge amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.RSSHuge),
				},
			}
		},
	},
	{
		name: "memory_mapped_file",
		help: "The mapped_file amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.MappedFile),
				},
			}
		},
	},
	{
		name: "memory_dirty",
		help: "The dirty amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Dirty),
				},
			}
		},
	},
	{
		name: "memory_writeback",
		help: "The writeback amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Writeback),
				},
			}
		},
	},
	{
		name: "memory_pgpgin",
		help: "The pgpgin amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.PgPgIn),
				},
			}
		},
	},
	{
		name: "memory_pgpgout",
		help: "The pgpgout amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.PgPgOut),
				},
			}
		},
	},
	{
		name: "memory_pgfault",
		help: "The pgfault amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.PgFault),
				},
			}
		},
	},
	{
		name: "memory_pgmajfault",
		help: "The pgmajfault amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.PgMajFault),
				},
			}
		},
	},
	{
		name: "memory_inactive_anon",
		help: "The inactive_anon amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.InactiveAnon),
				},
			}
		},
	},
	{
		name: "memory_active_anon",
		help: "The active_anon amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.ActiveAnon),
				},
			}
		},
	},
	{
		name: "memory_inactive_file",
		help: "The inactive_file amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.InactiveFile),
				},
			}
		},
	},
	{
		name: "memory_active_file",
		help: "The active_file amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.ActiveFile),
				},
			}
		},
	},
	{
		name: "memory_unevictable",
		help: "The unevictable amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Unevictable),
				},
			}
		},
	},
	{
		name: "memory_hierarchical_memory_limit",
		help: "The hierarchical_memory_limit amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.HierarchicalMemoryLimit),
				},
			}
		},
	},
	{
		name: "memory_hierarchical_memsw_limit",
		help: "The hierarchical_memsw_limit amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.HierarchicalSwapLimit),
				},
			}
		},
	},
	{
		name: "memory_total_cache",
		help: "The total_cache amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalCache),
				},
			}
		},
	},
	{
		name: "memory_total_rss",
		help: "The total_rss amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalRSS),
				},
			}
		},
	},
	{
		name: "memory_total_rss_huge",
		help: "The total_rss_huge amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalRSSHuge),
				},
			}
		},
	},
	{
		name: "memory_total_mapped_file",
		help: "The total_mapped_file amount used",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalMappedFile),
				},
			}
		},
	},
	{
		name: "memory_total_dirty",
		help: "The total_dirty amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalDirty),
				},
			}
		},
	},
	{
		name: "memory_total_writeback",
		help: "The total_writeback amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalWriteback),
				},
			}
		},
	},
	{
		name: "memory_total_pgpgin",
		help: "The total_pgpgin amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalPgPgIn),
				},
			}
		},
	},
	{
		name: "memory_total_pgpgout",
		help: "The total_pgpgout amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalPgPgOut),
				},
			}
		},
	},
	{
		name: "memory_total_pgfault",
		help: "The total_pgfault amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalPgFault),
				},
			}
		},
	},
	{
		name: "memory_total_pgmajfault",
		help: "The total_pgmajfault amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalPgMajFault),
				},
			}
		},
	},
	{
		name: "memory_total_inactive_anon",
		help: "The total_inactive_anon amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalInactiveAnon),
				},
			}
		},
	},
	{
		name: "memory_total_active_anon",
		help: "The total_active_anon amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalActiveAnon),
				},
			}
		},
	},
	{
		name: "memory_total_inactive_file",
		help: "The total_inactive_file amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalInactiveFile),
				},
			}
		},
	},
	{
		name: "memory_total_active_file",
		help: "The total_active_file amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalActiveFile),
				},
			}
		},
	},
	{
		name: "memory_total_unevictable",
		help: "The total_unevictable amount",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.TotalUnevictable),
				},
			}
		},
	},
	{
		name: "memory_usage_failcnt",
		help: "The usage failcnt",
		unit: metrics.Total,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Usage.Failcnt),
				},
			}
		},
	},
	{
		name: "memory_usage_limit",
		help: "The memory limit",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Usage.Limit),
				},
			}
		},
	},
	{
		name: "memory_usage_max",
		help: "The memory maximum usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Usage.Max),
				},
			}
		},
	},
	{
		name: "memory_usage_usage",
		help: "The memory usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Usage.Usage),
				},
			}
		},
	},
	{
		name: "memory_swap_failcnt",
		help: "The swap failcnt",
		unit: metrics.Total,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Swap.Failcnt),
				},
			}
		},
	},
	{
		name: "memory_swap_limit",
		help: "The swap limit",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Swap.Limit),
				},
			}
		},
	},
	{
		name: "memory_swap_max",
		help: "The swap maximum usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Swap.Max),
				},
			}
		},
	},
	{
		name: "memory_swap_usage",
		help: "The swap usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Swap.Usage),
				},
			}
		},
	},
	{
		name: "memory_kernel_failcnt",
		help: "The kernel failcnt",
		unit: metrics.Total,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Kernel.Failcnt),
				},
			}
		},
	},
	{
		name: "memory_kernel_limit",
		help: "The kernel limit",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Kernel.Limit),
				},
			}
		},
	},
	{
		name: "memory_kernel_max",
		help: "The kernel maximum usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Kernel.Max),
				},
			}
		},
	},
	{
		name: "memory_kernel_usage",
		help: "The kernel usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.Kernel.Usage),
				},
			}
		},
	},
	{
		name: "memory_kerneltcp_failcnt",
		help: "The kerneltcp failcnt",
		unit: metrics.Total,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.KernelTCP.Failcnt),
				},
			}
		},
	},
	{
		name: "memory_kerneltcp_limit",
		help: "The kerneltcp limit",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.KernelTCP.Limit),
				},
			}
		},
	},
	{
		name: "memory_kerneltcp_max",
		help: "The kerneltcp maximum usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.KernelTCP.Max),
				},
			}
		},
	},
	{
		name: "memory_kerneltcp_usage",
		help: "The kerneltcp usage",
		unit: metrics.Bytes,
		vt:   prometheus.GaugeValue,
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Memory == nil {
				return nil
			}
			return []value{
				{
					v: float64(stats.Memory.KernelTCP.Usage),
				},
			}
		},
	},
}

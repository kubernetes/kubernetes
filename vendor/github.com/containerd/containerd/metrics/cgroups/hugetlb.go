// +build linux

package cgroups

import (
	"github.com/containerd/cgroups"
	metrics "github.com/docker/go-metrics"
	"github.com/prometheus/client_golang/prometheus"
)

var hugetlbMetrics = []*metric{
	{
		name:   "hugetlb_usage",
		help:   "The hugetlb usage",
		unit:   metrics.Bytes,
		vt:     prometheus.GaugeValue,
		labels: []string{"page"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Hugetlb == nil {
				return nil
			}
			var out []value
			for _, v := range stats.Hugetlb {
				out = append(out, value{
					v: float64(v.Usage),
					l: []string{v.Pagesize},
				})
			}
			return out
		},
	},
	{
		name:   "hugetlb_failcnt",
		help:   "The hugetlb failcnt",
		unit:   metrics.Total,
		vt:     prometheus.GaugeValue,
		labels: []string{"page"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Hugetlb == nil {
				return nil
			}
			var out []value
			for _, v := range stats.Hugetlb {
				out = append(out, value{
					v: float64(v.Failcnt),
					l: []string{v.Pagesize},
				})
			}
			return out
		},
	},
	{
		name:   "hugetlb_max",
		help:   "The hugetlb maximum usage",
		unit:   metrics.Bytes,
		vt:     prometheus.GaugeValue,
		labels: []string{"page"},
		getValues: func(stats *cgroups.Metrics) []value {
			if stats.Hugetlb == nil {
				return nil
			}
			var out []value
			for _, v := range stats.Hugetlb {
				out = append(out, value{
					v: float64(v.Max),
					l: []string{v.Pagesize},
				})
			}
			return out
		},
	},
}

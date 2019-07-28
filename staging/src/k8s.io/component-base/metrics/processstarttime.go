/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/procfs"
	"k8s.io/klog"
	"os"
	"time"
)

var processStartTime = prometheus.NewGaugeVec(
	prometheus.GaugeOpts{
		Name: "process_start_time_seconds",
		Help: "Start time of the process since unix epoch in seconds.",
	},
	[]string{},
)

// RegisterProcessStartTime registers the process_start_time_seconds to
// a prometheus registry. This metric needs to be included to ensure counter
// data fidelity.
func RegisterProcessStartTime(registerer prometheus.Registerer) error {
	start, err := getProcessStart()
	if err != nil {
		klog.Errorf("Could not get process start time, %v", err)
		start = float64(time.Now().Unix())
	}
	processStartTime.WithLabelValues().Set(start)
	return registerer.Register(processStartTime)
}

func getProcessStart() (float64, error) {
	pid := os.Getpid()
	p, err := procfs.NewProc(pid)
	if err != nil {
		return 0, err
	}

	if stat, err := p.NewStat(); err == nil {
		return stat.StartTime()
	}
	return 0, err
}

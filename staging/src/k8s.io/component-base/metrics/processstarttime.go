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
	"time"

	"k8s.io/klog/v2"
)

var processStartTime = NewGaugeVec(
	&GaugeOpts{
		Name:           "process_start_time_seconds",
		Help:           "Start time of the process since unix epoch in seconds.",
		StabilityLevel: ALPHA,
	},
	[]string{},
)

// RegisterProcessStartTime registers the process_start_time_seconds to
// a prometheus registry. This metric needs to be included to ensure counter
// data fidelity.
func RegisterProcessStartTime(registrationFunc func(Registerable) error) error {
	start, err := GetProcessStart()
	if err != nil {
		klog.Errorf("Could not get process start time, %v", err)
		start = float64(time.Now().Unix())
	}
	// processStartTime is a lazy metric which only get initialized after registered.
	// so we need to register the metric first and then set the value for it
	if err = registrationFunc(processStartTime); err != nil {
		return err
	}

	processStartTime.WithLabelValues().Set(start)
	return nil
}

/*
Copyright 2015 The Kubernetes Authors.

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

package runtime

import (
	"sync"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/kubernetes/pkg/util/runtime"
)

const (
	runtimeSubsystem = "mesos_runtime"
)

var (
	panicCounter = prometheus.NewCounter(
		prometheus.CounterOpts{
			Subsystem: runtimeSubsystem,
			Name:      "panics",
			Help:      "Counter of panics handled by the internal crash handler.",
		},
	)
)

var registerMetrics sync.Once

func Register() {
	registerMetrics.Do(func() {
		prometheus.MustRegister(panicCounter)
		runtime.PanicHandlers = append(runtime.PanicHandlers, func(interface{}) { panicCounter.Inc() })
	})
}

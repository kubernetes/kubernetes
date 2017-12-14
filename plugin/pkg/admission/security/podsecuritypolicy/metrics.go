/*
Copyright 2017 The Kubernetes Authors.

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

package podsecuritypolicy

import (
	"strconv"

	"github.com/prometheus/client_golang/prometheus"
	"k8s.io/apiserver/pkg/admission"
)

const (
	namespace = "apiserver"
	subsystem = "admission"
)

var (
	admitCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Namespace: namespace,
			Subsystem: subsystem,
			Name:      "controller_admission_latencies_seconds_count",
			Help:      "Admission controller counts, identified by name and broken out for each operation and API resource and type (validate or admit).",
		},
		[]string{"name", "type", "operation", "group", "version", "resource", "subresource", "rejected"},
	)
)

func init() {
	prometheus.MustRegister(admitCounter)
}

func ObserveAdmit(rejected bool, attr admission.Attributes) {
	gvr := attr.GetResource()
	labels := []string{PluginName, "admit", string(attr.GetOperation()), gvr.Group, gvr.Version, gvr.Resource, attr.GetSubresource(), strconv.FormatBool(rejected)}
	admitCounter.WithLabelValues(labels...).Inc()
}

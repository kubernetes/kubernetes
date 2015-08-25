/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

//This is a utility for prometheus pushing functionality.
package e2e

import (
	"fmt"
	"github.com/prometheus/client_golang/prometheus"
)

// Prometheus stuff: Setup metrics.
var runningMetric = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "e2e_running",
	Help: "The num of running pods",
})

var pendingMetric = prometheus.NewGauge(prometheus.GaugeOpts{
	Name: "e2e_pending",
	Help: "The num of pending pods",
})

// Turn this to true after we register.
var prom_registered = false

// Reusable function for pushing metrics to prometheus.  Handles initialization and so on.
func promPushRunningPending(running, pending int) error {

	if testContext.PrometheusPushGateway == "" {
		Logf("Ignoring prom push, push gateway unavailable")
		return nil
	} else {
		// Register metrics if necessary
		if !prom_registered && testContext.PrometheusPushGateway != "" {
			prometheus.Register(runningMetric)
			prometheus.Register(pendingMetric)
			prom_registered = true
		}
		// Update metric values
		runningMetric.Set(float64(running))
		pendingMetric.Set(float64(pending))

		// Push them to the push gateway.  This will be scraped by prometheus
		// provided you launch it with the pushgateway as an endpoint.
		if err := prometheus.Push(
			"e2e",
			"none",
			testContext.PrometheusPushGateway, //i.e. "127.0.0.1:9091"
		); err != nil {
			fmt.Println("failed at pushing to pushgateway ", err)
			return err
		}
	}
	return nil
}

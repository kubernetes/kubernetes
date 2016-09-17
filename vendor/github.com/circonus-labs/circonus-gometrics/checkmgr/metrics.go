// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package checkmgr

import (
	"github.com/circonus-labs/circonus-gometrics/api"
)

// IsMetricActive checks whether a given metric name is currently active(enabled)
func (cm *CheckManager) IsMetricActive(name string) bool {
	active, _ := cm.availableMetrics[name]
	return active
}

// ActivateMetric determines if a given metric should be activated
func (cm *CheckManager) ActivateMetric(name string) bool {
	active, exists := cm.availableMetrics[name]

	if !exists {
		return true
	}

	if !active && cm.forceMetricActivation {
		return true
	}

	return false
}

// AddNewMetrics updates a check bundle with new metrics
func (cm *CheckManager) AddNewMetrics(newMetrics map[string]*api.CheckBundleMetric) {
	// only if check manager is enabled
	if !cm.enabled {
		return
	}

	// only if checkBundle has been populated
	if cm.checkBundle == nil {
		return
	}

	newCheckBundle := cm.checkBundle
	numCurrMetrics := len(newCheckBundle.Metrics)
	numNewMetrics := len(newMetrics)

	if numCurrMetrics+numNewMetrics >= cap(newCheckBundle.Metrics) {
		nm := make([]api.CheckBundleMetric, numCurrMetrics+numNewMetrics)
		copy(nm, newCheckBundle.Metrics)
		newCheckBundle.Metrics = nm
	}

	newCheckBundle.Metrics = newCheckBundle.Metrics[0 : numCurrMetrics+numNewMetrics]

	i := 0
	for _, metric := range newMetrics {
		newCheckBundle.Metrics[numCurrMetrics+i] = *metric
		i++
	}

	checkBundle, err := cm.apih.UpdateCheckBundle(newCheckBundle)
	if err != nil {
		cm.Log.Printf("[ERROR] updating check bundle with new metrics %v", err)
		return
	}

	cm.checkBundle = checkBundle
	cm.inventoryMetrics()
}

// inventoryMetrics creates list of active metrics in check bundle
func (cm *CheckManager) inventoryMetrics() {
	availableMetrics := make(map[string]bool)
	for _, metric := range cm.checkBundle.Metrics {
		availableMetrics[metric.Name] = metric.Status == "active"
	}
	cm.availableMetrics = availableMetrics
}

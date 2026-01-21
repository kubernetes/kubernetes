/*
Copyright 2021 The Kubernetes Authors.

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

package controllers

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/legacyregistry"
	"k8s.io/component-base/metrics/testutil"
)

func TestRunningManagedControllers(t *testing.T) {
	// Reset the metric to ensure clean state
	controllerInstanceCount.Reset()

	// Register the metric
	Register()

	// Create controller manager metrics instance
	managerName := "test-manager"
	metrics := NewControllerManagerMetrics(managerName)

	// Test ControllerStarted
	controllerName := "test-controller"
	metrics.ControllerStarted(controllerName)

	wantStarted := `
		# HELP running_managed_controllers [BETA] Indicates where instances of a controller are currently running
		# TYPE running_managed_controllers gauge
		running_managed_controllers{manager="test-manager",name="test-controller"} 1
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantStarted), "running_managed_controllers"); err != nil {
		t.Fatal(err)
	}

	// Test ControllerStopped
	metrics.ControllerStopped(controllerName)

	wantStopped := `
		# HELP running_managed_controllers [BETA] Indicates where instances of a controller are currently running
		# TYPE running_managed_controllers gauge
		running_managed_controllers{manager="test-manager",name="test-controller"} 0
	`

	if err := testutil.GatherAndCompare(legacyregistry.DefaultGatherer, strings.NewReader(wantStopped), "running_managed_controllers"); err != nil {
		t.Fatal(err)
	}
}

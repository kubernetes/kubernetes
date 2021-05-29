/*
Copyright 2020 The Kubernetes Authors.

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

package apiserver

import (
	"strings"
	"testing"

	"k8s.io/component-base/metrics/testutil"
)

func TestAPIServiceAvailabilityCollection(t *testing.T) {
	collector := newAvailabilityCollector()

	availableAPIService := "available"
	unavailableAPIService := "unavailable"

	collector.SetAPIServiceAvailable(availableAPIService)
	collector.SetAPIServiceUnavailable(unavailableAPIService)

	err := testutil.CustomCollectAndCompare(collector, strings.NewReader(`
	# HELP aggregator_unavailable_apiservice [ALPHA] Gauge of APIServices which are marked as unavailable broken down by APIService name.
	# TYPE aggregator_unavailable_apiservice gauge
	aggregator_unavailable_apiservice{name="available"} 0
	aggregator_unavailable_apiservice{name="unavailable"} 1
	`))
	if err != nil {
		t.Fatal(err)
	}

	collector.ClearState()

	collector.ForgetAPIService(availableAPIService)
	collector.ForgetAPIService(unavailableAPIService)

	err = testutil.CustomCollectAndCompare(collector, strings.NewReader(`
	# HELP aggregator_unavailable_apiservice [ALPHA] Gauge of APIServices which are marked as unavailable broken down by APIService name.
	# TYPE aggregator_unavailable_apiservice gauge
	`))
	if err != nil {
		t.Fatal(err)
	}
}

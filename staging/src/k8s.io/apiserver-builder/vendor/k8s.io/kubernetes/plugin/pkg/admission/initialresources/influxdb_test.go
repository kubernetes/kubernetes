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

package initialresources

import (
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api"
)

func TestInfluxDBGetUsagePercentileCPU(t *testing.T) {
	source, _ := newInfluxdbSource("", "", "", "")
	_, _, err := source.GetUsagePercentile(api.ResourceCPU, 90, "", "", true, time.Now(), time.Now())
	if err == nil {
		t.Errorf("Expected error because InfluxDB is temporarily disabled")
	}
}

func TestInfluxDBGetUsagePercentileMemory(t *testing.T) {
	source, _ := newInfluxdbSource("", "", "", "")
	_, _, err := source.GetUsagePercentile(api.ResourceMemory, 90, "", "foo", false, time.Now(), time.Now())
	if err == nil {
		t.Errorf("Expected error because InfluxDB is temporarily disabled")
	}
}

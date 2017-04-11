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

import "testing"

func TestInfluxDBDataSource(t *testing.T) {
	ds, _ := newDataSource("influxdb")
	if _, ok := ds.(*influxdbSource); !ok {
		t.Errorf("newDataSource did not return valid InfluxDB type")
	}
}

func TestGCMDataSource(t *testing.T) {
	// No ProjectID set
	newDataSource("gcm")
}

func TestHawkularDataSource(t *testing.T) {
	ds, _ := newDataSource("hawkular")
	if _, ok := ds.(*hawkularSource); !ok {
		t.Errorf("newDataSource did not return valid hawkularSource type")
	}
}

func TestNoDataSourceFound(t *testing.T) {
	ds, err := newDataSource("")
	if ds != nil || err == nil {
		t.Errorf("newDataSource found for empty input")
	}
}

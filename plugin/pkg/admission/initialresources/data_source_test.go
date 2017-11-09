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
	internalapi "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources"
	versionedapi "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources/v1alpha1"
	"testing"
)

func GetDefaultConfiConfiguration() (*internalapi.Configuration, error) {
	externalConfig := &versionedapi.Configuration{}
	scheme.Default(externalConfig)
	internalConfig := &internalapi.Configuration{}
	if err := scheme.Convert(externalConfig, internalConfig, nil); err != nil {
		return nil, err
	}
	return internalConfig, nil
}

func TestInfluxDBDataSource(t *testing.T) {
	config, err := GetDefaultConfiConfiguration()
	if err != nil {
		t.Errorf("get default confiConfiguration err: %v", err)
	}
	ds, _ := newDataSource(config)
	if _, ok := ds.(*influxdbSource); !ok {
		t.Errorf("newDataSource did not return valid InfluxDB type")
	}
}

func TestGCMDataSource(t *testing.T) {
	config, err := GetDefaultConfiConfiguration()
	if err != nil {
		t.Errorf("get default confiConfiguration err: %v", err)
	}
	config.DataSourceInfo.DataSource = internalapi.Gcm
	// No ProjectID set
	newDataSource(config)
}

func TestHawkularDataSource(t *testing.T) {
	config, err := GetDefaultConfiConfiguration()
	if err != nil {
		t.Errorf("get default confiConfiguration err: %v", err)
	}
	config.DataSourceInfo.DataSource = internalapi.Hawkular
	ds, _ := newDataSource(config)
	if _, ok := ds.(*hawkularSource); !ok {
		t.Errorf("newDataSource did not return valid hawkularSource type")
	}
}

func TestNoDataSourceFound(t *testing.T) {
	config, err := GetDefaultConfiConfiguration()
	if err != nil {
		t.Errorf("get default confiConfiguration err: %v", err)
	}
	config.DataSourceInfo.DataSource = ""
	ds, err := newDataSource(config)
	if ds != nil || err == nil {
		t.Errorf("newDataSource found for empty input")
	}
}

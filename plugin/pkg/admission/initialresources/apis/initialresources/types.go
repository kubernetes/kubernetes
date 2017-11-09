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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// DataSourceType is the type of the data source (e.g., influxdb, gcm, hawkular)
type DataSourceType string

const (
	Influxdb DataSourceType = "influxdb"
	Gcm      DataSourceType = "gcm"
	Hawkular DataSourceType = "hawkular"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Configuration provides configuration for the initialresources admission controller.
type Configuration struct {
	metav1.TypeMeta `json:",inline"`

	// Data source information used by InitialResources
	DataSourceInfo DataSourceInfo `json:"datasourceinfo,omitempty"`

	// Which percentile of samples should InitialResources use when estimating resources. For experiment purposes.
	Percentile int64 `json:"percentile,omitempty"`

	// Whether the estimation should be made only based on data from the same namespace.
	NamespaceOnly bool `json:"namespaceonly,omitempty"`
}

type DataSourceInfo struct {
	// Data source used by InitialResources. Supported options: influxdb, gcm, hawkular
	DataSource DataSourceType `json:"datasource,omitempty"`

	// Address of InfluxDB which contains metrics required by InitialResources
	InfluxdbHost string `json:"influxdbhost,omitempty"`

	// User used for connecting to InfluxDB
	InfluxdbUser string `json:"influxdbuser,omitempty"`

	// Password used for connecting to InfluxDB
	InfluxdbPassword string `json:"influxdbpassword,omitempty"`

	// InfluxDB database name which contains metrics required by InitialResources
	InfluxdbName string `json:"influxdbname,omitempty"`

	// Hawkular configuration URL
	HawkularUrl string `json:"hawkularurl,omitempty"`
}

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

package v1alpha1

import "k8s.io/apimachinery/pkg/runtime"

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

func SetDefaults_Configuration(obj *Configuration) {
	if obj.Percentile == 0 {
		obj.Percentile = 90
	}

	if obj.DataSourceInfo.DataSource == "" {
		obj.DataSourceInfo.DataSource = Influxdb
	}

	if obj.DataSourceInfo.InfluxdbHost == "" {
		obj.DataSourceInfo.InfluxdbHost = "localhost:8080/api/v1/namespaces/kube-system/services/monitoring-influxdb:api/proxy"
	}

	if obj.DataSourceInfo.InfluxdbUser == "" {
		obj.DataSourceInfo.InfluxdbUser = "root"
	}

	if obj.DataSourceInfo.InfluxdbPassword == "" {
		obj.DataSourceInfo.InfluxdbPassword = "root"
	}

	if obj.DataSourceInfo.InfluxdbName == "" {
		obj.DataSourceInfo.InfluxdbName = "k8s"
	}
}

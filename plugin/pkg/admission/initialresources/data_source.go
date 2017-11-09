/*
Copyright 2015 The Kubernetes Authors.

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
	"fmt"
	"time"

	api "k8s.io/kubernetes/pkg/apis/core"
	pluginapi "k8s.io/kubernetes/plugin/pkg/admission/initialresources/apis/initialresources"
)

// WARNING: If you are planning to add another implementation of dataSource interface please bear in mind,
// that dataSource will be moved to Heapster some time in the future and possibly rewritten.
type dataSource interface {
	// Returns <perc>th of sample values which represent usage of <kind> for containers running <image>,
	// within time range (start, end), number of samples considered and error if occurred.
	// If <exactMatch> then take only samples that concern the same image (both name and take are the same),
	// otherwise consider also samples with the same image a possibly different tag.
	GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (usage int64, samples int64, err error)
}

func newDataSource(pluginConfig *pluginapi.Configuration) (dataSource, error) {
	if pluginConfig == nil {
		return nil, fmt.Errorf("empty data source configuration")
	}

	switch pluginConfig.DataSourceInfo.DataSource {
	case pluginapi.Influxdb:
		influxdbHost := pluginConfig.DataSourceInfo.InfluxdbHost
		user := pluginConfig.DataSourceInfo.InfluxdbUser
		password := pluginConfig.DataSourceInfo.InfluxdbPassword
		db := pluginConfig.DataSourceInfo.InfluxdbName
		return newInfluxdbSource(influxdbHost, user, password, db)
	case pluginapi.Gcm:
		return newGcmSource()
	case pluginapi.Hawkular:
		return newHawkularSource(pluginConfig.DataSourceInfo.HawkularURL)
	default:
		return nil, fmt.Errorf("unknown data source %v", pluginConfig.DataSourceInfo.DataSource)
	}
}

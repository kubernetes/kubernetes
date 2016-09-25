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
	"flag"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
)

var (
	influxdbHost = flag.String("ir-influxdb-host", "localhost:8080/api/v1/proxy/namespaces/kube-system/services/monitoring-influxdb:api", "Address of InfluxDB which contains metrics requred by InitialResources")
	user         = flag.String("ir-user", "root", "User used for connecting to InfluxDB")
	// TODO: figure out how to better pass password here
	password       = flag.String("ir-password", "root", "Password used for connecting to InfluxDB")
	db             = flag.String("ir-dbname", "k8s", "InfluxDB database name which contains metrics requred by InitialResources")
	hawkularConfig = flag.String("ir-hawkular", "", "Hawkular configuration URL")
)

// WARNING: If you are planning to add another implementation of dataSource interface please bear in mind,
// that dataSource will be moved to Heapster some time in the future and possibly rewritten.
type dataSource interface {
	// Returns <perc>th of sample values which represent usage of <kind> for containers running <image>,
	// withing time range (start, end), number of samples considered and error if occured.
	// If <exactMatch> then take only samples that concern the same image (both name and take are the same),
	// otherwise consider also samples with the same image a possibly different tag.
	GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (usage int64, samples int64, err error)
}

func newDataSource(kind string) (dataSource, error) {
	if kind == "influxdb" {
		return newInfluxdbSource(*influxdbHost, *user, *password, *db)
	}
	if kind == "gcm" {
		return newGcmSource()
	}
	if kind == "hawkular" {
		return newHawkularSource(*hawkularConfig)
	}
	return nil, fmt.Errorf("unknown data source %v", kind)
}

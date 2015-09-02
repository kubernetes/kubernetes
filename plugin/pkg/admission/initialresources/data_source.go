/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"strings"
	"time"

	"github.com/golang/glog"
	influxdb "github.com/influxdb/influxdb/client"
	"k8s.io/kubernetes/pkg/api"
)

const (
	cpuSeriesName      = "autoscaling.cpu.usage.1m"
	memSeriesName      = "autoscaling.memory.usage.1m"
	cpuContinuousQuery = "select derivative(value) as value from \"cpu/usage_ns_cumulative\" where pod_id <> '' group by pod_id, container_name, container_base_image, time(1m) into " + cpuSeriesName
	memContinuousQuery = "select mean(value) as value from \"memory/usage_bytes_gauge\" where pod_id <> '' group by pod_id, container_name, container_base_image, time(1m) into " + memSeriesName
	timeFormat         = "2006-01-02 15:04:05"
)

var (
	influxdbHost = flag.String("ir-influxdb-host", "localhost:8080/api/v1/proxy/namespaces/kube-system/services/monitoring-influxdb:api", "Address of InfluxDB which contains metrics requred by InitialResources")
	user         = flag.String("ir-user", "root", "User used for connecting to InfluxDB")
	// TODO: figure out how to better pass password here
	password = flag.String("ir-password", "root", "Password used for connecting to InfluxDB")
	db       = flag.String("ir-dbname", "k8s", "InfluxDB database name which contains metrics requred by InitialResources")
)

// WARNING: If you are planning to add another implementation of dataSource interface please bear in mind,
// that dataSource will be moved to Heapster some time in the future and possibly rewritten.
type dataSource interface {
	// Returns <perc>th of sample values which represent usage of <kind> for containers running <image>,
	// withing time range (start, end), number of samples considered and error if occured.
	// If <exactMatch> then take only samples that concern the same image (both name and take are the same),
	// otherwise consider also samples with the same image a possibly different tag.
	GetUsagePercentile(kind api.ResourceName, perc int64, image string, exactMatch bool, start, end time.Time) (usage int64, samples int64, err error)
}

func newDataSource(kind string) (dataSource, error) {
	if kind == "influxdb" {
		return newInfluxdbSource(*influxdbHost, *user, *password, *db)
	}
	if kind == "gcm" {
		return newGcmSource()
	}
	return nil, fmt.Errorf("Unknown data source %v", kind)
}

// TODO(piosz): rewrite this once we will migrate into InfluxDB v0.9.
type influxdbSource struct {
	conf *influxdb.ClientConfig
}

func newInfluxdbSource(host, user, password, db string) (dataSource, error) {
	conf := &influxdb.ClientConfig{
		Host:     host,
		Username: user,
		Password: password,
		Database: db,
	}
	source := &influxdbSource{
		conf: conf,
	}
	go source.ensureAutoscalingSeriesExist()
	return source, nil
}

func ensureSeriesExists(conn *influxdb.Client, existingQueries *influxdb.Series, seriesName, contQuery string) error {
	queryExists := false
	for _, p := range existingQueries.GetPoints() {
		id := p[1].(float64)
		query := p[2].(string)
		if strings.Contains(query, "into "+seriesName) {
			if query != contQuery {
				if _, err := conn.Query(fmt.Sprintf("drop continuous query %v", id), influxdb.Second); err != nil {
					return err
				}
			} else {
				queryExists = true
			}
		}
	}
	if !queryExists {
		if _, err := conn.Query("drop series "+seriesName, influxdb.Second); err != nil {
			return err
		}
		if _, err := conn.Query(contQuery, influxdb.Second); err != nil {
			return err
		}
	}
	return nil
}

func (s *influxdbSource) ensureAutoscalingSeriesExist() {
	for {
		time.Sleep(30 * time.Second)
		client, err := influxdb.NewClient(s.conf)
		if err != nil {
			glog.Errorf("Error while trying to create InfluxDB client: %v", err)
			continue
		}
		series, err := client.Query("list continuous queries", influxdb.Second)
		if err != nil {
			glog.Errorf("Error while trying to list continuous queries: %v", err)
			continue
		}
		if err := ensureSeriesExists(client, series[0], cpuSeriesName, cpuContinuousQuery); err != nil {
			glog.Errorf("Error while trying to create create autoscaling series: %v", err)
			continue
		}
		if err := ensureSeriesExists(client, series[0], memSeriesName, memContinuousQuery); err != nil {
			glog.Errorf("Error while trying to create create autoscaling series: %v", err)
			continue
		}
		break
	}
}

func (s *influxdbSource) query(query string, precision ...influxdb.TimePrecision) ([]*influxdb.Series, error) {
	client, err := influxdb.NewClient(s.conf)
	if err != nil {
		return nil, err
	}
	return client.Query(query, precision...)
}

func (s *influxdbSource) GetUsagePercentile(kind api.ResourceName, perc int64, image string, exactMatch bool, start, end time.Time) (int64, int64, error) {
	var series string
	if kind == api.ResourceCPU {
		series = cpuSeriesName
	} else if kind == api.ResourceMemory {
		series = memSeriesName
	}

	var imgPattern string
	if exactMatch {
		imgPattern = "='" + image + "'"
	} else {
		imgPattern = "=~/^" + image + "/"
	}

	query := fmt.Sprintf("select percentile(value, %v), count(pod_id) from %v where container_base_image%v and time > '%v' and time < '%v'", perc, series, imgPattern, start.UTC().Format(timeFormat), end.UTC().Format(timeFormat))
	var res []*influxdb.Series
	var err error
	if res, err = s.query(query, influxdb.Second); err != nil {
		return 0, 0, fmt.Errorf("Error while trying to query InfluxDB: %v", err)
	}

	// TODO(pszczesniak): fix issue with dropped data base
	if len(res) == 0 {
		return 0, 0, fmt.Errorf("Missing series %v in InfluxDB", series)
	}
	points := res[0].GetPoints()
	if len(points) == 0 {
		return 0, 0, fmt.Errorf("Missing data in series %v in InfluxDB", series)
	}
	p := points[0]
	usage := p[1].(float64)
	count := p[2].(float64)
	if kind == api.ResourceCPU {
		// convert from ns to millicores
		usage = usage / 1000000
	}
	return int64(usage), int64(count), nil
}

type gcmSource struct{}

func newGcmSource() (dataSource, error) {
	return &gcmSource{}, fmt.Errorf("gcm source not implemented")
}

func (s *gcmSource) GetUsagePercentile(kind api.ResourceName, perc int64, image string, exactMatch bool, start, end time.Time) (int64, int64, error) {
	return 0, 0, fmt.Errorf("gcm source not implemented")
}

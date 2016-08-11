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
	"strings"
	"time"

	influxdb "github.com/influxdb/influxdb/client"
	"k8s.io/kubernetes/pkg/api"
)

const (
	cpuSeriesName      = "autoscaling.cpu.usage.2m"
	memSeriesName      = "autoscaling.memory.usage.2m"
	cpuContinuousQuery = "select derivative(value) as value from \"cpu/usage_ns_cumulative\" where pod_id <> '' group by pod_id, pod_namespace, container_name, container_base_image, time(2m) into " + cpuSeriesName
	memContinuousQuery = "select mean(value) as value from \"memory/usage_bytes_gauge\" where pod_id <> '' group by pod_id, pod_namespace, container_name, container_base_image, time(2m) into " + memSeriesName
	timeFormat         = "2006-01-02 15:04:05"
)

// TODO(piosz): rewrite this once we will migrate into InfluxDB v0.9.
type influxdbSource struct{}

func newInfluxdbSource(host, user, password, db string) (dataSource, error) {
	return &influxdbSource{}, nil
}

func (s *influxdbSource) query(query string) ([]*influxdb.Response, error) {
	// TODO(piosz): add support again
	return nil, fmt.Errorf("temporary not supported; see #18826 for more details")
}

func (s *influxdbSource) GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (int64, int64, error) {
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
		// Escape character "/" in image pattern.
		imgPattern = "=~/^" + strings.Replace(image, "/", "\\/", -1) + "/"
	}
	var namespaceCond string
	if namespace != "" {
		namespaceCond = " and pod_namespace='" + namespace + "'"
	}

	query := fmt.Sprintf("select percentile(value, %v), count(pod_id) from %v where container_base_image%v%v and time > '%v' and time < '%v'", perc, series, imgPattern, namespaceCond, start.UTC().Format(timeFormat), end.UTC().Format(timeFormat))
	if _, err := s.query(query); err != nil {
		return 0, 0, fmt.Errorf("error while trying to query InfluxDB: %v", err)
	}
	return 0, 0, nil
}

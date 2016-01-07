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
	"math"
	"sort"
	"time"

	"k8s.io/kubernetes/pkg/api"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
	gcm "google.golang.org/api/cloudmonitoring/v2beta2"
	gce "google.golang.org/cloud/compute/metadata"
)

const (
	kubePrefix    = "custom.cloudmonitoring.googleapis.com/kubernetes.io/"
	cpuMetricName = kubePrefix + "cpu/usage_rate"
	memMetricName = kubePrefix + "memory/usage"
	labelImage    = kubePrefix + "label/container_base_image"
	labelNs       = kubePrefix + "label/pod_namespace"
)

type gcmSource struct {
	project    string
	gcmService *gcm.Service
}

func newGcmSource() (dataSource, error) {
	// Detect project ID
	projectId, err := gce.ProjectID()
	if err != nil {
		return nil, err
	}

	// Create Google Cloud Monitoring service.
	client := oauth2.NewClient(oauth2.NoContext, google.ComputeTokenSource(""))
	s, err := gcm.New(client)
	if err != nil {
		return nil, err
	}

	return &gcmSource{
		project:    projectId,
		gcmService: s,
	}, nil
}

func (s *gcmSource) query(metric, oldest, youngest string, labels []string, pageToken string) (*gcm.ListTimeseriesResponse, error) {
	req := s.gcmService.Timeseries.List(s.project, metric, youngest, nil).
		Oldest(oldest).
		Aggregator("mean").
		Window("1m")
	for _, l := range labels {
		req = req.Labels(l)
	}
	if pageToken != "" {
		req = req.PageToken(pageToken)
	}
	return req.Do()
}

func retrieveRawSamples(res *gcm.ListTimeseriesResponse, output *[]int) {
	for _, ts := range res.Timeseries {
		for _, p := range ts.Points {
			*output = append(*output, int(p.DoubleValue))
		}
	}
}

func (s *gcmSource) GetUsagePercentile(kind api.ResourceName, perc int64, image, namespace string, exactMatch bool, start, end time.Time) (int64, int64, error) {
	var metric string
	if kind == api.ResourceCPU {
		metric = cpuMetricName
	} else if kind == api.ResourceMemory {
		metric = memMetricName
	}

	var labels []string
	if exactMatch {
		labels = append(labels, labelImage+"=="+image)
	} else {
		labels = append(labels, labelImage+"=~"+image+".*")
	}
	if namespace != "" {
		labels = append(labels, labelNs+"=="+namespace)
	}

	oldest := start.Format(time.RFC3339)
	youngest := end.Format(time.RFC3339)

	rawSamples := make([]int, 0)
	pageToken := ""
	for {
		res, err := s.query(metric, oldest, youngest, labels, pageToken)
		if err != nil {
			return 0, 0, err
		}

		retrieveRawSamples(res, &rawSamples)

		pageToken = res.NextPageToken
		if pageToken == "" {
			break
		}
	}

	count := len(rawSamples)
	if count == 0 {
		return 0, 0, nil
	}
	sort.Ints(rawSamples)
	usageIndex := int64(math.Ceil(float64(count)*9/10)) - 1
	usage := rawSamples[usageIndex]

	return int64(usage), int64(count), nil
}

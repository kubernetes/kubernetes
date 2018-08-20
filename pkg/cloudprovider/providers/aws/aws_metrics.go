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

package aws

import (
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/apimachinery/pkg/types"
)

var (
	awsAPIMetric = prometheus.NewHistogramVec(
		prometheus.HistogramOpts{
			Name: "cloudprovider_aws_api_request_duration_seconds",
			Help: "Latency of AWS API calls",
		},
		[]string{"request"})

	awsAPIErrorMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cloudprovider_aws_api_request_errors",
			Help: "AWS API errors",
		},
		[]string{"request"})

	awsAPIThrottlesMetric = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "cloudprovider_aws_api_throttled_requests_total",
			Help: "AWS API throttled requests",
		},
		[]string{"operation_name"})

	awsVolumesInStateTotal = prometheus.NewDesc(
		prometheus.BuildFQName("", "cloudprovider_aws", "volumes_in_state_total"),
		"AWS EBS volumes in state attaching, detaching, or busy",
		[]string{"node", "state"}, nil)
	awsVolumesInStateDurationMinutes = prometheus.NewDesc(
		prometheus.BuildFQName("", "cloudprovider_aws", "volumes_in_state_duration_minutes"),
		"Time spent by AWS EBS volumes in state attaching",
		[]string{"volume", "state"}, nil)
)

func recordAWSMetric(actionName string, timeTaken float64, err error) {
	if err != nil {
		awsAPIErrorMetric.With(prometheus.Labels{"request": actionName}).Inc()
	} else {
		awsAPIMetric.With(prometheus.Labels{"request": actionName}).Observe(timeTaken)
	}
}

func recordAWSThrottlesMetric(operation string) {
	awsAPIThrottlesMetric.With(prometheus.Labels{"operation_name": operation}).Inc()
}

func newVolumeStateCollector(cloud *Cloud) *volumeStateCollector {
	return &volumeStateCollector{cloud}
}

type volumeStateCollector struct {
	c *Cloud
}

var _ prometheus.Collector = &volumeStateCollector{}

func (vsc *volumeStateCollector) Describe(ch chan<- *prometheus.Desc) {
	ch <- awsVolumesInStateTotal
	ch <- awsVolumesInStateDurationMinutes
}

func (vsc *volumeStateCollector) Collect(ch chan<- prometheus.Metric) {
	filters := []*ec2.Filter{newEc2Filter("instance-state-name", "running")}
	instances, err := vsc.c.describeInstances(filters)
	if err != nil {
		glog.Warningf("Failed to describe instances while collecting volume state metrics: %v", err)
		return
	}

	for volume, durations := range getVolumesInStateDuration(instances) {
		for state, duration := range durations {
			metric, err := prometheus.NewConstMetric(awsVolumesInStateDurationMinutes, prometheus.CounterValue, duration, volume, state)
			if err != nil {
				glog.Warningf("Failed to create metric for volume %q: %v", volume, err)
			}
			ch <- metric
		}
	}
	for node, totals := range getVolumesInStateTotal(instances) {
		for state, total := range totals {
			metric, err := prometheus.NewConstMetric(awsVolumesInStateTotal, prometheus.GaugeValue, total, node, state)
			if err != nil {
				glog.Warningf("Failed to create metric for node %q: %v", node, err)
			}
			ch <- metric
		}
	}
}

func getVolumesInStateDuration(instances []*ec2.Instance) map[string]map[string]float64 {
	durationsByVolume := make(map[string]map[string]float64)
	for _, instance := range instances {
		for _, mapping := range instance.BlockDeviceMappings {
			ebs := mapping.Ebs
			volume := string(awsVolumeID(aws.StringValue(ebs.VolumeId)))
			state := *ebs.Status
			attachTime := *ebs.AttachTime

			switch state {
			case "attaching":
				duration := time.Since(attachTime).Minutes()
				durationsByVolume[volume] = map[string]float64{state: duration}
			}
		}
	}
	return durationsByVolume
}

func getVolumesInStateTotal(instances []*ec2.Instance) map[string]map[string]float64 {
	totalsByNode := make(map[string]map[string]float64)
	for _, instance := range instances {
		node := string(types.NodeName(aws.StringValue(instance.PrivateDnsName)))
		volumesInStateTotal := make(map[string]float64)

		for _, mapping := range instance.BlockDeviceMappings {
			state := *mapping.Ebs.Status
			switch state {
			case "attaching", "detaching", "busy":
				volumesInStateTotal[state]++
			}
		}
		if len(volumesInStateTotal) > 0 {
			totalsByNode[node] = volumesInStateTotal
		}
	}
	return totalsByNode
}

func registerMetrics() {
	prometheus.MustRegister(awsAPIMetric)
	prometheus.MustRegister(awsAPIErrorMetric)
	prometheus.MustRegister(awsAPIThrottlesMetric)
}

func registerControllerMetrics(cloud *Cloud) {
	prometheus.MustRegister(newVolumeStateCollector(cloud))
}

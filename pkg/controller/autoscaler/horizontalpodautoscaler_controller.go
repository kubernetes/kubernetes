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

package autoscalercontroller

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/pkg/expapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util"

	heapster "k8s.io/heapster/api/v1/types"
)

const (
	heapsterNamespace = "kube-system"
	heapsterService   = "monitoring-heapster"
)

type HorizontalPodAutoscalerController struct {
	client    client.Interface
	expClient client.ExperimentalInterface
}

// Aggregates results into ResourceConsumption. Also returns number of
// pods included in the aggregation.
type metricAggregator func(heapster.MetricResultList) (expapi.ResourceConsumption, int)

type metricDefinition struct {
	name       string
	aggregator metricAggregator
}

var resourceDefinitions = map[api.ResourceName]metricDefinition{
	//TODO: add memory
	api.ResourceCPU: {"cpu-usage",
		func(metrics heapster.MetricResultList) (expapi.ResourceConsumption, int) {
			sum, count := calculateSumFromLatestSample(metrics)
			value := "0"
			if count > 0 {
				// assumes that cpu usage is in millis
				value = fmt.Sprintf("%dm", sum/uint64(count))
			}
			return expapi.ResourceConsumption{Resource: api.ResourceCPU, Quantity: resource.MustParse(value)}, count
		}},
}

var heapsterQueryStart, _ = time.ParseDuration("-5m")
var downscaleForbiddenWindow, _ = time.ParseDuration("20m")
var upscaleForbiddenWindow, _ = time.ParseDuration("3m")

func New(client client.Interface, expClient client.ExperimentalInterface) *HorizontalPodAutoscalerController {
	return &HorizontalPodAutoscalerController{
		client:    client,
		expClient: expClient,
	}
}

func (a *HorizontalPodAutoscalerController) Run(syncPeriod time.Duration) {
	go util.Until(func() {
		if err := a.reconcileAutoscalers(); err != nil {
			glog.Errorf("Couldn't reconcile horizontal pod autoscalers: %v", err)
		}
	}, syncPeriod, util.NeverStop)
}

func (a *HorizontalPodAutoscalerController) reconcileAutoscalers() error {
	ns := api.NamespaceAll
	list, err := a.expClient.HorizontalPodAutoscalers(ns).List(labels.Everything(), fields.Everything())
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	for _, hpa := range list.Items {
		reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Namespace, hpa.Spec.ScaleRef.Name)

		scale, err := a.expClient.Scales(hpa.Spec.ScaleRef.Namespace).Get(hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Name)
		if err != nil {
			glog.Warningf("Failed to query scale subresource for %s: %v", reference, err)
			continue
		}
		podList, err := a.client.Pods(hpa.Spec.ScaleRef.Namespace).
			List(labels.SelectorFromSet(labels.Set(scale.Status.Selector)), fields.Everything())

		if err != nil {
			glog.Warningf("Failed to get pod list for %s: %v", reference, err)
			continue
		}
		podNames := []string{}
		for _, pod := range podList.Items {
			podNames = append(podNames, pod.Name)
		}

		metricSpec, metricDefined := resourceDefinitions[hpa.Spec.Target.Resource]
		if !metricDefined {
			glog.Warningf("Heapster metric not defined for %s %v", reference, hpa.Spec.Target.Resource)
			continue
		}
		now := time.Now()

		startTime := now.Add(heapsterQueryStart)
		metricPath := fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s/metrics/%s",
			hpa.Spec.ScaleRef.Namespace,
			strings.Join(podNames, ","),
			metricSpec.name)

		resultRaw, err := a.client.Services(heapsterNamespace).
			ProxyGet(heapsterService, metricPath, map[string]string{"start": startTime.Format(time.RFC3339)}).
			DoRaw()

		if err != nil {
			glog.Warningf("Failed to get pods metrics for %s: %v", reference, err)
			continue
		}

		var metrics heapster.MetricResultList
		err = json.Unmarshal(resultRaw, &metrics)
		if err != nil {
			glog.Warningf("Failed to unmarshall heapster response: %v", err)
			continue
		}

		glog.Infof("Metrics available for %s: %s", reference, string(resultRaw))

		currentConsumption, count := metricSpec.aggregator(metrics)
		if count != len(podList.Items) {
			glog.Warningf("Metrics obtained for %d/%d of pods", count, len(podList.Items))
			continue
		}

		// if the ratio is 1.2 we want to have 2 replicas
		desiredReplicas := 1 + int((currentConsumption.Quantity.MilliValue()*int64(count))/hpa.Spec.Target.Quantity.MilliValue())

		if desiredReplicas < hpa.Spec.MinCount {
			desiredReplicas = hpa.Spec.MinCount
		}
		if desiredReplicas > hpa.Spec.MaxCount {
			desiredReplicas = hpa.Spec.MaxCount
		}

		rescale := false

		if desiredReplicas != count {
			// Going down
			if desiredReplicas < count && (hpa.Status == nil || hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(downscaleForbiddenWindow).Before(now)) {
				rescale = true
			}

			// Going up
			if desiredReplicas > count && (hpa.Status == nil || hpa.Status.LastScaleTimestamp == nil ||
				hpa.Status.LastScaleTimestamp.Add(upscaleForbiddenWindow).Before(now)) {
				rescale = true
			}

			if rescale {
				scale.Spec.Replicas = desiredReplicas
				_, err = a.expClient.Scales(hpa.Namespace).Update(hpa.Spec.ScaleRef.Kind, scale)
				if err != nil {
					glog.Warningf("Failed to rescale %s: %v", reference, err)
					continue
				}
			}
		}

		status := expapi.HorizontalPodAutoscalerStatus{
			CurrentReplicas:    count,
			DesiredReplicas:    desiredReplicas,
			CurrentConsumption: &currentConsumption,
		}
		hpa.Status = &status
		if rescale {
			now := util.NewTime(now)
			hpa.Status.LastScaleTimestamp = &now
		}

		_, err = a.expClient.HorizontalPodAutoscalers(hpa.Namespace).Update(&hpa)
		if err != nil {
			glog.Warningf("Failed to update HorizontalPodAutoscaler %s: %v", hpa.Name, err)
			continue
		}
	}
	return nil
}

func calculateSumFromLatestSample(metrics heapster.MetricResultList) (uint64, int) {
	sum := uint64(0)
	count := 0
	for _, metrics := range metrics.Items {
		var newest *heapster.MetricPoint
		newest = nil
		for _, metricPoint := range metrics.Metrics {
			if newest == nil || newest.Timestamp.Before(metricPoint.Timestamp) {
				newest = &metricPoint
			}
		}
		if newest != nil {
			sum += newest.Value
			count++
		}
	}
	return sum, count
}

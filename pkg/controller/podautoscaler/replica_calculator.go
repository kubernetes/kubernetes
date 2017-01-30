/*
Copyright 2016 The Kubernetes Authors.

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

package podautoscaler

import (
	"fmt"
	"math"
	"time"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/kubernetes/pkg/api/v1"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	metricsclient "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
)

type ReplicaCalculator struct {
	metricsClient metricsclient.MetricsClient
	podsGetter    v1core.PodsGetter
}

func NewReplicaCalculator(metricsClient metricsclient.MetricsClient, podsGetter v1core.PodsGetter) *ReplicaCalculator {
	return &ReplicaCalculator{
		metricsClient: metricsClient,
		podsGetter:    podsGetter,
	}
}

// GetResourceReplicas calculates the desired replica count based on a target resource utilization percentage
// of the given resource for pods matching the given selector in the given namespace, and the current replica count
func (c *ReplicaCalculator) GetResourceReplicas(currentReplicas int32, targetUtilization int32, resource v1.ResourceName, namespace string, selector labels.Selector) (replicaCount int32, utilization int32, timestamp time.Time, err error) {
	metrics, timestamp, err := c.metricsClient.GetResourceMetric(resource, namespace, selector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metrics for resource %s: %v", resource, err)
	}

	podList, err := c.podsGetter.Pods(namespace).List(v1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get pods while calculating replica count: %v", err)
	}

	if len(podList.Items) == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("no pods returned by selector while calculating replica count")
	}

	requests := make(map[string]int64, len(podList.Items))
	readyPodCount := 0
	unreadyPods := sets.NewString()
	missingPods := sets.NewString()

	for _, pod := range podList.Items {
		podSum := int64(0)
		for _, container := range pod.Spec.Containers {
			if containerRequest, ok := container.Resources.Requests[resource]; ok {
				podSum += containerRequest.MilliValue()
			} else {
				return 0, 0, time.Time{}, fmt.Errorf("missing request for %s on container %s in pod %s/%s", resource, container.Name, namespace, pod.Name)
			}
		}

		requests[pod.Name] = podSum

		if pod.Status.Phase != v1.PodRunning || !v1.IsPodReady(&pod) {
			// save this pod name for later, but pretend it doesn't exist for now
			unreadyPods.Insert(pod.Name)
			delete(metrics, pod.Name)
			continue
		}

		if _, found := metrics[pod.Name]; !found {
			// save this pod name for later, but pretend it doesn't exist for now
			missingPods.Insert(pod.Name)
			continue
		}

		readyPodCount++
	}

	if len(metrics) == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("did not receive metrics for any ready pods")
	}

	usageRatio, utilization, err := metricsclient.GetResourceUtilizationRatio(metrics, requests, targetUtilization)
	if err != nil {
		return 0, 0, time.Time{}, err
	}

	rebalanceUnready := len(unreadyPods) > 0 && usageRatio > 1.0
	if !rebalanceUnready && len(missingPods) == 0 {
		if math.Abs(1.0-usageRatio) <= tolerance {
			// return the current replicas if the change would be too small
			return currentReplicas, utilization, timestamp, nil
		}

		// if we don't have any unready or missing pods, we can calculate the new replica count now
		return int32(math.Ceil(usageRatio * float64(readyPodCount))), utilization, timestamp, nil
	}

	if len(missingPods) > 0 {
		if usageRatio < 1.0 {
			// on a scale-down, treat missing pods as using 100% of the resource request
			for podName := range missingPods {
				metrics[podName] = requests[podName]
			}
		} else if usageRatio > 1.0 {
			// on a scale-up, treat missing pods as using 0% of the resource request
			for podName := range missingPods {
				metrics[podName] = 0
			}
		}
	}

	if rebalanceUnready {
		// on a scale-up, treat unready pods as using 0% of the resource request
		for podName := range unreadyPods {
			metrics[podName] = 0
		}
	}

	// re-run the utilization calculation with our new numbers
	newUsageRatio, _, err := metricsclient.GetResourceUtilizationRatio(metrics, requests, targetUtilization)
	if err != nil {
		return 0, utilization, time.Time{}, err
	}

	if math.Abs(1.0-newUsageRatio) <= tolerance || (usageRatio < 1.0 && newUsageRatio > 1.0) || (usageRatio > 1.0 && newUsageRatio < 1.0) {
		// return the current replicas if the change would be too small,
		// or if the new usage ratio would cause a change in scale direction
		return currentReplicas, utilization, timestamp, nil
	}

	// return the result, where the number of replicas considered is
	// however many replicas factored into our calculation
	return int32(math.Ceil(newUsageRatio * float64(len(metrics)))), utilization, timestamp, nil
}

// GetMetricReplicas calculates the desired replica count based on a target resource utilization percentage
// of the given resource for pods matching the given selector in the given namespace, and the current replica count
func (c *ReplicaCalculator) GetMetricReplicas(currentReplicas int32, targetUtilization float64, metricName string, namespace string, selector labels.Selector) (replicaCount int32, utilization float64, timestamp time.Time, err error) {
	metrics, timestamp, err := c.metricsClient.GetRawMetric(metricName, namespace, selector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metric  %s: %v", metricName, err)
	}

	podList, err := c.podsGetter.Pods(namespace).List(v1.ListOptions{LabelSelector: selector.String()})
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get pods while calculating replica count: %v", err)
	}

	if len(podList.Items) == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("no pods returned by selector while calculating replica count")
	}

	readyPodCount := 0
	unreadyPods := sets.NewString()
	missingPods := sets.NewString()

	for _, pod := range podList.Items {
		if pod.Status.Phase != v1.PodRunning || !v1.IsPodReady(&pod) {
			// save this pod name for later, but pretend it doesn't exist for now
			unreadyPods.Insert(pod.Name)
			delete(metrics, pod.Name)
			continue
		}

		if _, found := metrics[pod.Name]; !found {
			// save this pod name for later, but pretend it doesn't exist for now
			missingPods.Insert(pod.Name)
			continue
		}

		readyPodCount++
	}

	if len(metrics) == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("did not recieve metrics for any ready pods")
	}

	usageRatio, utilization := metricsclient.GetMetricUtilizationRatio(metrics, targetUtilization)
	if err != nil {
		return 0, 0, time.Time{}, err
	}

	rebalanceUnready := len(unreadyPods) > 0 && usageRatio > 1.0

	if !rebalanceUnready && len(missingPods) == 0 {
		if math.Abs(1.0-usageRatio) <= tolerance {
			// return the current replicas if the change would be too small
			return currentReplicas, utilization, timestamp, nil
		}

		// if we don't have any unready or missing pods, we can calculate the new replica count now
		return int32(math.Ceil(usageRatio * float64(readyPodCount))), utilization, timestamp, nil
	}

	if len(missingPods) > 0 {
		if usageRatio < 1.0 {
			// on a scale-down, treat missing pods as using 100% of the resource request
			for podName := range missingPods {
				metrics[podName] = targetUtilization
			}
		} else {
			// on a scale-up, treat missing pods as using 0% of the resource request
			for podName := range missingPods {
				metrics[podName] = 0
			}
		}
	}

	if rebalanceUnready {
		// on a scale-up, treat unready pods as using 0% of the resource request
		for podName := range unreadyPods {
			metrics[podName] = 0
		}
	}

	// re-run the utilization calculation with our new numbers
	newUsageRatio, _ := metricsclient.GetMetricUtilizationRatio(metrics, targetUtilization)
	if err != nil {
		return 0, utilization, time.Time{}, err
	}

	if math.Abs(1.0-newUsageRatio) <= tolerance || (usageRatio < 1.0 && newUsageRatio > 1.0) || (usageRatio > 1.0 && newUsageRatio < 1.0) {
		// return the current replicas if the change would be too small,
		// or if the new usage ratio would cause a change in scale direction
		return currentReplicas, utilization, timestamp, nil
	}

	// return the result, where the number of replicas considered is
	// however many replicas factored into our calculation
	return int32(math.Ceil(newUsageRatio * float64(len(metrics)))), utilization, timestamp, nil
}

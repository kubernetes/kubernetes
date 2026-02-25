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
	"context"
	"fmt"
	"math"
	"time"

	autoscaling "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	resourcehelpers "k8s.io/component-helpers/resource"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	metricsclient "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// defaultTestingTolerance is default value for calculating when to
	// scale up/scale down.
	defaultTestingTolerance                     = 0.1
	defaultTestingCPUInitializationPeriod       = 2 * time.Minute
	defaultTestingDelayOfInitialReadinessStatus = 10 * time.Second
)

// Tolerances contains metric usage ratio scale-up and scale-down tolerances.
type Tolerances struct {
	scaleDown float64
	scaleUp   float64
}

func (t Tolerances) String() string {
	return fmt.Sprintf("[down:%.1f%%, up:%.1f%%]", t.scaleDown*100., t.scaleUp*100.)
}

func (t Tolerances) isWithin(usageRatio float64) bool {
	return (1.0-t.scaleDown) <= usageRatio && usageRatio <= (1.0+t.scaleUp)
}

// ReplicaCalculator bundles all needed information to calculate the target amount of replicas
type ReplicaCalculator struct {
	metricsClient                 metricsclient.MetricsClient
	podLister                     corelisters.PodLister
	cpuInitializationPeriod       time.Duration
	delayOfInitialReadinessStatus time.Duration
}

// NewReplicaCalculator creates a new ReplicaCalculator and passes all necessary information to the new instance
func NewReplicaCalculator(metricsClient metricsclient.MetricsClient, podLister corelisters.PodLister, cpuInitializationPeriod, delayOfInitialReadinessStatus time.Duration) *ReplicaCalculator {
	return &ReplicaCalculator{
		metricsClient:                 metricsClient,
		podLister:                     podLister,
		cpuInitializationPeriod:       cpuInitializationPeriod,
		delayOfInitialReadinessStatus: delayOfInitialReadinessStatus,
	}
}

// GetResourceReplicas calculates the desired replica count based on a target resource utilization percentage
// of the given resource for pods matching the given selector in the given namespace, and the current replica count
func (c *ReplicaCalculator) GetResourceReplicas(ctx context.Context, currentReplicas int32, targetUtilization int32, resource v1.ResourceName, tolerances Tolerances, namespace string, selector labels.Selector, container string) (replicaCount int32, utilization int32, rawUtilization int64, timestamp time.Time, err error) {
	metrics, timestamp, err := c.metricsClient.GetResourceMetric(ctx, resource, namespace, selector, container)
	if err != nil {
		return 0, 0, 0, time.Time{}, fmt.Errorf("unable to get metrics for resource %s: %v", resource, err)
	}
	podList, err := c.podLister.Pods(namespace).List(selector)
	if err != nil {
		return 0, 0, 0, time.Time{}, fmt.Errorf("unable to get pods while calculating replica count: %v", err)
	}
	if len(podList) == 0 {
		return 0, 0, 0, time.Time{}, fmt.Errorf("no pods returned by selector while calculating replica count")
	}

	readyPodCount, unreadyPods, missingPods, ignoredPods := groupPods(podList, metrics, resource, c.cpuInitializationPeriod, c.delayOfInitialReadinessStatus)
	removeMetricsForPods(metrics, ignoredPods)
	removeMetricsForPods(metrics, unreadyPods)
	if len(metrics) == 0 {
		return 0, 0, 0, time.Time{}, fmt.Errorf("did not receive metrics for targeted pods (pods might be unready)")
	}

	requests, err := calculateRequests(podList, container, resource)
	if err != nil {
		return 0, 0, 0, time.Time{}, err
	}

	usageRatio, utilization, rawUtilization, err := metricsclient.GetResourceUtilizationRatio(metrics, requests, targetUtilization)
	if err != nil {
		return 0, 0, 0, time.Time{}, err
	}

	scaleUpWithUnready := len(unreadyPods) > 0 && usageRatio > 1.0
	if !scaleUpWithUnready && len(missingPods) == 0 {
		if tolerances.isWithin(usageRatio) {
			// return the current replicas if the change would be too small
			return currentReplicas, utilization, rawUtilization, timestamp, nil
		}

		// if we don't have any unready or missing pods, we can calculate the new replica count now
		return int32(math.Ceil(usageRatio * float64(readyPodCount))), utilization, rawUtilization, timestamp, nil
	}

	if len(missingPods) > 0 {
		if usageRatio < 1.0 {
			// on a scale-down, treat missing pods as using 100% (all) of the resource request
			// or the utilization target for targets higher than 100%
			fallbackUtilization := int64(max(100, targetUtilization))
			for podName := range missingPods {
				metrics[podName] = metricsclient.PodMetric{Value: requests[podName] * fallbackUtilization / 100}
			}
		} else if usageRatio > 1.0 {
			// on a scale-up, treat missing pods as using 0% of the resource request
			for podName := range missingPods {
				metrics[podName] = metricsclient.PodMetric{Value: 0}
			}
		}
	}

	if scaleUpWithUnready {
		// on a scale-up, treat unready pods as using 0% of the resource request
		for podName := range unreadyPods {
			metrics[podName] = metricsclient.PodMetric{Value: 0}
		}
	}

	// re-run the utilization calculation with our new numbers
	newUsageRatio, _, _, err := metricsclient.GetResourceUtilizationRatio(metrics, requests, targetUtilization)
	if err != nil {
		return 0, utilization, rawUtilization, time.Time{}, err
	}

	if tolerances.isWithin(newUsageRatio) || (usageRatio < 1.0 && newUsageRatio > 1.0) || (usageRatio > 1.0 && newUsageRatio < 1.0) {
		// return the current replicas if the change would be too small,
		// or if the new usage ratio would cause a change in scale direction
		return currentReplicas, utilization, rawUtilization, timestamp, nil
	}

	newReplicas := int32(math.Ceil(newUsageRatio * float64(len(metrics))))
	if (newUsageRatio < 1.0 && newReplicas > currentReplicas) || (newUsageRatio > 1.0 && newReplicas < currentReplicas) {
		// return the current replicas if the change of metrics length would cause a change in scale direction
		return currentReplicas, utilization, rawUtilization, timestamp, nil
	}

	// return the result, where the number of replicas considered is
	// however many replicas factored into our calculation
	return newReplicas, utilization, rawUtilization, timestamp, nil
}

// GetRawResourceReplicas calculates the desired replica count based on a target resource usage (as a raw milli-value)
// for pods matching the given selector in the given namespace, and the current replica count
func (c *ReplicaCalculator) GetRawResourceReplicas(ctx context.Context, currentReplicas int32, targetUsage int64, resource v1.ResourceName, tolerances Tolerances, namespace string, selector labels.Selector, container string) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	metrics, timestamp, err := c.metricsClient.GetResourceMetric(ctx, resource, namespace, selector, container)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metrics for resource %s: %v", resource, err)
	}

	replicaCount, usage, err = c.calcPlainMetricReplicas(metrics, currentReplicas, targetUsage, tolerances, namespace, selector, resource)
	return replicaCount, usage, timestamp, err
}

// GetMetricReplicas calculates the desired replica count based on a target metric usage
// (as a milli-value) for pods matching the given selector in the given namespace, and the
// current replica count
func (c *ReplicaCalculator) GetMetricReplicas(currentReplicas int32, targetUsage int64, metricName string, tolerances Tolerances, namespace string, selector labels.Selector, metricSelector labels.Selector) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	metrics, timestamp, err := c.metricsClient.GetRawMetric(metricName, namespace, selector, metricSelector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metric %s: %v", metricName, err)
	}

	replicaCount, usage, err = c.calcPlainMetricReplicas(metrics, currentReplicas, targetUsage, tolerances, namespace, selector, "")
	return replicaCount, usage, timestamp, err
}

// calcPlainMetricReplicas calculates the desired replicas for plain (i.e. non-utilization percentage) metrics.
func (c *ReplicaCalculator) calcPlainMetricReplicas(metrics metricsclient.PodMetricsInfo, currentReplicas int32, targetUsage int64, tolerances Tolerances, namespace string, selector labels.Selector, resource v1.ResourceName) (replicaCount int32, usage int64, err error) {

	podList, err := c.podLister.Pods(namespace).List(selector)
	if err != nil {
		return 0, 0, fmt.Errorf("unable to get pods while calculating replica count: %v", err)
	}

	if len(podList) == 0 {
		return 0, 0, fmt.Errorf("no pods returned by selector while calculating replica count")
	}

	readyPodCount, unreadyPods, missingPods, ignoredPods := groupPods(podList, metrics, resource, c.cpuInitializationPeriod, c.delayOfInitialReadinessStatus)
	removeMetricsForPods(metrics, ignoredPods)
	removeMetricsForPods(metrics, unreadyPods)

	if len(metrics) == 0 {
		return 0, 0, fmt.Errorf("did not receive metrics for targeted pods (pods might be unready)")
	}

	usageRatio, usage := metricsclient.GetMetricUsageRatio(metrics, targetUsage)

	scaleUpWithUnready := len(unreadyPods) > 0 && usageRatio > 1.0

	if !scaleUpWithUnready && len(missingPods) == 0 {
		if tolerances.isWithin(usageRatio) {
			// return the current replicas if the change would be too small
			return currentReplicas, usage, nil
		}

		// if we don't have any unready or missing pods, we can calculate the new replica count now
		return int32(math.Ceil(usageRatio * float64(readyPodCount))), usage, nil
	}

	if len(missingPods) > 0 {
		if usageRatio < 1.0 {
			// on a scale-down, treat missing pods as using exactly the target amount
			for podName := range missingPods {
				metrics[podName] = metricsclient.PodMetric{Value: targetUsage}
			}
		} else if usageRatio > 1.0 {
			// on a scale-up, treat missing pods as using 0% of the resource request
			for podName := range missingPods {
				metrics[podName] = metricsclient.PodMetric{Value: 0}
			}
		}
	}

	if scaleUpWithUnready {
		// on a scale-up, treat unready pods as using 0% of the resource request
		for podName := range unreadyPods {
			metrics[podName] = metricsclient.PodMetric{Value: 0}
		}
	}

	// re-run the usage calculation with our new numbers
	newUsageRatio, _ := metricsclient.GetMetricUsageRatio(metrics, targetUsage)

	if tolerances.isWithin(newUsageRatio) || (usageRatio < 1.0 && newUsageRatio > 1.0) || (usageRatio > 1.0 && newUsageRatio < 1.0) {
		// return the current replicas if the change would be too small,
		// or if the new usage ratio would cause a change in scale direction
		return currentReplicas, usage, nil
	}

	newReplicas := int32(math.Ceil(newUsageRatio * float64(len(metrics))))
	if (newUsageRatio < 1.0 && newReplicas > currentReplicas) || (newUsageRatio > 1.0 && newReplicas < currentReplicas) {
		// return the current replicas if the change of metrics length would cause a change in scale direction
		return currentReplicas, usage, nil
	}

	// return the result, where the number of replicas considered is
	// however many replicas factored into our calculation
	return newReplicas, usage, nil
}

// GetObjectMetricReplicas calculates the desired replica count based on a target metric usage (as a milli-value)
// for the given object in the given namespace, and the current replica count.
func (c *ReplicaCalculator) GetObjectMetricReplicas(currentReplicas int32, targetUsage int64, metricName string, tolerances Tolerances, namespace string, objectRef *autoscaling.CrossVersionObjectReference, selector labels.Selector, metricSelector labels.Selector) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	usage, _, err = c.metricsClient.GetObjectMetric(metricName, namespace, objectRef, metricSelector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metric %s: %v on %s %s/%s", metricName, objectRef.Kind, namespace, objectRef.Name, err)
	}

	usageRatio := float64(usage) / float64(targetUsage)
	replicaCount, timestamp, err = c.getUsageRatioReplicaCount(currentReplicas, usageRatio, tolerances, namespace, selector)
	return replicaCount, usage, timestamp, err
}

// getUsageRatioReplicaCount calculates the desired replica count based on usageRatio and ready pods count.
// For currentReplicas=0 doesn't take into account ready pods count and tolerance to support scaling to zero pods.
func (c *ReplicaCalculator) getUsageRatioReplicaCount(currentReplicas int32, usageRatio float64, tolerances Tolerances, namespace string, selector labels.Selector) (replicaCount int32, timestamp time.Time, err error) {
	if currentReplicas != 0 {
		if tolerances.isWithin(usageRatio) {
			// return the current replicas if the change would be too small
			return currentReplicas, timestamp, nil
		}
		readyPodCount := int64(0)
		readyPodCount, err = c.getReadyPodsCount(namespace, selector)
		if err != nil {
			return 0, time.Time{}, fmt.Errorf("unable to calculate ready pods: %s", err)
		}
		// Calculate replicaCount as float64 first
		replicaCountFloat := usageRatio * float64(readyPodCount)
		// Check if replicaCount exceeds max int32
		if replicaCountFloat > math.MaxInt32 {
			replicaCount = math.MaxInt32
		} else {
			replicaCount = int32(math.Ceil(replicaCountFloat))
		}
	} else {
		// Scale to zero or n pods depending on usageRatio
		replicaCount = int32(math.Ceil(usageRatio))
	}

	return replicaCount, timestamp, err
}

// GetObjectPerPodMetricReplicas calculates the desired replica count based on a target metric usage (as a milli-value)
// for the given object in the given namespace, and the current replica count.
func (c *ReplicaCalculator) GetObjectPerPodMetricReplicas(statusReplicas int32, targetAverageUsage int64, metricName string, tolerances Tolerances, namespace string, objectRef *autoscaling.CrossVersionObjectReference, metricSelector labels.Selector) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	usage, timestamp, err = c.metricsClient.GetObjectMetric(metricName, namespace, objectRef, metricSelector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get metric %s: %v on %s %s/%s", metricName, objectRef.Kind, namespace, objectRef.Name, err)
	}

	replicaCount = statusReplicas
	usageRatio := float64(usage) / (float64(targetAverageUsage) * float64(replicaCount))
	if !tolerances.isWithin(usageRatio) {
		// update number of replicas if change is large enough
		replicaCount = int32(math.Ceil(float64(usage) / float64(targetAverageUsage)))
	}
	usage = int64(math.Ceil(float64(usage) / float64(statusReplicas)))
	return replicaCount, usage, timestamp, nil
}

// @TODO(mattjmcnaughton) Many different functions in this module use variations
// of this function. Make this function generic, so we don't repeat the same
// logic in multiple places.
func (c *ReplicaCalculator) getReadyPodsCount(namespace string, selector labels.Selector) (int64, error) {
	podList, err := c.podLister.Pods(namespace).List(selector)
	if err != nil {
		return 0, fmt.Errorf("unable to get pods while calculating replica count: %v", err)
	}

	if len(podList) == 0 {
		return 0, fmt.Errorf("no pods returned by selector while calculating replica count")
	}

	readyPodCount := 0

	for _, pod := range podList {
		if pod.Status.Phase == v1.PodRunning && podutil.IsPodReady(pod) {
			readyPodCount++
		}
	}

	return int64(readyPodCount), nil
}

// GetExternalMetricReplicas calculates the desired replica count based on a
// target metric value (as a milli-value) for the external metric in the given
// namespace, and the current replica count.
func (c *ReplicaCalculator) GetExternalMetricReplicas(currentReplicas int32, targetUsage int64, metricName string, tolerances Tolerances, namespace string, metricSelector *metav1.LabelSelector, podSelector labels.Selector) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	metricLabelSelector, err := metav1.LabelSelectorAsSelector(metricSelector)
	if err != nil {
		return 0, 0, time.Time{}, err
	}
	metrics, _, err := c.metricsClient.GetExternalMetric(metricName, namespace, metricLabelSelector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get external metric %s/%s/%+v: %s", namespace, metricName, metricSelector, err)
	}

	usage = 0
	for _, val := range metrics {
		// Cap at MaxInt64 for positive overflow
		if val > 0 && usage > math.MaxInt64-val {
			usage = math.MaxInt64
			break
		}
		usage = usage + val
	}

	usageRatio := float64(usage) / float64(targetUsage)
	replicaCount, timestamp, err = c.getUsageRatioReplicaCount(currentReplicas, usageRatio, tolerances, namespace, podSelector)
	return replicaCount, usage, timestamp, err
}

// GetExternalPerPodMetricReplicas calculates the desired replica count based on a
// target metric value per pod (as a milli-value) for the external metric in the
// given namespace, and the current replica count.
func (c *ReplicaCalculator) GetExternalPerPodMetricReplicas(statusReplicas int32, targetUsagePerPod int64, metricName string, tolerances Tolerances, namespace string, metricSelector *metav1.LabelSelector) (replicaCount int32, usage int64, timestamp time.Time, err error) {
	metricLabelSelector, err := metav1.LabelSelectorAsSelector(metricSelector)
	if err != nil {
		return 0, 0, time.Time{}, err
	}
	metrics, timestamp, err := c.metricsClient.GetExternalMetric(metricName, namespace, metricLabelSelector)
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("unable to get external metric %s/%s/%+v: %s", namespace, metricName, metricSelector, err)
	}
	usage = 0
	for _, val := range metrics {
		usage = usage + val
		if usage < 0 {
			// the only way we would ever get here is because of an Int64 overflow
			usage = math.MaxInt64
			break
		}
	}

	replicaCount = statusReplicas
	usageRatio := float64(usage) / (float64(targetUsagePerPod) * float64(replicaCount))
	if !tolerances.isWithin(usageRatio) {
		// update number of replicas if the change is large enough
		replicaCountResult := math.Ceil(float64(usage) / float64(targetUsagePerPod))
		// Ensure that the result exceeds the bounds of an int32
		if replicaCountResult > float64(math.MaxInt32) {
			replicaCount = math.MaxInt32
		} else {
			replicaCount = int32(replicaCountResult)
		}
	}
	// Handle usage overflow cases
	if float64(usage) >= float64(math.MaxInt64) {
		usage = math.MaxInt64
	} else {
		usage = int64(math.Ceil(float64(usage) / float64(statusReplicas)))
	}
	return replicaCount, usage, timestamp, nil
}

func groupPods(pods []*v1.Pod, metrics metricsclient.PodMetricsInfo, resource v1.ResourceName, cpuInitializationPeriod, delayOfInitialReadinessStatus time.Duration) (readyPodCount int, unreadyPods, missingPods, ignoredPods sets.Set[string]) {
	missingPods = sets.New[string]()
	unreadyPods = sets.New[string]()
	ignoredPods = sets.New[string]()
	for _, pod := range pods {
		if pod.DeletionTimestamp != nil || pod.Status.Phase == v1.PodFailed {
			ignoredPods.Insert(pod.Name)
			continue
		}
		// Pending pods are ignored.
		if pod.Status.Phase == v1.PodPending {
			unreadyPods.Insert(pod.Name)
			continue
		}
		// Pods missing metrics.
		metric, found := metrics[pod.Name]
		if !found {
			missingPods.Insert(pod.Name)
			continue
		}
		// Unready pods are ignored.
		if resource == v1.ResourceCPU {
			var unready bool
			_, condition := podutil.GetPodCondition(&pod.Status, v1.PodReady)
			if condition == nil || pod.Status.StartTime == nil {
				unready = true
			} else {
				// Pod still within possible initialisation period.
				if pod.Status.StartTime.Add(cpuInitializationPeriod).After(time.Now()) {
					// Ignore sample if pod is unready or one window of metric wasn't collected since last state transition.
					unready = condition.Status == v1.ConditionFalse || metric.Timestamp.Before(condition.LastTransitionTime.Time.Add(metric.Window))
				} else {
					// Ignore metric if pod is unready and it has never been ready.
					unready = condition.Status == v1.ConditionFalse && pod.Status.StartTime.Add(delayOfInitialReadinessStatus).After(condition.LastTransitionTime.Time)
				}
			}
			if unready {
				unreadyPods.Insert(pod.Name)
				continue
			}
		}
		readyPodCount++
	}
	return
}

// calculateRequests computes the request value for each pod for the specified
// resource.
// If container is non-empty, it uses the request of that specific container.
// If container is empty, it uses pod-level requests if pod-level requests are
// set on the pod. Otherwise, it sums the requests of all containers in the pod
// (including restartable init containers).
// It returns a map of pod names to their calculated request values.
func calculateRequests(pods []*v1.Pod, container string, resource v1.ResourceName) (map[string]int64, error) {
	podLevelResourcesEnabled := feature.DefaultFeatureGate.Enabled(features.PodLevelResources)
	requests := make(map[string]int64, len(pods))
	for _, pod := range pods {
		var request int64
		var err error
		// Determine if we should use pod-level requests: see KEP-2837
		// https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/2837-pod-level-resource-spec/README.md
		usePodLevelRequests := podLevelResourcesEnabled &&
			resourcehelpers.IsPodLevelRequestsSet(pod) &&
			// If a container name is specified in the HPA, it takes precedence over
			// the pod-level requests.
			container == ""

		if usePodLevelRequests {
			request, err = calculatePodLevelRequests(pod, resource)
		} else {
			request, err = calculatePodRequestsFromContainers(pod, container, resource)
		}
		if err != nil {
			return nil, err
		}
		requests[pod.Name] = request
	}
	return requests, nil
}

// calculatePodLevelRequests computes the requests for the specific resource at
// the pod level.
func calculatePodLevelRequests(pod *v1.Pod, resource v1.ResourceName) (int64, error) {
	podLevelRequests := resourcehelpers.PodRequests(pod, resourcehelpers.PodResourcesOptions{})
	podRequest, ok := podLevelRequests[resource]
	if !ok {
		return 0, fmt.Errorf("missing pod-level request for %s in Pod %s", resource, pod.Name)
	}
	return podRequest.MilliValue(), nil
}

// calculatePodRequestsFromContainers computes the requests for the specified
// resource by summing requests from all containers in the pod.
// If a container name is specified, it uses only that container.
func calculatePodRequestsFromContainers(pod *v1.Pod, container string, resource v1.ResourceName) (int64, error) {
	containers := append([]v1.Container{}, pod.Spec.Containers...)
	for _, c := range pod.Spec.InitContainers {
		if c.RestartPolicy != nil && *c.RestartPolicy == v1.ContainerRestartPolicyAlways {
			containers = append(containers, c)
		}
	}

	request := int64(0)
	for _, c := range containers {
		if container == "" || container == c.Name {
			containerRequest, ok := c.Resources.Requests[resource]
			if !ok {
				return 0, fmt.Errorf("missing request for %s in container %s of Pod %s", resource, c.Name, pod.Name)
			}
			request += containerRequest.MilliValue()
		}
		// container names are unique inside the pod
		if container == c.Name {
			return request, nil
		}
	}

	// If we're looking for a specific container and didn't find it
	if container != "" {
		return 0, fmt.Errorf("container %s not found in Pod %s", container, pod.Name)
	}

	return request, nil
}

func removeMetricsForPods(metrics metricsclient.PodMetricsInfo, pods sets.Set[string]) {
	for _, pod := range pods.UnsortedList() {
		delete(metrics, pod)
	}
}

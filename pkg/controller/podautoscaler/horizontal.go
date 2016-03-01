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

package podautoscaler

import (
	"encoding/json"
	"fmt"
	"math"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/record"
	unversionedcore "k8s.io/kubernetes/pkg/client/typed/generated/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/typed/generated/extensions/unversioned"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/util/wait"
)

const (
	// Usage shoud exceed the tolerance before we start downscale or upscale the pods.
	// TODO: make it a flag or HPA spec element.
	tolerance = 0.1

	HpaCustomMetricsTargetAnnotationName = "alpha/target.custom-metrics.podautoscaler.kubernetes.io"
	HpaCustomMetricsStatusAnnotationName = "alpha/status.custom-metrics.podautoscaler.kubernetes.io"
)

type HorizontalController struct {
	scaleNamespacer unversionedextensions.ScalesGetter
	hpaNamespacer   unversionedextensions.HorizontalPodAutoscalersGetter

	metricsClient metrics.MetricsClient
	eventRecorder record.EventRecorder
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func NewHorizontalController(evtNamespacer unversionedcore.EventsGetter, scaleNamespacer unversionedextensions.ScalesGetter, hpaNamespacer unversionedextensions.HorizontalPodAutoscalersGetter, metricsClient metrics.MetricsClient) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "horizontal-pod-autoscaler"})

	return &HorizontalController{
		metricsClient:   metricsClient,
		eventRecorder:   recorder,
		scaleNamespacer: scaleNamespacer,
		hpaNamespacer:   hpaNamespacer,
	}
}

func (a *HorizontalController) Run(syncPeriod time.Duration) {
	go wait.Until(func() {
		if err := a.reconcileAutoscalers(); err != nil {
			glog.Errorf("Couldn't reconcile horizontal pod autoscalers: %v", err)
		}
	}, syncPeriod, wait.NeverStop)
}

func (a *HorizontalController) computeReplicasForCPUUtilization(hpa extensions.HorizontalPodAutoscaler, scale *extensions.Scale) (int, *int, time.Time, error) {
	if hpa.Spec.CPUUtilization == nil {
		// If CPUTarget is not specified than we should return some default values.
		// Since we always take maximum number of replicas from all policies it is safe
		// to just return 0.
		return 0, nil, time.Time{}, nil
	}
	currentReplicas := scale.Status.Replicas
	currentUtilization, timestamp, err := a.metricsClient.GetCPUUtilization(hpa.Namespace, scale.Status.Selector)

	// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
	if err != nil {
		a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedGetMetrics", err.Error())
		return 0, nil, time.Time{}, fmt.Errorf("failed to get cpu utilization: %v", err)
	}

	usageRatio := float64(*currentUtilization) / float64(hpa.Spec.CPUUtilization.TargetPercentage)
	if math.Abs(1.0-usageRatio) > tolerance {
		return int(math.Ceil(usageRatio * float64(currentReplicas))), currentUtilization, timestamp, nil
	} else {
		return currentReplicas, currentUtilization, timestamp, nil
	}
}

// Computes the desired number of replicas based on the CustomMetrics passed in cmAnnotation as json-serialized
// extensions.CustomMetricsTargetList.
// Returns number of replicas, status string (also json-serialized extensions.CustomMetricsCurrentStatusList),
// last timestamp of the metrics involved in computations or error, if occurred.
func (a *HorizontalController) computeReplicasForCustomMetrics(hpa extensions.HorizontalPodAutoscaler, scale *extensions.Scale,
	cmAnnotation string) (int, string, time.Time, error) {

	currentReplicas := scale.Status.Replicas
	replicas := 0
	timestamp := time.Time{}

	if cmAnnotation == "" {
		return 0, "", time.Time{}, nil
	}

	var targetList extensions.CustomMetricTargetList
	if err := json.Unmarshal([]byte(cmAnnotation), &targetList); err != nil {
		return 0, "", time.Time{}, fmt.Errorf("failed to parse custom metrics annotation: %v", err)
	}
	if len(targetList.Items) == 0 {
		return 0, "", time.Time{}, fmt.Errorf("no custom metrics in annotation")
	}

	statusList := extensions.CustomMetricCurrentStatusList{
		Items: make([]extensions.CustomMetricCurrentStatus, 0),
	}

	for _, customMetricTarget := range targetList.Items {
		value, currentTimestamp, err := a.metricsClient.GetCustomMetric(customMetricTarget.Name, hpa.Namespace, scale.Status.Selector)
		// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
		if err != nil {
			a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedGetCustomMetrics", err.Error())
			return 0, "", time.Time{}, fmt.Errorf("failed to get custom metric value: %v", err)
		}
		floatTarget := float64(customMetricTarget.TargetValue.MilliValue()) / 1000.0
		usageRatio := *value / floatTarget

		replicaCountProposal := 0
		if math.Abs(1.0-usageRatio) > tolerance {
			replicaCountProposal = int(math.Ceil(usageRatio * float64(currentReplicas)))
		} else {
			replicaCountProposal = currentReplicas
		}
		if replicaCountProposal > replicas {
			timestamp = currentTimestamp
			replicas = replicaCountProposal
		}
		quantity, err := resource.ParseQuantity(fmt.Sprintf("%.3f", *value))
		if err != nil {
			return 0, "", time.Time{}, fmt.Errorf("failed to set custom metric value: %v", err)
		}
		statusList.Items = append(statusList.Items, extensions.CustomMetricCurrentStatus{
			Name:         customMetricTarget.Name,
			CurrentValue: *quantity,
		})
	}
	byteStatusList, err := json.Marshal(statusList)
	if err != nil {
		return 0, "", time.Time{}, fmt.Errorf("failed to serialize custom metric status: %v", err)
	}

	return replicas, string(byteStatusList), timestamp, nil
}

func (a *HorizontalController) reconcileAutoscaler(hpa extensions.HorizontalPodAutoscaler) error {
	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleRef.Kind, hpa.Namespace, hpa.Spec.ScaleRef.Name)

	scale, err := a.scaleNamespacer.Scales(hpa.Namespace).Get(hpa.Spec.ScaleRef.Kind, hpa.Spec.ScaleRef.Name)
	if err != nil {
		a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedGetScale", err.Error())
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas

	cpuDesiredReplicas := 0
	var cpuCurrentUtilization *int = nil
	cpuTimestamp := time.Time{}

	cmDesiredReplicas := 0
	cmStatus := ""
	cmTimestamp := time.Time{}

	desiredReplicas := 0
	timestamp := time.Now()

	if currentReplicas > hpa.Spec.MaxReplicas {
		desiredReplicas = hpa.Spec.MaxReplicas
	} else if hpa.Spec.MinReplicas != nil && currentReplicas < *hpa.Spec.MinReplicas {
		desiredReplicas = *hpa.Spec.MinReplicas
	} else if currentReplicas == 0 {
		desiredReplicas = 1
	} else {
		// All basic scenarios covered, the state should be sane, lets use metrics.

		if hpa.Spec.CPUUtilization != nil {
			cpuDesiredReplicas, cpuCurrentUtilization, cpuTimestamp, err = a.computeReplicasForCPUUtilization(hpa, scale)
			if err != nil {
				a.updateCurrentReplicasInStatus(hpa, currentReplicas)
				a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedComputeReplicas", err.Error())
				return fmt.Errorf("failed to compute desired number of replicas based on CPU utilization for %s: %v", reference, err)
			}
		}

		if cmAnnotation, cmAnnotationFound := hpa.Annotations[HpaCustomMetricsTargetAnnotationName]; cmAnnotationFound {
			cmDesiredReplicas, cmStatus, cmTimestamp, err = a.computeReplicasForCustomMetrics(hpa, scale, cmAnnotation)
			if err != nil {
				a.updateCurrentReplicasInStatus(hpa, currentReplicas)
				a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedComputeCMReplicas", err.Error())
				return fmt.Errorf("failed to compute desired number of replicas based on Custom Metrics for %s: %v", reference, err)
			}
		}

		if cpuDesiredReplicas > desiredReplicas {
			desiredReplicas = cpuDesiredReplicas
			timestamp = cpuTimestamp
		}
		if cmDesiredReplicas > desiredReplicas {
			desiredReplicas = cmDesiredReplicas
			timestamp = cmTimestamp
		}

		if hpa.Spec.MinReplicas != nil && desiredReplicas < *hpa.Spec.MinReplicas {
			desiredReplicas = *hpa.Spec.MinReplicas
		}

		// TODO: remove when pod idling is done.
		if desiredReplicas == 0 {
			desiredReplicas = 1
		}

		if desiredReplicas > hpa.Spec.MaxReplicas {
			desiredReplicas = hpa.Spec.MaxReplicas
		}
	}

	rescale := shouldScale(hpa, currentReplicas, desiredReplicas, timestamp)
	if rescale {
		scale.Spec.Replicas = desiredReplicas
		_, err = a.scaleNamespacer.Scales(hpa.Namespace).Update(hpa.Spec.ScaleRef.Kind, scale)
		if err != nil {
			a.eventRecorder.Eventf(&hpa, api.EventTypeWarning, "FailedRescale", "New size: %d; error: %v", desiredReplicas, err.Error())
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		a.eventRecorder.Eventf(&hpa, api.EventTypeNormal, "SuccessfulRescale", "New size: %d", desiredReplicas)
		glog.Infof("Successfull rescale of %s, old size: %d, new size: %d",
			hpa.Name, currentReplicas, desiredReplicas)
	} else {
		desiredReplicas = currentReplicas
	}

	return a.updateStatus(hpa, currentReplicas, desiredReplicas, cpuCurrentUtilization, cmStatus, rescale)
}

func shouldScale(hpa extensions.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int, timestamp time.Time) bool {
	if desiredReplicas != currentReplicas {
		// Going down only if the usageRatio dropped significantly below the target
		// and there was no rescaling in the last downscaleForbiddenWindow.
		if desiredReplicas < currentReplicas &&
			(hpa.Status.LastScaleTime == nil ||
				hpa.Status.LastScaleTime.Add(downscaleForbiddenWindow).Before(timestamp)) {
			return true
		}

		// Going up only if the usage ratio increased significantly above the target
		// and there was no rescaling in the last upscaleForbiddenWindow.
		if desiredReplicas > currentReplicas &&
			(hpa.Status.LastScaleTime == nil ||
				hpa.Status.LastScaleTime.Add(upscaleForbiddenWindow).Before(timestamp)) {
			return true
		}
	}
	return false
}

func (a *HorizontalController) updateCurrentReplicasInStatus(hpa extensions.HorizontalPodAutoscaler, currentReplicas int) {
	err := a.updateStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentCPUUtilizationPercentage, hpa.Annotations[HpaCustomMetricsStatusAnnotationName], false)
	if err != nil {
		glog.Errorf("%v", err)
	}
}

func (a *HorizontalController) updateStatus(hpa extensions.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int, cpuCurrentUtilization *int, cmStatus string, rescale bool) error {
	hpa.Status = extensions.HorizontalPodAutoscalerStatus{
		CurrentReplicas:                 currentReplicas,
		DesiredReplicas:                 desiredReplicas,
		CurrentCPUUtilizationPercentage: cpuCurrentUtilization,
		LastScaleTime:                   hpa.Status.LastScaleTime,
	}
	if cmStatus != "" {
		hpa.Annotations[HpaCustomMetricsStatusAnnotationName] = cmStatus
	}

	if rescale {
		now := unversioned.NewTime(time.Now())
		hpa.Status.LastScaleTime = &now
	}

	_, err := a.hpaNamespacer.HorizontalPodAutoscalers(hpa.Namespace).UpdateStatus(&hpa)
	if err != nil {
		a.eventRecorder.Event(&hpa, api.EventTypeWarning, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	return nil
}

func (a *HorizontalController) reconcileAutoscalers() error {
	ns := api.NamespaceAll
	list, err := a.hpaNamespacer.HorizontalPodAutoscalers(ns).List(api.ListOptions{})
	if err != nil {
		return fmt.Errorf("error listing nodes: %v", err)
	}
	for _, hpa := range list.Items {
		err := a.reconcileAutoscaler(hpa)
		if err != nil {
			glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
		}
	}
	return nil
}

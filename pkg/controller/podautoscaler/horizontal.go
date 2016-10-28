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
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/client/cache"
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/unversioned"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/podautoscaler/metrics"
	"k8s.io/kubernetes/pkg/runtime"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	// Usage shoud exceed the tolerance before we start downscale or upscale the pods.
	// TODO: make it a flag or HPA spec element.
	tolerance = 0.1

	defaultTargetCPUUtilizationPercentage = 80

	HpaCustomMetricsTargetAnnotationName = "alpha/target.custom-metrics.podautoscaler.kubernetes.io"
	HpaCustomMetricsStatusAnnotationName = "alpha/status.custom-metrics.podautoscaler.kubernetes.io"

	scaleUpLimitFactor  = 2
	scaleUpLimitMinimum = 4
)

func calculateScaleUpLimit(currentReplicas int32) int32 {
	return int32(math.Max(scaleUpLimitFactor*float64(currentReplicas), scaleUpLimitMinimum))
}

type HorizontalController struct {
	scaleNamespacer unversionedextensions.ScalesGetter
	hpaNamespacer   unversionedautoscaling.HorizontalPodAutoscalersGetter

	metricsClient metrics.MetricsClient
	eventRecorder record.EventRecorder

	// A store of HPA objects, populated by the controller.
	store cache.Store
	// Watches changes to all HPA objects.
	controller *framework.Controller
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func newInformer(controller *HorizontalController, resyncPeriod time.Duration) (cache.Store, *framework.Controller) {
	return framework.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return controller.hpaNamespacer.HorizontalPodAutoscalers(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return controller.hpaNamespacer.HorizontalPodAutoscalers(api.NamespaceAll).Watch(options)
			},
		},
		&autoscaling.HorizontalPodAutoscaler{},
		resyncPeriod,
		framework.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				hpa := obj.(*autoscaling.HorizontalPodAutoscaler)
				hasCPUPolicy := hpa.Spec.TargetCPUUtilizationPercentage != nil
				_, hasCustomMetricsPolicy := hpa.Annotations[HpaCustomMetricsTargetAnnotationName]
				if !hasCPUPolicy && !hasCustomMetricsPolicy {
					controller.eventRecorder.Event(hpa, api.EventTypeNormal, "DefaultPolicy", "No scaling policy specified - will use default one. See documentation for details")
				}
				err := controller.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				hpa := cur.(*autoscaling.HorizontalPodAutoscaler)
				err := controller.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			// We are not interested in deletions.
		},
	)
}

func NewHorizontalController(evtNamespacer unversionedcore.EventsGetter, scaleNamespacer unversionedextensions.ScalesGetter, hpaNamespacer unversionedautoscaling.HorizontalPodAutoscalersGetter, metricsClient metrics.MetricsClient, resyncPeriod time.Duration) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "horizontal-pod-autoscaler"})

	controller := &HorizontalController{
		metricsClient:   metricsClient,
		eventRecorder:   recorder,
		scaleNamespacer: scaleNamespacer,
		hpaNamespacer:   hpaNamespacer,
	}
	store, frameworkController := newInformer(controller, resyncPeriod)
	controller.store = store
	controller.controller = frameworkController

	return controller
}

func (a *HorizontalController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	glog.Infof("Starting HPA Controller")
	go a.controller.Run(stopCh)
	<-stopCh
	glog.Infof("Shutting down HPA Controller")
}

func (a *HorizontalController) computeReplicasForCPUUtilization(hpa *autoscaling.HorizontalPodAutoscaler, scale *extensions.Scale) (int32, *int32, time.Time, error) {
	targetUtilization := int32(defaultTargetCPUUtilizationPercentage)
	if hpa.Spec.TargetCPUUtilizationPercentage != nil {
		targetUtilization = *hpa.Spec.TargetCPUUtilizationPercentage
	}
	currentReplicas := scale.Status.Replicas

	if scale.Status.Selector == nil {
		errMsg := "selector is required"
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "SelectorRequired", errMsg)
		return 0, nil, time.Time{}, fmt.Errorf(errMsg)
	}

	selector, err := unversioned.LabelSelectorAsSelector(scale.Status.Selector)
	if err != nil {
		errMsg := fmt.Sprintf("couldn't convert selector string to a corresponding selector object: %v", err)
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "InvalidSelector", errMsg)
		return 0, nil, time.Time{}, fmt.Errorf(errMsg)
	}
	currentUtilization, numRunningPods, timestamp, err := a.metricsClient.GetCPUUtilization(hpa.Namespace, selector)

	// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
	if err != nil {
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetMetrics", err.Error())
		return 0, nil, time.Time{}, fmt.Errorf("failed to get CPU utilization: %v", err)
	}

	utilization := int32(*currentUtilization)

	usageRatio := float64(utilization) / float64(targetUtilization)
	if math.Abs(1.0-usageRatio) <= tolerance {
		return currentReplicas, &utilization, timestamp, nil
	}

	desiredReplicas := math.Ceil(usageRatio * float64(numRunningPods))

	a.eventRecorder.Eventf(hpa, api.EventTypeNormal, "DesiredReplicasComputed",
		"Computed the desired num of replicas: %d, on a base of %d report(s) (avgCPUutil: %d, current replicas: %d)",
		int32(desiredReplicas), numRunningPods, utilization, scale.Status.Replicas)

	return int32(desiredReplicas), &utilization, timestamp, nil
}

// Computes the desired number of replicas based on the CustomMetrics passed in cmAnnotation as json-serialized
// extensions.CustomMetricsTargetList.
// Returns number of replicas, metric which required highest number of replicas,
// status string (also json-serialized extensions.CustomMetricsCurrentStatusList),
// last timestamp of the metrics involved in computations or error, if occurred.
func (a *HorizontalController) computeReplicasForCustomMetrics(hpa *autoscaling.HorizontalPodAutoscaler, scale *extensions.Scale,
	cmAnnotation string) (replicas int32, metric string, status string, timestamp time.Time, err error) {

	if cmAnnotation == "" {
		return
	}

	currentReplicas := scale.Status.Replicas

	var targetList extensions.CustomMetricTargetList
	if err := json.Unmarshal([]byte(cmAnnotation), &targetList); err != nil {
		return 0, "", "", time.Time{}, fmt.Errorf("failed to parse custom metrics annotation: %v", err)
	}
	if len(targetList.Items) == 0 {
		return 0, "", "", time.Time{}, fmt.Errorf("no custom metrics in annotation")
	}

	statusList := extensions.CustomMetricCurrentStatusList{
		Items: make([]extensions.CustomMetricCurrentStatus, 0),
	}

	for _, customMetricTarget := range targetList.Items {
		if scale.Status.Selector == nil {
			errMsg := "selector is required"
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "SelectorRequired", errMsg)
			return 0, "", "", time.Time{}, fmt.Errorf("selector is required")
		}

		selector, err := unversioned.LabelSelectorAsSelector(scale.Status.Selector)
		if err != nil {
			errMsg := fmt.Sprintf("couldn't convert selector string to a corresponding selector object: %v", err)
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "InvalidSelector", errMsg)
			return 0, "", "", time.Time{}, fmt.Errorf("couldn't convert selector string to a corresponding selector object: %v", err)
		}
		value, currentTimestamp, err := a.metricsClient.GetCustomMetric(customMetricTarget.Name, hpa.Namespace, selector)
		// TODO: what to do on partial errors (like metrics obtained for 75% of pods).
		if err != nil {
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetCustomMetrics", err.Error())
			return 0, "", "", time.Time{}, fmt.Errorf("failed to get custom metric value: %v", err)
		}
		floatTarget := float64(customMetricTarget.TargetValue.MilliValue()) / 1000.0
		usageRatio := *value / floatTarget

		replicaCountProposal := int32(0)
		if math.Abs(1.0-usageRatio) > tolerance {
			replicaCountProposal = int32(math.Ceil(usageRatio * float64(currentReplicas)))
		} else {
			replicaCountProposal = currentReplicas
		}
		if replicaCountProposal > replicas {
			timestamp = currentTimestamp
			replicas = replicaCountProposal
			metric = fmt.Sprintf("Custom metric %s", customMetricTarget.Name)
		}
		quantity, err := resource.ParseQuantity(fmt.Sprintf("%.3f", *value))
		if err != nil {
			return 0, "", "", time.Time{}, fmt.Errorf("failed to set custom metric value: %v", err)
		}
		statusList.Items = append(statusList.Items, extensions.CustomMetricCurrentStatus{
			Name:         customMetricTarget.Name,
			CurrentValue: quantity,
		})
	}
	byteStatusList, err := json.Marshal(statusList)
	if err != nil {
		return 0, "", "", time.Time{}, fmt.Errorf("failed to serialize custom metric status: %v", err)
	}

	return replicas, metric, string(byteStatusList), timestamp, nil
}

func (a *HorizontalController) reconcileAutoscaler(hpa *autoscaling.HorizontalPodAutoscaler) error {
	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleTargetRef.Kind, hpa.Namespace, hpa.Spec.ScaleTargetRef.Name)

	scale, err := a.scaleNamespacer.Scales(hpa.Namespace).Get(hpa.Spec.ScaleTargetRef.Kind, hpa.Spec.ScaleTargetRef.Name)
	if err != nil {
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetScale", err.Error())
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas

	cpuDesiredReplicas := int32(0)
	cpuCurrentUtilization := new(int32)
	cpuTimestamp := time.Time{}

	cmDesiredReplicas := int32(0)
	cmMetric := ""
	cmStatus := ""
	cmTimestamp := time.Time{}

	desiredReplicas := int32(0)
	rescaleReason := ""
	timestamp := time.Now()

	if scale.Spec.Replicas == 0 {
		// Autoscaling is disabled for this resource
		desiredReplicas = 0
	} else if currentReplicas > hpa.Spec.MaxReplicas {
		rescaleReason = "Current number of replicas above Spec.MaxReplicas"
		desiredReplicas = hpa.Spec.MaxReplicas
	} else if hpa.Spec.MinReplicas != nil && currentReplicas < *hpa.Spec.MinReplicas {
		rescaleReason = "Current number of replicas below Spec.MinReplicas"
		desiredReplicas = *hpa.Spec.MinReplicas
	} else {
		// All basic scenarios covered, the state should be sane, lets use metrics.
		cmAnnotation, cmAnnotationFound := hpa.Annotations[HpaCustomMetricsTargetAnnotationName]

		if hpa.Spec.TargetCPUUtilizationPercentage != nil || !cmAnnotationFound {
			cpuDesiredReplicas, cpuCurrentUtilization, cpuTimestamp, err = a.computeReplicasForCPUUtilization(hpa, scale)
			if err != nil {
				a.updateCurrentReplicasInStatus(hpa, currentReplicas)
				a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedComputeReplicas", err.Error())
				return fmt.Errorf("failed to compute desired number of replicas based on CPU utilization for %s: %v", reference, err)
			}
		}

		if cmAnnotationFound {
			cmDesiredReplicas, cmMetric, cmStatus, cmTimestamp, err = a.computeReplicasForCustomMetrics(hpa, scale, cmAnnotation)
			if err != nil {
				a.updateCurrentReplicasInStatus(hpa, currentReplicas)
				a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedComputeCMReplicas", err.Error())
				return fmt.Errorf("failed to compute desired number of replicas based on Custom Metrics for %s: %v", reference, err)
			}
		}

		rescaleMetric := ""
		if cpuDesiredReplicas > desiredReplicas {
			desiredReplicas = cpuDesiredReplicas
			timestamp = cpuTimestamp
			rescaleMetric = "CPU utilization"
		}
		if cmDesiredReplicas > desiredReplicas {
			desiredReplicas = cmDesiredReplicas
			timestamp = cmTimestamp
			rescaleMetric = cmMetric
		}
		if desiredReplicas > currentReplicas {
			rescaleReason = fmt.Sprintf("%s above target", rescaleMetric)
		}
		if desiredReplicas < currentReplicas {
			rescaleReason = "All metrics below target"
		}

		if hpa.Spec.MinReplicas != nil && desiredReplicas < *hpa.Spec.MinReplicas {
			desiredReplicas = *hpa.Spec.MinReplicas
		}

		//  never scale down to 0, reserved for disabling autoscaling
		if desiredReplicas == 0 {
			desiredReplicas = 1
		}

		if desiredReplicas > hpa.Spec.MaxReplicas {
			desiredReplicas = hpa.Spec.MaxReplicas
		}

		// Do not upscale too much to prevent incorrect rapid increase of the number of master replicas caused by
		// bogus CPU usage report from heapster/kubelet (like in issue #32304).
		if desiredReplicas > calculateScaleUpLimit(currentReplicas) {
			desiredReplicas = calculateScaleUpLimit(currentReplicas)
		}
	}

	rescale := shouldScale(hpa, currentReplicas, desiredReplicas, timestamp)
	if rescale {
		scale.Spec.Replicas = desiredReplicas
		_, err = a.scaleNamespacer.Scales(hpa.Namespace).Update(hpa.Spec.ScaleTargetRef.Kind, scale)
		if err != nil {
			a.eventRecorder.Eventf(hpa, api.EventTypeWarning, "FailedRescale", "New size: %d; reason: %s; error: %v", desiredReplicas, rescaleReason, err.Error())
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		a.eventRecorder.Eventf(hpa, api.EventTypeNormal, "SuccessfulRescale", "New size: %d; reason: %s", desiredReplicas, rescaleReason)
		glog.Infof("Successfull rescale of %s, old size: %d, new size: %d, reason: %s",
			hpa.Name, currentReplicas, desiredReplicas, rescaleReason)
	} else {
		desiredReplicas = currentReplicas
	}

	return a.updateStatus(hpa, currentReplicas, desiredReplicas, cpuCurrentUtilization, cmStatus, rescale)
}

func shouldScale(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, timestamp time.Time) bool {
	if desiredReplicas == currentReplicas {
		return false
	}

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
	return false
}

func (a *HorizontalController) updateCurrentReplicasInStatus(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas int32) {
	err := a.updateStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentCPUUtilizationPercentage, hpa.Annotations[HpaCustomMetricsStatusAnnotationName], false)
	if err != nil {
		glog.Errorf("%v", err)
	}
}

func (a *HorizontalController) updateStatus(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, cpuCurrentUtilization *int32, cmStatus string, rescale bool) error {
	hpa.Status = autoscaling.HorizontalPodAutoscalerStatus{
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

	_, err := a.hpaNamespacer.HorizontalPodAutoscalers(hpa.Namespace).UpdateStatus(hpa)
	if err != nil {
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	glog.V(2).Infof("Successfully updated status for %s", hpa.Name)
	return nil
}

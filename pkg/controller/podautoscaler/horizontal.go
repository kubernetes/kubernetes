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
	unversionedautoscaling "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/autoscaling/internalversion"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	unversionedextensions "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion"
	"k8s.io/kubernetes/pkg/client/record"
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

	replicaCalc   *ReplicaCalculator
	eventRecorder record.EventRecorder

	// A store of HPA objects, populated by the controller.
	store cache.Store
	// Watches changes to all HPA objects.
	controller *cache.Controller
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func newInformer(controller *HorizontalController, resyncPeriod time.Duration) (cache.Store, *cache.Controller) {
	return cache.NewInformer(
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
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				hpa := obj.(*autoscaling.HorizontalPodAutoscaler)
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

func NewHorizontalController(evtNamespacer unversionedcore.EventsGetter, scaleNamespacer unversionedextensions.ScalesGetter, hpaNamespacer unversionedautoscaling.HorizontalPodAutoscalersGetter, replicaCalc *ReplicaCalculator, resyncPeriod time.Duration) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&unversionedcore.EventSinkImpl{Interface: evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(api.EventSource{Component: "horizontal-pod-autoscaler"})

	controller := &HorizontalController{
		replicaCalc:     replicaCalc,
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

// Computes the desired number of replicas for the metric specifications listed in the HPA, returning the maximum
// of the computed replica counts, a description of the associated metric, and the statuses of all metrics
// computed.
func (a *HorizontalController) computeReplicasForMetrics(hpa *autoscaling.HorizontalPodAutoscaler, scale *extensions.Scale,
	metricSpecs []autoscaling.MetricSpec) (replicas int32, metric string, statuses []autoscaling.MetricStatus, timestamp time.Time, err error) {

	currentReplicas := scale.Status.Replicas

	statuses = make([]autoscaling.MetricStatus, len(metricSpecs))

	for i, metricSpec := range metricSpecs {
		if scale.Status.Selector == nil {
			errMsg := "selector is required"
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "SelectorRequired", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		selector, err := unversioned.LabelSelectorAsSelector(scale.Status.Selector)
		if err != nil {
			errMsg := fmt.Sprintf("couldn't convert selector string to a corresponding selector object: %v", err)
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "InvalidSelector", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		var replicaCountProposal int32
		var utilizationProposal float64
		var timestampProposal time.Time
		var metricNameProposal string

		switch metricSpec.Type {
		case autoscaling.ObjectSourceType:
			floatTarget := float64(metricSpec.Object.TargetValue.MilliValue()) / 1000.0
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetObjectMetricReplicas(currentReplicas, floatTarget, metricSpec.Object.MetricName, hpa.Namespace, &metricSpec.Object.Target)
			if err != nil {
				a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetObjectMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get object metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("%s metric %s", metricSpec.Object.Target.Kind, metricSpec.Object.MetricName)
			quantity, err := resource.ParseQuantity(fmt.Sprintf("%.3f", utilizationProposal))
			if err != nil {
				return 0, "", nil, time.Time{}, fmt.Errorf("failed convert metric value to quantity: %v", err)
			}
			statuses[i] = autoscaling.MetricStatus{
				Type: autoscaling.ObjectSourceType,
				Object: &autoscaling.ObjectMetricStatus{
					Target:       metricSpec.Object.Target,
					MetricName:   metricSpec.Object.MetricName,
					CurrentValue: quantity,
				},
			}
		case autoscaling.PodsSourceType:
			floatTarget := float64(metricSpec.Pods.TargetValue.MilliValue()) / 1000.0
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetMetricReplicas(currentReplicas, floatTarget, metricSpec.Pods.MetricName, hpa.Namespace, selector)
			if err != nil {
				a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetPodsMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get pods metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("pods metric %s", metricSpec.Pods.MetricName)
			quantity, err := resource.ParseQuantity(fmt.Sprintf("%.3f", utilizationProposal))
			if err != nil {
				return 0, "", nil, time.Time{}, fmt.Errorf("failed convert metric value to quantity: %v", err)
			}
			statuses[i] = autoscaling.MetricStatus{
				Type: autoscaling.PodsSourceType,
				Pods: &autoscaling.PodsMetricStatus{
					MetricName:   metricSpec.Pods.MetricName,
					CurrentValue: quantity,
				},
			}
		case autoscaling.ResourceSourceType:
			if metricSpec.Resource.TargetRawValue != nil {
				var rawProposal int64
				replicaCountProposal, rawProposal, timestampProposal, err = a.replicaCalc.GetRawResourceReplicas(currentReplicas, metricSpec.Resource.TargetRawValue.MilliValue(), metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					return 0, "", nil, time.Time{}, fmt.Errorf("failed to get %s utilization: %v", metricSpec.Resource.Name, err)
				}
				metricNameProposal = fmt.Sprintf("%s resource", metricSpec.Resource.Name)
				quantity := resource.NewMilliQuantity(rawProposal, resource.DecimalSI)
				statuses[i] = autoscaling.MetricStatus{
					Type: autoscaling.ResourceSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name:            metricSpec.Resource.Name,
						CurrentRawValue: quantity,
					},
				}
			} else {
				// set a default utilization percentage if none is set
				targetUtilization := int32(defaultTargetCPUUtilizationPercentage)
				if metricSpec.Resource.TargetPercentageOfRequest != nil {
					targetUtilization = *metricSpec.Resource.TargetPercentageOfRequest
				}

				var percentageProposal int32
				replicaCountProposal, percentageProposal, timestampProposal, err = a.replicaCalc.GetResourceReplicas(currentReplicas, targetUtilization, metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					return 0, "", nil, time.Time{}, fmt.Errorf("failed to get %s utilization: %v", metricSpec.Resource.Name, err)
				}
				metricNameProposal = fmt.Sprintf("%s resource utilization (percentage of request)", metricSpec.Resource.Name)
				statuses[i] = autoscaling.MetricStatus{
					Type: autoscaling.ResourceSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: metricSpec.Resource.Name,
						CurrentPercentageOfRequest: &percentageProposal,
					},
				}
			}
		default:
			errMsg := fmt.Sprintf("unknown metric source type %q", string(metricSpec.Type))
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "InvalidMetricSourceType", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		if replicaCountProposal > replicas {
			timestamp = timestampProposal
			replicas = replicaCountProposal
			metric = metricNameProposal
		}
	}

	return replicas, metric, statuses, timestamp, nil
}

func (a *HorizontalController) reconcileAutoscaler(hpa *autoscaling.HorizontalPodAutoscaler) error {
	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleTargetRef.Kind, hpa.Namespace, hpa.Spec.ScaleTargetRef.Name)

	scale, err := a.scaleNamespacer.Scales(hpa.Namespace).Get(hpa.Spec.ScaleTargetRef.Kind, hpa.Spec.ScaleTargetRef.Name)
	if err != nil {
		a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedGetScale", err.Error())
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas

	var metricStatuses []autoscaling.MetricStatus
	metricDesiredReplicas := int32(0)
	metricName := ""
	metricTimestamp := time.Time{}

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
	} else if currentReplicas == 0 {
		rescaleReason = "Current number of replicas must be greater than 0"
		desiredReplicas = 1
	} else {
		metricDesiredReplicas, metricName, metricStatuses, metricTimestamp, err = a.computeReplicasForMetrics(hpa, scale, hpa.Spec.Metrics)
		if err != nil {
			a.updateCurrentReplicasInStatus(hpa, currentReplicas)
			a.eventRecorder.Event(hpa, api.EventTypeWarning, "FailedComputeMetricsReplicas", err.Error())
			return fmt.Errorf("failed to compute desired number of replicas based on listed metrics for %s: %v", reference, err)
		}

		glog.V(4).Infof("proposing %v desired replicas (based on %s from %s) for %s", metricDesiredReplicas, metricName, timestamp, reference)

		rescaleMetric := ""
		if metricDesiredReplicas > desiredReplicas {
			desiredReplicas = metricDesiredReplicas
			timestamp = metricTimestamp
			rescaleMetric = metricName
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
		glog.V(4).Infof("decided not to scale %s to %v (last scale time was %s)", reference, desiredReplicas, hpa.Status.LastScaleTime)
		desiredReplicas = currentReplicas
	}

	return a.updateStatus(hpa, currentReplicas, desiredReplicas, metricStatuses, rescale)
}

func shouldScale(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, timestamp time.Time) bool {
	if desiredReplicas == currentReplicas {
		return false
	}

	if hpa.Status.LastScaleTime == nil {
		return true
	}

	// Going down only if the usageRatio dropped significantly below the target
	// and there was no rescaling in the last downscaleForbiddenWindow.
	if desiredReplicas < currentReplicas && hpa.Status.LastScaleTime.Add(downscaleForbiddenWindow).Before(timestamp) {
		return true
	}

	// Going up only if the usage ratio increased significantly above the target
	// and there was no rescaling in the last upscaleForbiddenWindow.
	if desiredReplicas > currentReplicas && hpa.Status.LastScaleTime.Add(upscaleForbiddenWindow).Before(timestamp) {
		return true
	}

	return false
}

func (a *HorizontalController) updateCurrentReplicasInStatus(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas int32) {
	err := a.updateStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentMetrics, false)
	if err != nil {
		glog.Errorf("%v", err)
	}
}

func (a *HorizontalController) updateStatus(hpa *autoscaling.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, metricStatuses []autoscaling.MetricStatus, rescale bool) error {
	hpa.Status = autoscaling.HorizontalPodAutoscalerStatus{
		CurrentReplicas: currentReplicas,
		DesiredReplicas: desiredReplicas,
		LastScaleTime:   hpa.Status.LastScaleTime,
		CurrentMetrics:  metricStatuses,
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

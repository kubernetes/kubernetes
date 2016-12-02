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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	autoscaling "k8s.io/kubernetes/pkg/apis/autoscaling/v2alpha1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	autoscalingclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v2alpha1"
	v1coreclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/record"
)

const (
	// Usage shoud exceed the tolerance before we start downscale or upscale the pods.
	// TODO: make it a flag or HPA spec element.
	tolerance = 0.1

	scaleUpLimitFactor  = 2
	scaleUpLimitMinimum = 4
)

func calculateScaleUpLimit(currentReplicas int32) int32 {
	return int32(math.Max(scaleUpLimitFactor*float64(currentReplicas), scaleUpLimitMinimum))
}

type HorizontalController struct {
	scaleNamespacer extensionsclient.ScalesGetter
	hpaNamespacer   autoscalingclient.HorizontalPodAutoscalersGetter

	replicaCalc   *ReplicaCalculator
	eventRecorder record.EventRecorder

	// A store of HPA objects, populated by the controller.
	store cache.Store
	// Watches changes to all HPA objects.
	controller cache.Controller
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func newInformer(controller *HorizontalController, resyncPeriod time.Duration) (cache.Store, cache.Controller) {
	return cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return controller.hpaNamespacer.HorizontalPodAutoscalers(metav1.NamespaceAll).List(options)
			},
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				return controller.hpaNamespacer.HorizontalPodAutoscalers(metav1.NamespaceAll).Watch(options)
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

func NewHorizontalController(evtNamespacer v1coreclient.EventsGetter, scaleNamespacer extensionsclient.ScalesGetter, hpaNamespacer autoscalingclient.HorizontalPodAutoscalersGetter, replicaCalc *ReplicaCalculator, resyncPeriod time.Duration) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(&v1coreclient.EventSinkImpl{Interface: evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(v1.EventSource{Component: "horizontal-pod-autoscaler"})

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
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "SelectorRequired", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		var selector labels.Selector
		var err error
		if len(scale.Status.Selector) > 0 {
			selector = labels.SelectorFromSet(labels.Set(scale.Status.Selector))
			err = nil
		} else {
			selector, err = labels.Parse(scale.Status.TargetSelector)
		}
		if err != nil {
			errMsg := fmt.Sprintf("couldn't convert selector into a corresponding internal selector object: %v", err)
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "InvalidSelector", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		var replicaCountProposal int32
		var utilizationProposal int64
		var timestampProposal time.Time
		var metricNameProposal string

		switch metricSpec.Type {
		case autoscaling.ObjectSourceType:
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetObjectMetricReplicas(currentReplicas, metricSpec.Object.TargetValue.MilliValue(), metricSpec.Object.MetricName, hpa.Namespace, &metricSpec.Object.Target)
			if err != nil {
				a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetObjectMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get object metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("%s metric %s", metricSpec.Object.Target.Kind, metricSpec.Object.MetricName)
			statuses[i] = autoscaling.MetricStatus{
				Type: autoscaling.ObjectSourceType,
				Object: &autoscaling.ObjectMetricStatus{
					Target:       metricSpec.Object.Target,
					MetricName:   metricSpec.Object.MetricName,
					CurrentValue: *resource.NewMilliQuantity(utilizationProposal, resource.DecimalSI),
				},
			}
		case autoscaling.PodsSourceType:
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetMetricReplicas(currentReplicas, metricSpec.Pods.TargetAverageValue.MilliValue(), metricSpec.Pods.MetricName, hpa.Namespace, selector)
			if err != nil {
				a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetPodsMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get pods metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("pods metric %s", metricSpec.Pods.MetricName)
			statuses[i] = autoscaling.MetricStatus{
				Type: autoscaling.PodsSourceType,
				Pods: &autoscaling.PodsMetricStatus{
					MetricName:          metricSpec.Pods.MetricName,
					CurrentAverageValue: *resource.NewMilliQuantity(utilizationProposal, resource.DecimalSI),
				},
			}
		case autoscaling.ResourceSourceType:
			if metricSpec.Resource.TargetAverageValue != nil {
				var rawProposal int64
				replicaCountProposal, rawProposal, timestampProposal, err = a.replicaCalc.GetRawResourceReplicas(currentReplicas, metricSpec.Resource.TargetAverageValue.MilliValue(), metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					return 0, "", nil, time.Time{}, fmt.Errorf("failed to get %s utilization: %v", metricSpec.Resource.Name, err)
				}
				metricNameProposal = fmt.Sprintf("%s resource", metricSpec.Resource.Name)
				statuses[i] = autoscaling.MetricStatus{
					Type: autoscaling.ResourceSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name:                metricSpec.Resource.Name,
						CurrentAverageValue: *resource.NewMilliQuantity(rawProposal, resource.DecimalSI),
					},
				}
			} else {
				// set a default utilization percentage if none is set
				if metricSpec.Resource.TargetAverageUtilization == nil {
					errMsg := "invalid resource metric source: neither a utilization target nor a value target was set"
					a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetResourceMetric", errMsg)
					return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
				}

				targetUtilization := *metricSpec.Resource.TargetAverageUtilization

				var percentageProposal int32
				var rawProposal int64
				replicaCountProposal, percentageProposal, rawProposal, timestampProposal, err = a.replicaCalc.GetResourceReplicas(currentReplicas, targetUtilization, metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					return 0, "", nil, time.Time{}, fmt.Errorf("failed to get %s utilization: %v", metricSpec.Resource.Name, err)
				}
				metricNameProposal = fmt.Sprintf("%s resource utilization (percentage of request)", metricSpec.Resource.Name)
				statuses[i] = autoscaling.MetricStatus{
					Type: autoscaling.ResourceSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: metricSpec.Resource.Name,
						CurrentAverageUtilization: &percentageProposal,
						CurrentAverageValue:       *resource.NewMilliQuantity(rawProposal, resource.DecimalSI),
					},
				}
			}
		default:
			errMsg := fmt.Sprintf("unknown metric source type %q", string(metricSpec.Type))
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "InvalidMetricSourceType", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		if replicas == 0 || replicaCountProposal > replicas {
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
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetScale", err.Error())
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

	rescale := true

	if scale.Spec.Replicas == 0 {
		// Autoscaling is disabled for this resource
		desiredReplicas = 0
		rescale = false
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
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedComputeMetricsReplicas", err.Error())
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

		rescale = shouldScale(hpa, currentReplicas, desiredReplicas, timestamp)
	}

	if rescale {
		scale.Spec.Replicas = desiredReplicas
		_, err = a.scaleNamespacer.Scales(hpa.Namespace).Update(hpa.Spec.ScaleTargetRef.Kind, scale)
		if err != nil {
			a.eventRecorder.Eventf(hpa, v1.EventTypeWarning, "FailedRescale", "New size: %d; reason: %s; error: %v", desiredReplicas, rescaleReason, err.Error())
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		a.eventRecorder.Eventf(hpa, v1.EventTypeNormal, "SuccessfulRescale", "New size: %d; reason: %s", desiredReplicas, rescaleReason)
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
		now := metav1.NewTime(time.Now())
		hpa.Status.LastScaleTime = &now
	}

	_, err := a.hpaNamespacer.HorizontalPodAutoscalers(hpa.Namespace).UpdateStatus(hpa)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	glog.V(2).Infof("Successfully updated status for %s", hpa.Name)
	return nil
}

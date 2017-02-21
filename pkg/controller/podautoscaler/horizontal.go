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
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	clientv1 "k8s.io/client-go/pkg/api/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	autoscalingv2 "k8s.io/kubernetes/pkg/apis/autoscaling/v2alpha1"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	autoscalingclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/autoscaling/v1"
	extensionsclient "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/extensions/v1beta1"
	autoscalinginformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/autoscaling/v1"
	autoscalinglisters "k8s.io/kubernetes/pkg/client/listers/autoscaling/v1"
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

// UnsafeConvertToVersionVia is like api.Scheme.UnsafeConvertToVersion, but it does so via an internal version first.
// We use it since working with v2alpha1 is convenient here, but we want to use the v1 client (and
// can't just use the internal version).  Note that conversion mutates the object, so you need to deepcopy
// *before* you call this if the input object came out of a shared cache.
func UnsafeConvertToVersionVia(obj runtime.Object, externalVersion schema.GroupVersion) (runtime.Object, error) {
	objInt, err := api.Scheme.UnsafeConvertToVersion(obj, schema.GroupVersion{Group: externalVersion.Group, Version: runtime.APIVersionInternal})
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object to the internal version: %v", err)
	}

	objExt, err := api.Scheme.UnsafeConvertToVersion(objInt, externalVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object back to the external version: %v", err)
	}

	return objExt, err
}

type HorizontalController struct {
	scaleNamespacer extensionsclient.ScalesGetter
	hpaNamespacer   autoscalingclient.HorizontalPodAutoscalersGetter

	replicaCalc   *ReplicaCalculator
	eventRecorder record.EventRecorder

	// hpaLister is able to list/get HPAs from the shared cache from the informer passed in to
	// NewHorizontalController.
	hpaLister       autoscalinglisters.HorizontalPodAutoscalerLister
	hpaListerSynced cache.InformerSynced
}

var downscaleForbiddenWindow = 5 * time.Minute
var upscaleForbiddenWindow = 3 * time.Minute

func NewHorizontalController(
	evtNamespacer v1core.EventsGetter,
	scaleNamespacer extensionsclient.ScalesGetter,
	hpaNamespacer autoscalingclient.HorizontalPodAutoscalersGetter,
	replicaCalc *ReplicaCalculator,
	hpaInformer autoscalinginformers.HorizontalPodAutoscalerInformer,
	resyncPeriod time.Duration,
) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(api.Scheme, clientv1.EventSource{Component: "horizontal-pod-autoscaler"})

	controller := &HorizontalController{
		replicaCalc:     replicaCalc,
		eventRecorder:   recorder,
		scaleNamespacer: scaleNamespacer,
		hpaNamespacer:   hpaNamespacer,
	}

	hpaInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
				err := controller.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				hpa := cur.(*autoscalingv1.HorizontalPodAutoscaler)
				err := controller.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			// We are not interested in deletions.
		},
		resyncPeriod,
	)
	controller.hpaLister = hpaInformer.Lister()
	controller.hpaListerSynced = hpaInformer.Informer().HasSynced

	return controller
}

func (a *HorizontalController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	glog.Infof("Starting HPA Controller")

	if !cache.WaitForCacheSync(stopCh, a.hpaListerSynced) {
		utilruntime.HandleError(fmt.Errorf("timed out waiting for caches to sync"))
		return
	}

	<-stopCh
	glog.Infof("Shutting down HPA Controller")
}

// Computes the desired number of replicas for the metric specifications listed in the HPA, returning the maximum
// of the computed replica counts, a description of the associated metric, and the statuses of all metrics
// computed.
func (a *HorizontalController) computeReplicasForMetrics(hpa *autoscalingv2.HorizontalPodAutoscaler, scale *extensions.Scale,
	metricSpecs []autoscalingv2.MetricSpec) (replicas int32, metric string, statuses []autoscalingv2.MetricStatus, timestamp time.Time, err error) {

	currentReplicas := scale.Status.Replicas

	statuses = make([]autoscalingv2.MetricStatus, len(metricSpecs))

	for i, metricSpec := range metricSpecs {
		if len(scale.Status.Selector) == 0 && len(scale.Status.TargetSelector) == 0 {
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
		case autoscalingv2.ObjectMetricSourceType:
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetObjectMetricReplicas(currentReplicas, metricSpec.Object.TargetValue.MilliValue(), metricSpec.Object.MetricName, hpa.Namespace, &metricSpec.Object.Target)
			if err != nil {
				a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetObjectMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get object metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("%s metric %s", metricSpec.Object.Target.Kind, metricSpec.Object.MetricName)
			statuses[i] = autoscalingv2.MetricStatus{
				Type: autoscalingv2.ObjectMetricSourceType,
				Object: &autoscalingv2.ObjectMetricStatus{
					Target:       metricSpec.Object.Target,
					MetricName:   metricSpec.Object.MetricName,
					CurrentValue: *resource.NewMilliQuantity(utilizationProposal, resource.DecimalSI),
				},
			}
		case autoscalingv2.PodsMetricSourceType:
			replicaCountProposal, utilizationProposal, timestampProposal, err = a.replicaCalc.GetMetricReplicas(currentReplicas, metricSpec.Pods.TargetAverageValue.MilliValue(), metricSpec.Pods.MetricName, hpa.Namespace, selector)
			if err != nil {
				a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetPodsMetric", err.Error())
				return 0, "", nil, time.Time{}, fmt.Errorf("failed to get pods metric value: %v", err)
			}
			metricNameProposal = fmt.Sprintf("pods metric %s", metricSpec.Pods.MetricName)
			statuses[i] = autoscalingv2.MetricStatus{
				Type: autoscalingv2.PodsMetricSourceType,
				Pods: &autoscalingv2.PodsMetricStatus{
					MetricName:          metricSpec.Pods.MetricName,
					CurrentAverageValue: *resource.NewMilliQuantity(utilizationProposal, resource.DecimalSI),
				},
			}
		case autoscalingv2.ResourceMetricSourceType:
			if metricSpec.Resource.TargetAverageValue != nil {
				var rawProposal int64
				replicaCountProposal, rawProposal, timestampProposal, err = a.replicaCalc.GetRawResourceReplicas(currentReplicas, metricSpec.Resource.TargetAverageValue.MilliValue(), metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					return 0, "", nil, time.Time{}, fmt.Errorf("failed to get %s utilization: %v", metricSpec.Resource.Name, err)
				}
				metricNameProposal = fmt.Sprintf("%s resource", metricSpec.Resource.Name)
				statuses[i] = autoscalingv2.MetricStatus{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricStatus{
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
				statuses[i] = autoscalingv2.MetricStatus{
					Type: autoscalingv2.ResourceMetricSourceType,
					Resource: &autoscalingv2.ResourceMetricStatus{
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

func (a *HorizontalController) reconcileAutoscaler(hpav1Shared *autoscalingv1.HorizontalPodAutoscaler) error {
	// make a copy so that we never mutate the shared informer cache (conversion can mutate the object)
	hpav1Raw, err := api.Scheme.DeepCopy(hpav1Shared)
	if err != nil {
		a.eventRecorder.Event(hpav1Shared, v1.EventTypeWarning, "FailedConvertHPA", err.Error())
		return fmt.Errorf("failed to deep-copy the HPA: %v", err)
	}

	// then, convert to autoscaling/v2, which makes our lives easier when calculating metrics
	hpav1 := hpav1Raw.(*autoscalingv1.HorizontalPodAutoscaler)
	hpaRaw, err := UnsafeConvertToVersionVia(hpav1, autoscalingv2.SchemeGroupVersion)
	if err != nil {
		a.eventRecorder.Event(hpav1, v1.EventTypeWarning, "FailedConvertHPA", err.Error())
		return fmt.Errorf("failed to convert the given HPA to %s: %v", autoscalingv2.SchemeGroupVersion.String(), err)
	}
	hpa := hpaRaw.(*autoscalingv2.HorizontalPodAutoscaler)

	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleTargetRef.Kind, hpa.Namespace, hpa.Spec.ScaleTargetRef.Name)

	scale, err := a.scaleNamespacer.Scales(hpa.Namespace).Get(hpa.Spec.ScaleTargetRef.Kind, hpa.Spec.ScaleTargetRef.Name)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetScale", err.Error())
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas

	var metricStatuses []autoscalingv2.MetricStatus
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

func shouldScale(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, timestamp time.Time) bool {
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

func (a *HorizontalController) updateCurrentReplicasInStatus(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas int32) {
	err := a.updateStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentMetrics, false)
	if err != nil {
		utilruntime.HandleError(err)
	}
}

func (a *HorizontalController) updateStatus(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, metricStatuses []autoscalingv2.MetricStatus, rescale bool) error {
	hpa.Status = autoscalingv2.HorizontalPodAutoscalerStatus{
		CurrentReplicas: currentReplicas,
		DesiredReplicas: desiredReplicas,
		LastScaleTime:   hpa.Status.LastScaleTime,
		CurrentMetrics:  metricStatuses,
	}

	if rescale {
		now := metav1.NewTime(time.Now())
		hpa.Status.LastScaleTime = &now
	}

	// convert back to autoscalingv1
	hpaRaw, err := UnsafeConvertToVersionVia(hpa, autoscalingv1.SchemeGroupVersion)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedConvertHPA", err.Error())
		return fmt.Errorf("failed to convert the given HPA to %s: %v", autoscalingv2.SchemeGroupVersion.String(), err)
	}
	hpav1 := hpaRaw.(*autoscalingv1.HorizontalPodAutoscaler)

	_, err = a.hpaNamespacer.HorizontalPodAutoscalers(hpav1.Namespace).UpdateStatus(hpav1)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	glog.V(2).Infof("Successfully updated status for %s", hpa.Name)
	return nil
}

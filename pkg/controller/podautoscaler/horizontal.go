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
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2beta1"
	"k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	autoscalinginformers "k8s.io/client-go/informers/autoscaling/v1"
	"k8s.io/client-go/kubernetes/scheme"
	autoscalingclient "k8s.io/client-go/kubernetes/typed/autoscaling/v1"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	autoscalinglisters "k8s.io/client-go/listers/autoscaling/v1"
	scaleclient "k8s.io/client-go/scale"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	"k8s.io/kubernetes/pkg/controller"
)

var (
	scaleUpLimitFactor  = 2.0
	scaleUpLimitMinimum = 4.0
)

// HorizontalController is responsible for the synchronizing HPA objects stored
// in the system with the actual deployments/replication controllers they
// control.
type HorizontalController struct {
	scaleNamespacer scaleclient.ScalesGetter
	hpaNamespacer   autoscalingclient.HorizontalPodAutoscalersGetter
	mapper          apimeta.RESTMapper

	replicaCalc   *ReplicaCalculator
	eventRecorder record.EventRecorder

	upscaleForbiddenWindow   time.Duration
	downscaleForbiddenWindow time.Duration

	// hpaLister is able to list/get HPAs from the shared cache from the informer passed in to
	// NewHorizontalController.
	hpaLister       autoscalinglisters.HorizontalPodAutoscalerLister
	hpaListerSynced cache.InformerSynced

	// Controllers that need to be synced
	queue workqueue.RateLimitingInterface
}

// NewHorizontalController creates a new HorizontalController.
func NewHorizontalController(
	evtNamespacer v1core.EventsGetter,
	scaleNamespacer scaleclient.ScalesGetter,
	hpaNamespacer autoscalingclient.HorizontalPodAutoscalersGetter,
	mapper apimeta.RESTMapper,
	replicaCalc *ReplicaCalculator,
	hpaInformer autoscalinginformers.HorizontalPodAutoscalerInformer,
	resyncPeriod time.Duration,
	upscaleForbiddenWindow time.Duration,
	downscaleForbiddenWindow time.Duration,

) *HorizontalController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartLogging(glog.Infof)
	// TODO: remove the wrapper when every clients have moved to use the clientset.
	broadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: evtNamespacer.Events("")})
	recorder := broadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "horizontal-pod-autoscaler"})

	hpaController := &HorizontalController{
		replicaCalc:              replicaCalc,
		eventRecorder:            recorder,
		scaleNamespacer:          scaleNamespacer,
		hpaNamespacer:            hpaNamespacer,
		upscaleForbiddenWindow:   upscaleForbiddenWindow,
		downscaleForbiddenWindow: downscaleForbiddenWindow,
		queue:  workqueue.NewNamedRateLimitingQueue(NewDefaultHPARateLimiter(resyncPeriod), "horizontalpodautoscaler"),
		mapper: mapper,
	}

	hpaInformer.Informer().AddEventHandlerWithResyncPeriod(
		cache.ResourceEventHandlerFuncs{
			AddFunc:    hpaController.enqueueHPA,
			UpdateFunc: hpaController.updateHPA,
			DeleteFunc: hpaController.deleteHPA,
		},
		resyncPeriod,
	)
	hpaController.hpaLister = hpaInformer.Lister()
	hpaController.hpaListerSynced = hpaInformer.Informer().HasSynced

	return hpaController
}

// Run begins watching and syncing.
func (a *HorizontalController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer a.queue.ShutDown()

	glog.Infof("Starting HPA controller")
	defer glog.Infof("Shutting down HPA controller")

	if !controller.WaitForCacheSync("HPA", stopCh, a.hpaListerSynced) {
		return
	}

	// start a single worker (we may wish to start more in the future)
	go wait.Until(a.worker, time.Second, stopCh)

	<-stopCh
}

// obj could be an *v1.HorizontalPodAutoscaler, or a DeletionFinalStateUnknown marker item.
func (a *HorizontalController) updateHPA(old, cur interface{}) {
	a.enqueueHPA(cur)
}

// obj could be an *v1.HorizontalPodAutoscaler, or a DeletionFinalStateUnknown marker item.
func (a *HorizontalController) enqueueHPA(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}

	// always add rate-limitted so we don't fetch metrics more that once per resync interval
	a.queue.AddRateLimited(key)
}

func (a *HorizontalController) deleteHPA(obj interface{}) {
	key, err := controller.KeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("couldn't get key for object %+v: %v", obj, err))
		return
	}

	// TODO: could we leak if we fail to get the key?
	a.queue.Forget(key)
}

func (a *HorizontalController) worker() {
	for a.processNextWorkItem() {
	}
	glog.Infof("horizontal pod autoscaler controller worker shutting down")
}

func (a *HorizontalController) processNextWorkItem() bool {
	key, quit := a.queue.Get()
	if quit {
		return false
	}
	defer a.queue.Done(key)

	err := a.reconcileKey(key.(string))
	if err == nil {
		// don't "forget" here because we want to only process a given HPA once per resync interval
		return true
	}

	a.queue.AddRateLimited(key)
	utilruntime.HandleError(err)
	return true
}

// Computes the desired number of replicas for the metric specifications listed in the HPA, returning the maximum
// of the computed replica counts, a description of the associated metric, and the statuses of all metrics
// computed.
func (a *HorizontalController) computeReplicasForMetrics(hpa *autoscalingv2.HorizontalPodAutoscaler, scale *autoscalingv1.Scale,
	metricSpecs []autoscalingv2.MetricSpec) (replicas int32, metric string, statuses []autoscalingv2.MetricStatus, timestamp time.Time, err error) {

	currentReplicas := scale.Status.Replicas

	statuses = make([]autoscalingv2.MetricStatus, len(metricSpecs))

	for i, metricSpec := range metricSpecs {
		if scale.Status.Selector == "" {
			errMsg := "selector is required"
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "SelectorRequired", errMsg)
			setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "InvalidSelector", "the HPA target's scale is missing a selector")
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		selector, err := labels.Parse(scale.Status.Selector)
		if err != nil {
			errMsg := fmt.Sprintf("couldn't convert selector into a corresponding internal selector object: %v", err)
			a.eventRecorder.Event(hpa, v1.EventTypeWarning, "InvalidSelector", errMsg)
			setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "InvalidSelector", errMsg)
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
				setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "FailedGetObjectMetric", "the HPA was unable to compute the replica count: %v", err)
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
				setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "FailedGetPodsMetric", "the HPA was unable to compute the replica count: %v", err)
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
					setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "FailedGetResourceMetric", "the HPA was unable to compute the replica count: %v", err)
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
					setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "FailedGetResourceMetric", "the HPA was unable to compute the replica count: %s", errMsg)
					return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
				}

				targetUtilization := *metricSpec.Resource.TargetAverageUtilization

				var percentageProposal int32
				var rawProposal int64
				replicaCountProposal, percentageProposal, rawProposal, timestampProposal, err = a.replicaCalc.GetResourceReplicas(currentReplicas, targetUtilization, metricSpec.Resource.Name, hpa.Namespace, selector)
				if err != nil {
					a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetResourceMetric", err.Error())
					setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "FailedGetResourceMetric", "the HPA was unable to compute the replica count: %v", err)
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
			setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "InvalidMetricSourceType", "the HPA was unable to compute the replica count: %s", errMsg)
			return 0, "", nil, time.Time{}, fmt.Errorf(errMsg)
		}

		if replicas == 0 || replicaCountProposal > replicas {
			timestamp = timestampProposal
			replicas = replicaCountProposal
			metric = metricNameProposal
		}
	}

	setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionTrue, "ValidMetricFound", "the HPA was able to succesfully calculate a replica count from %s", metric)
	return replicas, metric, statuses, timestamp, nil
}

func (a *HorizontalController) reconcileKey(key string) error {
	namespace, name, err := cache.SplitMetaNamespaceKey(key)
	if err != nil {
		return err
	}

	hpa, err := a.hpaLister.HorizontalPodAutoscalers(namespace).Get(name)
	if errors.IsNotFound(err) {
		glog.Infof("Horizontal Pod Autoscaler has been deleted %v", key)
		return nil
	}

	return a.reconcileAutoscaler(hpa)
}

func (a *HorizontalController) reconcileAutoscaler(hpav1Shared *autoscalingv1.HorizontalPodAutoscaler) error {
	// make a copy so that we never mutate the shared informer cache (conversion can mutate the object)
	hpav1 := hpav1Shared.DeepCopy()
	// then, convert to autoscaling/v2, which makes our lives easier when calculating metrics
	hpaRaw, err := unsafeConvertToVersionVia(hpav1, autoscalingv2.SchemeGroupVersion)
	if err != nil {
		a.eventRecorder.Event(hpav1, v1.EventTypeWarning, "FailedConvertHPA", err.Error())
		return fmt.Errorf("failed to convert the given HPA to %s: %v", autoscalingv2.SchemeGroupVersion.String(), err)
	}
	hpa := hpaRaw.(*autoscalingv2.HorizontalPodAutoscaler)
	hpaStatusOriginal := hpa.Status.DeepCopy()

	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleTargetRef.Kind, hpa.Namespace, hpa.Spec.ScaleTargetRef.Name)

	targetGV, err := schema.ParseGroupVersion(hpa.Spec.ScaleTargetRef.APIVersion)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetScale", err.Error())
		setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "FailedGetScale", "the HPA controller was unable to get the target's current scale: %v", err)
		a.updateStatusIfNeeded(hpaStatusOriginal, hpa)
		return fmt.Errorf("invalid API version in scale target reference: %v", err)
	}

	targetGK := schema.GroupKind{
		Group: targetGV.Group,
		Kind:  hpa.Spec.ScaleTargetRef.Kind,
	}

	mappings, err := a.mapper.RESTMappings(targetGK)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetScale", err.Error())
		setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "FailedGetScale", "the HPA controller was unable to get the target's current scale: %v", err)
		a.updateStatusIfNeeded(hpaStatusOriginal, hpa)
		return fmt.Errorf("unable to determine resource for scale target reference: %v", err)
	}

	scale, targetGR, err := a.scaleForResourceMappings(hpa.Namespace, hpa.Spec.ScaleTargetRef.Name, mappings)
	if err != nil {
		a.eventRecorder.Event(hpa, v1.EventTypeWarning, "FailedGetScale", err.Error())
		setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "FailedGetScale", "the HPA controller was unable to get the target's current scale: %v", err)
		a.updateStatusIfNeeded(hpaStatusOriginal, hpa)
		return fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionTrue, "SucceededGetScale", "the HPA controller was able to get the target's current scale")
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
		setCondition(hpa, autoscalingv2.ScalingActive, v1.ConditionFalse, "ScalingDisabled", "scaling is disabled since the replica count of the target is zero")
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
			a.setCurrentReplicasInStatus(hpa, currentReplicas)
			if err := a.updateStatusIfNeeded(hpaStatusOriginal, hpa); err != nil {
				utilruntime.HandleError(err)
			}
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

		// Do not upscale too much to prevent incorrect rapid increase of the number of master replicas caused by
		// bogus CPU usage report from heapster/kubelet (like in issue #32304).
		scaleUpLimit := calculateScaleUpLimit(currentReplicas)

		switch {
		case desiredReplicas > scaleUpLimit:
			setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionTrue, "ScaleUpLimit", "the desired replica count is increasing faster than the maximum scale rate")
			desiredReplicas = scaleUpLimit

			// Ensure that even if the scaleUpLimit is greater
			// than the maximum number of replicas, we only
			// set the max number of replicas as desired.
			if scaleUpLimit > hpa.Spec.MaxReplicas {
				setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionTrue, "TooManyReplicas", "the desired replica count was more than the maximum replica count")
				desiredReplicas = hpa.Spec.MaxReplicas
			}

		case hpa.Spec.MinReplicas != nil && desiredReplicas < *hpa.Spec.MinReplicas:
			// make sure we aren't below our minimum
			var statusMsg string
			if desiredReplicas == 0 {
				statusMsg = "the desired replica count was zero"
			} else {
				statusMsg = "the desired replica count was less than the minimum replica count"
			}

			setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionTrue, "TooFewReplicas", statusMsg)
			desiredReplicas = *hpa.Spec.MinReplicas
		case desiredReplicas == 0:
			//  never scale down to 0, reserved for disabling autoscaling
			setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionTrue, "TooFewReplicas", "the desired replica count was zero")
			desiredReplicas = 1
		case desiredReplicas > hpa.Spec.MaxReplicas:
			// make sure we aren't above our maximum
			setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionTrue, "TooManyReplicas", "the desired replica count was more than the maximum replica count")
			desiredReplicas = hpa.Spec.MaxReplicas
		default:
			// mark that we're within acceptible limits
			setCondition(hpa, autoscalingv2.ScalingLimited, v1.ConditionFalse, "DesiredWithinRange", "the desired replica count is within the acceptible range")
		}

		rescale = a.shouldScale(hpa, currentReplicas, desiredReplicas, timestamp)
		backoffDown := false
		backoffUp := false
		if hpa.Status.LastScaleTime != nil {
			if !hpa.Status.LastScaleTime.Add(a.downscaleForbiddenWindow).Before(timestamp) {
				setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "BackoffDownscale", "the time since the previous scale is still within the downscale forbidden window")
				backoffDown = true
			}

			if !hpa.Status.LastScaleTime.Add(a.upscaleForbiddenWindow).Before(timestamp) {
				backoffUp = true
				if backoffDown {
					setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "BackoffBoth", "the time since the previous scale is still within both the downscale and upscale forbidden windows")
				} else {
					setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "BackoffUpscale", "the time since the previous scale is still within the upscale forbidden window")
				}
			}
		}

		if !backoffDown && !backoffUp {
			// mark that we're not backing off
			setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionTrue, "ReadyForNewScale", "the last scale time was sufficiently old as to warrant a new scale")
		}

	}

	if rescale {
		scale.Spec.Replicas = desiredReplicas
		_, err = a.scaleNamespacer.Scales(hpa.Namespace).Update(targetGR, scale)
		if err != nil {
			a.eventRecorder.Eventf(hpa, v1.EventTypeWarning, "FailedRescale", "New size: %d; reason: %s; error: %v", desiredReplicas, rescaleReason, err.Error())
			setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionFalse, "FailedUpdateScale", "the HPA controller was unable to update the target scale: %v", err)
			a.setCurrentReplicasInStatus(hpa, currentReplicas)
			if err := a.updateStatusIfNeeded(hpaStatusOriginal, hpa); err != nil {
				utilruntime.HandleError(err)
			}
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		setCondition(hpa, autoscalingv2.AbleToScale, v1.ConditionTrue, "SucceededRescale", "the HPA controller was able to update the target scale to %d", desiredReplicas)
		a.eventRecorder.Eventf(hpa, v1.EventTypeNormal, "SuccessfulRescale", "New size: %d; reason: %s", desiredReplicas, rescaleReason)
		glog.Infof("Successful rescale of %s, old size: %d, new size: %d, reason: %s",
			hpa.Name, currentReplicas, desiredReplicas, rescaleReason)
	} else {
		glog.V(4).Infof("decided not to scale %s to %v (last scale time was %s)", reference, desiredReplicas, hpa.Status.LastScaleTime)
		desiredReplicas = currentReplicas
	}

	a.setStatus(hpa, currentReplicas, desiredReplicas, metricStatuses, rescale)
	return a.updateStatusIfNeeded(hpaStatusOriginal, hpa)
}

func (a *HorizontalController) shouldScale(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, timestamp time.Time) bool {
	if desiredReplicas == currentReplicas {
		return false
	}

	if hpa.Status.LastScaleTime == nil {
		return true
	}

	// Going down only if the usageRatio dropped significantly below the target
	// and there was no rescaling in the last downscaleForbiddenWindow.
	if desiredReplicas < currentReplicas && hpa.Status.LastScaleTime.Add(a.downscaleForbiddenWindow).Before(timestamp) {
		return true
	}

	// Going up only if the usage ratio increased significantly above the target
	// and there was no rescaling in the last upscaleForbiddenWindow.
	if desiredReplicas > currentReplicas && hpa.Status.LastScaleTime.Add(a.upscaleForbiddenWindow).Before(timestamp) {
		return true
	}

	return false
}

// scaleForResourceMappings attempts to fetch the scale for the
// resource with the given name and namespace, trying each RESTMapping
// in turn until a working one is found.  If none work, the first error
// is returned.  It returns both the scale, as well as the group-resource from
// the working mapping.
func (a *HorizontalController) scaleForResourceMappings(namespace, name string, mappings []*apimeta.RESTMapping) (*autoscalingv1.Scale, schema.GroupResource, error) {
	var firstErr error
	for i, mapping := range mappings {
		targetGR := mapping.GroupVersionKind.GroupVersion().WithResource(mapping.Resource).GroupResource()
		scale, err := a.scaleNamespacer.Scales(namespace).Get(targetGR, name)
		if err == nil {
			return scale, targetGR, nil
		}

		// if this is the first error, remember it,
		// then go on and try other mappings until we find a good one
		if i == 0 {
			firstErr = err
		}
	}

	// make sure we handle an empty set of mappings
	if firstErr == nil {
		firstErr = fmt.Errorf("unrecognized resource")
	}

	return nil, schema.GroupResource{}, firstErr
}

// setCurrentReplicasInStatus sets the current replica count in the status of the HPA.
func (a *HorizontalController) setCurrentReplicasInStatus(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas int32) {
	a.setStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentMetrics, false)
}

// setStatus recreates the status of the given HPA, updating the current and
// desired replicas, as well as the metric statuses
func (a *HorizontalController) setStatus(hpa *autoscalingv2.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, metricStatuses []autoscalingv2.MetricStatus, rescale bool) {
	hpa.Status = autoscalingv2.HorizontalPodAutoscalerStatus{
		CurrentReplicas: currentReplicas,
		DesiredReplicas: desiredReplicas,
		LastScaleTime:   hpa.Status.LastScaleTime,
		CurrentMetrics:  metricStatuses,
		Conditions:      hpa.Status.Conditions,
	}

	if rescale {
		now := metav1.NewTime(time.Now())
		hpa.Status.LastScaleTime = &now
	}
}

// updateStatusIfNeeded calls updateStatus only if the status of the new HPA is not the same as the old status
func (a *HorizontalController) updateStatusIfNeeded(oldStatus *autoscalingv2.HorizontalPodAutoscalerStatus, newHPA *autoscalingv2.HorizontalPodAutoscaler) error {
	// skip a write if we wouldn't need to update
	if apiequality.Semantic.DeepEqual(oldStatus, &newHPA.Status) {
		return nil
	}
	return a.updateStatus(newHPA)
}

// updateStatus actually does the update request for the status of the given HPA
func (a *HorizontalController) updateStatus(hpa *autoscalingv2.HorizontalPodAutoscaler) error {
	// convert back to autoscalingv1
	hpaRaw, err := unsafeConvertToVersionVia(hpa, autoscalingv1.SchemeGroupVersion)
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

// unsafeConvertToVersionVia is like api.Scheme.UnsafeConvertToVersion, but it does so via an internal version first.
// We use it since working with v2alpha1 is convenient here, but we want to use the v1 client (and
// can't just use the internal version).  Note that conversion mutates the object, so you need to deepcopy
// *before* you call this if the input object came out of a shared cache.
func unsafeConvertToVersionVia(obj runtime.Object, externalVersion schema.GroupVersion) (runtime.Object, error) {
	objInt, err := legacyscheme.Scheme.UnsafeConvertToVersion(obj, schema.GroupVersion{Group: externalVersion.Group, Version: runtime.APIVersionInternal})
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object to the internal version: %v", err)
	}

	objExt, err := legacyscheme.Scheme.UnsafeConvertToVersion(objInt, externalVersion)
	if err != nil {
		return nil, fmt.Errorf("failed to convert the given object back to the external version: %v", err)
	}

	return objExt, err
}

// setCondition sets the specific condition type on the given HPA to the specified value with the given reason
// and message.  The message and args are treated like a format string.  The condition will be added if it is
// not present.
func setCondition(hpa *autoscalingv2.HorizontalPodAutoscaler, conditionType autoscalingv2.HorizontalPodAutoscalerConditionType, status v1.ConditionStatus, reason, message string, args ...interface{}) {
	hpa.Status.Conditions = setConditionInList(hpa.Status.Conditions, conditionType, status, reason, message, args...)
}

// setConditionInList sets the specific condition type on the given HPA to the specified value with the given
// reason and message.  The message and args are treated like a format string.  The condition will be added if
// it is not present.  The new list will be returned.
func setConditionInList(inputList []autoscalingv2.HorizontalPodAutoscalerCondition, conditionType autoscalingv2.HorizontalPodAutoscalerConditionType, status v1.ConditionStatus, reason, message string, args ...interface{}) []autoscalingv2.HorizontalPodAutoscalerCondition {
	resList := inputList
	var existingCond *autoscalingv2.HorizontalPodAutoscalerCondition
	for i, condition := range resList {
		if condition.Type == conditionType {
			// can't take a pointer to an iteration variable
			existingCond = &resList[i]
			break
		}
	}

	if existingCond == nil {
		resList = append(resList, autoscalingv2.HorizontalPodAutoscalerCondition{
			Type: conditionType,
		})
		existingCond = &resList[len(resList)-1]
	}

	if existingCond.Status != status {
		existingCond.LastTransitionTime = metav1.Now()
	}

	existingCond.Status = status
	existingCond.Reason = reason
	existingCond.Message = fmt.Sprintf(message, args...)

	return resList
}

func calculateScaleUpLimit(currentReplicas int32) int32 {
	return int32(math.Max(scaleUpLimitFactor*float64(currentReplicas), scaleUpLimitMinimum))
}

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

package autoscaler

import (
	"fmt"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedmetricsclient "k8s.io/kubernetes/federation/pkg/federation-controller/autoscaler/metrics"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/deletionhelper"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util/eventsink"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"math"
)

const (
	allClustersKey                        = "THE_ALL_CLUSTER_KEY"
	defaultTargetCPUUtilizationPercentage = 80

	HpaCustomMetricsTargetAnnotationName = "alpha/target.custom-metrics.podautoscaler.kubernetes.io"
	HpaCustomMetricsStatusAnnotationName = "alpha/status.custom-metrics.podautoscaler.kubernetes.io"

	scaleUpLimitFactor  = 2
	scaleUpLimitMinimum = 4
)

var (
	autoscalerReviewDelay    = 10 * time.Second
	clusterAvailableDelay    = 20 * time.Second
	clusterUnavailableDelay  = 60 * time.Second
	updateTimeout            = 30 * time.Second
	hpaResyncDelay           = 5 * time.Second
	backoffInitial           = 5 * time.Second
	backoffMax               = 1 * time.Minute
	tolerance                = 0.1
	downscaleForbiddenWindow = 5 * time.Minute
	upscaleForbiddenWindow   = 3 * time.Minute
)

func calculateScaleUpLimit(currentReplicas int32) int32 {
	return int32(math.Max(scaleUpLimitFactor*float64(currentReplicas), scaleUpLimitMinimum))
}

type AutoscalerController struct {
	//client to federation api server
	fedClient fedclientset.Interface
	//Informer on pods present in members of federation.
	autoscalerPodInformer fedutil.FederatedInformer
	//clusterDeliverer is needed only to get information to
	//pod informer of federated clusters
	//basically maintain an updated list of federated clusters
	clusterDeliverer *fedutil.DelayingDeliverer
	//Informer controller for the hpa in federation
	//IRF: TODO why name this informer controller?
	autoscalerInformerController cache.Controller
	//stores details of hpas that are created in federation
	autoscalerInformerStore cache.Store
	//For triggering hpa reconciliations
	//autoscalerDeliverer *fedutil.DelayingDeliverer
	//Backoff manager for hpas
	//autoscalerBackoff *flowcontrol.Backoff
	// For events
	eventRecorder record.EventRecorder

	metricsClient fedmetricsclient.MetricsClient

	//IRF TODO: need to check if this is needed
	deletionHelper *deletionhelper.DeletionHelper
}

// NewclusterController returns a new cluster controller
func NewAutoscalerController(fedClient fedclientset.Interface) *AutoscalerController {
	broadcaster := record.NewBroadcaster()
	broadcaster.StartRecordingToSink(eventsink.NewFederatedEventSink(fedClient))
	recorder := broadcaster.NewRecorder(apiv1.EventSource{Component: "federated-autoscaler-controller"})

	fac := &AutoscalerController{
		fedClient:     fedClient,
		eventRecorder: recorder,
	}

	fac.autoscalerInformerStore, fac.autoscalerInformerController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
				return fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).List(options)
			},
			WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
				return fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(apiv1.NamespaceAll).Watch(options)
			},
		},
		&autoscalingv1.HorizontalPodAutoscaler{},
		//IRF: TODO should we get this from some other configurable entity
		hpaResyncDelay, //we want the reconcile to be called periodically
		cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
				err := fac.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			UpdateFunc: func(old, cur interface{}) {
				hpa := cur.(*autoscalingv1.HorizontalPodAutoscaler)
				err := fac.reconcileAutoscaler(hpa)
				if err != nil {
					glog.Warningf("Failed to reconcile %s: %v", hpa.Name, err)
				}
			},
			//no need to care about deletions here
		})

	podFedInformerFactory := func(cluster *fedv1.Cluster, clientset kubeclientset.Interface) (cache.Store, cache.Controller) {
		return cache.NewInformer(
			&cache.ListWatch{
				ListFunc: func(options apiv1.ListOptions) (runtime.Object, error) {
					return clientset.Core().Pods(apiv1.NamespaceAll).List(options)
				},
				WatchFunc: func(options apiv1.ListOptions) (watch.Interface, error) {
					return clientset.Core().Pods(apiv1.NamespaceAll).Watch(options)
				},
			},
			&apiv1.Pod{},
			controller.NoResyncPeriodFunc(),
			fedutil.NewTriggerOnAllChanges(
				func(obj runtime.Object) {
					//TODO: check if this noop can be handled in a better way
					return
				},
			),
		)
	}
	fac.autoscalerPodInformer = fedutil.NewFederatedInformer(fedClient, podFedInformerFactory, &fedutil.ClusterLifecycleHandlerFuncs{})

	fac.metricsClient = fedmetricsclient.NewHeapsterMetricsClient(fedmetricsclient.DefaultHeapsterNamespace,
		fedmetricsclient.DefaultHeapsterScheme,
		fedmetricsclient.DefaultHeapsterService,
		fedmetricsclient.DefaultHeapsterPort)

	return fac
}

func (fac *AutoscalerController) Run(stopChan <-chan struct{}) {
	go fac.autoscalerInformerController.Run(stopChan)
	fac.autoscalerPodInformer.Start()
	go func() {
		<-stopChan
		glog.Infof("Shutting down autoscaler controller")
		fac.autoscalerPodInformer.Stop()
	}()
}

func (fac *AutoscalerController) isSynced() bool {
	if !fac.autoscalerPodInformer.ClustersSynced() {
		glog.V(2).Infof("Cluster list not synced")
		return false
	}
	clusters, err := fac.autoscalerPodInformer.GetReadyClusters()
	if err != nil {
		glog.Errorf("Failed to get ready clusters: %v", err)
		return false
	}
	if len(clusters) <= 0 {
		//no point in calling reconcile now as ther are no ready clusters
		glog.V(2).Infof("There are no ready federated clusters, hpa reconcile cannot function")
		return false
	}
	return true
}

func (fac *AutoscalerController) reconcileAutoscaler(hpa *autoscalingv1.HorizontalPodAutoscaler) error {

	if !fac.isSynced() {
		return nil
	}

	//TODO: check if this is really needed
	if hpa.DeletionTimestamp != nil {
		err := fac.delete(hpa)
		if err != nil {
			glog.Errorf("Failed to delete %s: %v", hpa, err)
			fac.eventRecorder.Eventf(hpa, api.EventTypeNormal, "DeleteFailed",
				"HPA delete failed: %v", err)
		}
		return err
	}

	reference := fmt.Sprintf("%s/%s/%s", hpa.Spec.ScaleTargetRef.Kind, hpa.Namespace, hpa.Spec.ScaleTargetRef.Name)
	scale, err := fac.fedClient.ExtensionsV1beta1().Scales(hpa.Namespace).Get(hpa.Spec.ScaleTargetRef.Kind, hpa.Spec.ScaleTargetRef.Name)
	if err != nil {
		fac.eventRecorder.Event(hpa, apiv1.EventTypeWarning, "FailedGetScale", err.Error())
		fmt.Errorf("failed to query scale subresource for %s: %v", reference, err)
	}
	currentReplicas := scale.Status.Replicas

	cpuDesiredReplicas := int32(0)
	cpuCurrentUtilization := new(int32)
	//cpuTimestamp := time.Time{}

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
		// All basic scenarios covered, the state should be sane, lets use metrics.
		//cmAnnotation, cmAnnotationFound := hpa.Annotations[HpaCustomMetricsTargetAnnotationName]
		_, cmAnnotationFound := hpa.Annotations[HpaCustomMetricsTargetAnnotationName]

		if hpa.Spec.TargetCPUUtilizationPercentage != nil || !cmAnnotationFound {
			//cpuDesiredReplicas, cpuCurrentUtilization, cpuTimestamp, err = fac.computeReplicasForCPUUtilization(hpa, scale)
			cpuDesiredReplicas, cpuCurrentUtilization, timestamp, err = fac.computeReplicasForCPUUtilization(hpa, scale)
			if err != nil {
				fac.updateCurrentReplicasInStatus(hpa, currentReplicas)
				return fmt.Errorf("failed to compute desired number of replicas based on CPU utilization for %s: %v", reference, err)
			}
		}

		if cmAnnotationFound {
			//cmDesiredReplicas, cmMetric, cmStatus, cmTimestamp, err = a.computeReplicasForCustomMetrics(hpa, scale, cmAnnotation)
			//if err != nil {
			//	//fac.updateCurrentReplicasInStatus(hpa, currentReplicas)
			//	return fmt.Errorf("failed to compute desired number of replicas based on Custom Metrics for %s: %v", reference, err)
			//}
		}

		rescaleMetric := ""
		if cpuDesiredReplicas > desiredReplicas {
			desiredReplicas = cpuDesiredReplicas
			//timestamp = cpuTimestamp
			rescaleMetric = "CPU utilization"
		}
		/*if cmDesiredReplicas > desiredReplicas {
			desiredReplicas = cmDesiredReplicas
			timestamp = cmTimestamp
			rescaleMetric = cmMetric
		}*/
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
		_, err = fac.fedClient.ExtensionsV1beta1().Scales(hpa.Namespace).Update(hpa.Spec.ScaleTargetRef.Kind, scale)
		if err != nil {
			fac.eventRecorder.Eventf(hpa, apiv1.EventTypeWarning, "FailedRescale", "New size: %d; reason: %s; error: %v", desiredReplicas, rescaleReason, err.Error())
			return fmt.Errorf("failed to rescale %s: %v", reference, err)
		}
		fac.eventRecorder.Eventf(hpa, apiv1.EventTypeNormal, "SuccessfulRescale", "New size: %d; reason: %s", desiredReplicas, rescaleReason)
		glog.Infof("Successfull rescale of %s, old size: %d, new size: %d, reason: %s",
			hpa.Name, currentReplicas, desiredReplicas, rescaleReason)
	} else {
		desiredReplicas = currentReplicas
	}

	//return fac.updateStatus(hpa, currentReplicas, desiredReplicas, cpuCurrentUtilization, cmStatus, rescale)
	return fac.updateStatus(hpa, currentReplicas, desiredReplicas, cpuCurrentUtilization, "", rescale)
}

// getLastScaleTime returns the hpa's last scale time or the hpa's creation time if the last scale time is nil.
func getLastScaleTime(hpa *autoscalingv1.HorizontalPodAutoscaler) time.Time {
	lastScaleTime := hpa.Status.LastScaleTime
	if lastScaleTime == nil {
		lastScaleTime = &hpa.CreationTimestamp
	}
	return lastScaleTime.Time
}

func (fac *AutoscalerController) computeReplicasForCPUUtilization(hpa *autoscalingv1.HorizontalPodAutoscaler, scale *extensionsv1beta1.Scale) (int32, *int32, time.Time, error) {
	targetUtilization := int32(defaultTargetCPUUtilizationPercentage)
	if hpa.Spec.TargetCPUUtilizationPercentage != nil {
		targetUtilization = *hpa.Spec.TargetCPUUtilizationPercentage
	}
	currentReplicas := scale.Status.Replicas

	if scale.Status.Selector == nil {
		errMsg := "selector is required"
		fac.eventRecorder.Event(hpa, apiv1.EventTypeWarning, "SelectorRequired", errMsg)
		return 0, nil, time.Time{}, fmt.Errorf(errMsg)
	}

	selector, err := metav1.LabelSelectorAsSelector(&metav1.LabelSelector{MatchLabels: scale.Status.Selector})
	if err != nil {
		errMsg := fmt.Sprintf("couldn't convert selector string to a corresponding selector object: %v", err)
		fac.eventRecorder.Event(hpa, apiv1.EventTypeWarning, "InvalidSelector", errMsg)
		return 0, nil, time.Time{}, fmt.Errorf(errMsg)
	}

	desiredReplicas, utilization, timestamp, err := fac.calculateResourceReplicas(currentReplicas, targetUtilization, apiv1.ResourceCPU, hpa.Namespace, selector)
	if err != nil {
		lastScaleTime := getLastScaleTime(hpa)
		if time.Now().After(lastScaleTime.Add(upscaleForbiddenWindow)) {
			fac.eventRecorder.Event(hpa, apiv1.EventTypeWarning, "FailedGetMetrics", err.Error())
		} else {
			fac.eventRecorder.Event(hpa, apiv1.EventTypeNormal, "MetricsNotAvailableYet", err.Error())
		}

		return 0, nil, time.Time{}, fmt.Errorf("failed to get CPU utilization: %v", err)
	}

	if desiredReplicas != currentReplicas {
		fac.eventRecorder.Eventf(hpa, apiv1.EventTypeNormal, "DesiredReplicasComputed",
			"Computed the desired num of replicas: %d (avgCPUutil: %d, current replicas: %d)",
			desiredReplicas, utilization, scale.Status.Replicas)
	}

	return desiredReplicas, &utilization, timestamp, nil
}

func (fac *AutoscalerController) updateCurrentReplicasInStatus(hpa *autoscalingv1.HorizontalPodAutoscaler, currentReplicas int32) {
	err := fac.updateStatus(hpa, currentReplicas, hpa.Status.DesiredReplicas, hpa.Status.CurrentCPUUtilizationPercentage, hpa.Annotations[HpaCustomMetricsStatusAnnotationName], false)
	if err != nil {
		glog.Errorf("%v", err)
	}
}

func shouldScale(hpa *autoscalingv1.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, timestamp time.Time) bool {
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

func (fac *AutoscalerController) updateStatus(hpa *autoscalingv1.HorizontalPodAutoscaler, currentReplicas, desiredReplicas int32, cpuCurrentUtilization *int32, cmStatus string, rescale bool) error {
	hpa.Status = autoscalingv1.HorizontalPodAutoscalerStatus{
		CurrentReplicas:                 currentReplicas,
		DesiredReplicas:                 desiredReplicas,
		CurrentCPUUtilizationPercentage: cpuCurrentUtilization,
		LastScaleTime:                   hpa.Status.LastScaleTime,
	}
	if cmStatus != "" {
		hpa.Annotations[HpaCustomMetricsStatusAnnotationName] = cmStatus
	}

	if rescale {
		now := metav1.NewTime(time.Now())
		hpa.Status.LastScaleTime = &now
	}

	_, err := fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).UpdateStatus(hpa)
	if err != nil {
		fac.eventRecorder.Event(hpa, apiv1.EventTypeWarning, "FailedUpdateStatus", err.Error())
		return fmt.Errorf("failed to update status for %s: %v", hpa.Name, err)
	}
	glog.V(2).Infof("Successfully updated status for %s", hpa.Name)
	return nil
}

// delete deletes the given hpa or returns error if the deletion was not complete.
func (fac *AutoscalerController) delete(hpa *autoscalingv1.HorizontalPodAutoscaler) error {
	err := fac.fedClient.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Delete(hpa.Name, nil)
	if err != nil {
		// Its all good if the error type is not found. That means it is deleted already and we do not have to do anything.
		if !errors.IsNotFound(err) {
			return fmt.Errorf("failed to delete hpa: %s/%s, %v\n", hpa.Namespace, hpa.Name, err)
		}
	}
	return nil
}

type clusterMetricDetails struct {
	utilisation float64
	podCount    int32
	unreadyPods sets.String
	missingPods sets.String
}

// calculateResourceReplicas calculates the desired replica count based on a target resource utilization percentage
// of the given resource for pods of all federated clusters matching the given selector in the given
// namespace, and the current replica count
func (fac *AutoscalerController) calculateResourceReplicas(currentReplicas int32, targetUtilization int32, resource apiv1.ResourceName, namespace string, selector labels.Selector) (int32, int32, time.Time, error) {
	clusters, err := fac.autoscalerPodInformer.GetReadyClusters()
	if err != nil {
		return 0, 0, time.Time{}, fmt.Errorf("Autoscaler: Unable to get ready clusters in federation")
	}
	if len(clusters) == 0 {
		return 0, 0, time.Time{}, fmt.Errorf("Autoscaler: There are no clusters in federation")
	}

	//TODO: this map would be used for next phase of this work,
	//where this info can be passed as weighted annotations to target objects
	allClusterMetrics := make(map[string]*clusterMetricDetails, 0)
	timestamp, _ := time.Parse(time.RFC822, "01 Jan 01 10:00 UTC")
	for _, cluster := range clusters {
		unreadyPods := sets.NewString()
		missingPods := sets.NewString()
		//readyPodCount := 0

		clusterclient, err := fac.autoscalerPodInformer.GetClientsetForCluster(cluster.Name)
		if err != nil {
			//this ideally should not happen as informer gave this as a ready cluster
			//lets skip this pass and try again in the next reconcile
			glog.Errorf("Autoscaler: Pod Informer unable to get cluster client for cluster %s", cluster.Name)
			return 0, 0, time.Time{}, err
		}

		metrics, t, err := fac.metricsClient.GetClusterResourceMetrics(clusterclient, resource, namespace, selector)
		if err != nil {
			glog.Warning("Autoscaler: Unable to get metrics for cluster %s for resource %s: %v", cluster.Name, resource, err)
			continue
		}

		//as we get multiple timestamps from multiple clusters
		//its safest to pick up the latest time among all
		if t.After(timestamp) {
			timestamp = t
		}

		podList, err := fac.getPodListBySelector(cluster.Name, selector)
		if err != nil {
			glog.Warning("Autoscaler: unable to get pods while calculating replica count for cluster %s: %v", cluster.Name, err)
			continue
		}

		if len(podList) == 0 {
			glog.Warning("Autoscaler: no pods returned by selector while calculating replica count for cluster %s", cluster.Name)
			continue
		}

		requests := make(map[string]int64, len(podList))
		for _, pod := range podList {
			podSum := int64(0)
			for _, container := range pod.Spec.Containers {
				if containerRequest, ok := container.Resources.Requests[resource]; ok {
					podSum += containerRequest.MilliValue()
				} else {
					//This is a peculiar scenario.
					//because we are talking about pods of one common target object
					//resource request missing on one container in a pod in one cluster
					//would mean, the same spec replicated in other clusters
					//so no point trying for other clusters
					return 0, 0, time.Time{}, fmt.Errorf("missing request for %s on container %s in pod %s/%s", resource, container.Name, namespace, pod.Name)
				}
			}

			requests[pod.Name] = podSum

			if pod.Status.Phase != apiv1.PodRunning || !apiv1.IsPodReady(pod) {
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
		}

		if len(metrics) == 0 {
			//TODO: IRFAN this condition needs to be checked for each cluster
			//doesn't seem right to omit this cluster from calculations
			glog.Warning("Autoscaler: did not receive metrics for any ready pods for cluster: %s", cluster.Name)
			continue
		}

		allClusterMetrics[cluster.Name] = &clusterMetricDetails{
			utilisation: 0,
			podCount:    0,
			unreadyPods: unreadyPods,
			missingPods: missingPods,
		}
		allClusterMetrics[cluster.Name].utilisation, allClusterMetrics[cluster.Name].podCount, err =
			fedmetricsclient.GetAvgUtilizationForCluster(metrics, requests)
		if err != nil {
			return 0, 0, time.Time{}, err
		}
	}

	//TODO: IRFAN: There still is a need of handling some conditions.
	//we need to handle cases where for some reason there
	//are missing or unready pods in above metrics calculations.
	//Also we need to check a case where complete metrics info
	//is missing for a cluster
	//basically this whole function might need a rewrite.. :-)

	overallUsageTot := float64(0)
	utilisationSum := float64(0)
	totUnreadyPods := 0
	totMissingPods := 0
	for _, value := range allClusterMetrics {
		overallUsageTot += (value.utilisation * float64(value.podCount))
		utilisationSum += value.utilisation
		totUnreadyPods += len(value.unreadyPods)
		totMissingPods += len(value.missingPods)
	}

	targetReplicas := int32(math.Ceil(overallUsageTot / float64(targetUtilization)))
	allClusterAverageUtilisation := int32(utilisationSum / float64(len(clusters)))
	usageRatio := float64(allClusterAverageUtilisation) / float64(targetUtilization)

	rebalanceUnready := totUnreadyPods > 0 && usageRatio > 1.0
	if !rebalanceUnready && totMissingPods == 0 {
		if math.Abs(1.0-usageRatio) <= tolerance {
			// return the current replicas if the change would be too small
			return currentReplicas, allClusterAverageUtilisation, timestamp, nil
		}
	}

	//the target total replicas and averaged current utilisation per pod across all clusters
	return targetReplicas, allClusterAverageUtilisation, timestamp, nil
}

func (fac *AutoscalerController) getPodListBySelector(cluster string, selector labels.Selector) ([]*apiv1.Pod, error) {

	allPods, err := fac.autoscalerPodInformer.GetTargetStore().ListFromCluster(cluster)
	if err != nil {
		return nil, fmt.Errorf("error in listing pods from cluster %s: %v", cluster, err)
	}

	result := make([]*apiv1.Pod, 0)
	for _, fedObject := range allPods {
		pod, isPod := fedObject.(*apiv1.Pod)
		if !isPod {
			return nil, fmt.Errorf("invalid arg content - not a *pod")
		}
		if !selector.Empty() && selector.Matches(labels.Set(pod.Labels)) {
			result = append(result, pod)
		}
	}
	return result, nil
}

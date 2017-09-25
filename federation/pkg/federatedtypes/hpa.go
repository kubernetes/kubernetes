/*
Copyright 2017 The Kubernetes Authors.

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

package federatedtypes

import (
	"fmt"
	"time"

	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	kubeclientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	federationapi "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	fedutil "k8s.io/kubernetes/federation/pkg/federation-controller/util"
	hpautil "k8s.io/kubernetes/federation/pkg/federation-controller/util/hpa"
	extensionsinternal "k8s.io/kubernetes/pkg/apis/extensions"

	"github.com/golang/glog"
)

const (
	HpaKind           = "horizontalpodautoscaler"
	HpaControllerName = "horizontalpodautoscalers"
	// This is used as the default min for hpa object submitted
	// to federation, in a situation where the default is for
	// some reason not present (Spec.MinReplicas == nil)
	hpaMinReplicaDefault = int32(1)
	// This is a tunable which does not change replica nums
	// on an existing local hpa, before this timeout, if it
	// did scale already (avoids thrashing of replicas around).
	ScaleForbiddenWindow = 2 * time.Minute
)

func init() {
	RegisterFederatedType(HpaKind, HpaControllerName, []schema.GroupVersionResource{autoscalingv1.SchemeGroupVersion.WithResource(HpaControllerName)}, NewHpaAdapter)
}

type HpaAdapter struct {
	client               federationclientset.Interface
	scaleForbiddenWindow time.Duration
}

func NewHpaAdapter(client federationclientset.Interface, config *restclient.Config, adapterSpecificArgs map[string]interface{}) FederatedTypeAdapter {
	var scaleForbiddenWindow time.Duration
	if adapterSpecificArgs != nil && adapterSpecificArgs[HpaKind] != nil {
		scaleForbiddenWindow = adapterSpecificArgs[HpaKind].(*metav1.Duration).Duration
	} else {
		scaleForbiddenWindow = ScaleForbiddenWindow
	}

	return &HpaAdapter{
		client:               client,
		scaleForbiddenWindow: scaleForbiddenWindow,
	}
}

func (a *HpaAdapter) Kind() string {
	return HpaKind
}

func (a *HpaAdapter) ObjectType() pkgruntime.Object {
	return &autoscalingv1.HorizontalPodAutoscaler{}
}

func (a *HpaAdapter) IsExpectedType(obj interface{}) bool {
	_, ok := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return ok
}

func (a *HpaAdapter) Copy(obj pkgruntime.Object) pkgruntime.Object {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: fedutil.DeepCopyRelevantObjectMeta(hpa.ObjectMeta),
		Spec:       *fedutil.DeepCopyApiTypeOrPanic(&hpa.Spec).(*autoscalingv1.HorizontalPodAutoscalerSpec),
	}
}

func (a *HpaAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	return fedutil.ObjectMetaAndSpecEquivalent(obj1, obj2)
}

func (a *HpaAdapter) QualifiedName(obj pkgruntime.Object) QualifiedName {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return QualifiedName{Namespace: hpa.Namespace, Name: hpa.Name}
}

func (a *HpaAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*autoscalingv1.HorizontalPodAutoscaler).ObjectMeta
}

func (a *HpaAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Create(hpa)
}

func (a *HpaAdapter) FedDelete(qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *HpaAdapter) FedGet(qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *HpaAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespace).List(options)
}

func (a *HpaAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Update(hpa)
}

func (a *HpaAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespace).Watch(options)
}

func (a *HpaAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Create(hpa)
}

func (a *HpaAdapter) ClusterDelete(client kubeclientset.Interface, qualifiedName QualifiedName, options *metav1.DeleteOptions) error {
	return client.AutoscalingV1().HorizontalPodAutoscalers(qualifiedName.Namespace).Delete(qualifiedName.Name, options)
}

func (a *HpaAdapter) ClusterGet(client kubeclientset.Interface, qualifiedName QualifiedName) (pkgruntime.Object, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(qualifiedName.Namespace).Get(qualifiedName.Name, metav1.GetOptions{})
}

func (a *HpaAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(namespace).List(options)
}

func (a *HpaAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return client.AutoscalingV1().HorizontalPodAutoscalers(hpa.Namespace).Update(hpa)
}

func (a *HpaAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(namespace).Watch(options)
}

func (a *HpaAdapter) NewTestObject(namespace string) pkgruntime.Object {
	var min int32 = 4
	var targetCPU int32 = 70
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-hpa-",
			Namespace:    namespace,
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: "ReplicaSet",
				Name: "myrs",
			},
			MinReplicas:                    &min,
			MaxReplicas:                    int32(10),
			TargetCPUUtilizationPercentage: &targetCPU,
		},
	}
}

func (a *HpaAdapter) IsSchedulingAdapter() bool {
	return true
}

func (a *HpaAdapter) EquivalentIgnoringSchedule(obj1, obj2 pkgruntime.Object) bool {
	hpa1 := obj1.(*autoscalingv1.HorizontalPodAutoscaler)
	hpa2 := a.Copy(obj2).(*autoscalingv1.HorizontalPodAutoscaler)
	if hpa1.Spec.MinReplicas == nil {
		hpa2.Spec.MinReplicas = nil
	} else if hpa2.Spec.MinReplicas == nil {
		var r int32 = *hpa1.Spec.MinReplicas
		hpa2.Spec.MinReplicas = &r
	} else {
		*hpa2.Spec.MinReplicas = *hpa1.Spec.MinReplicas
	}
	hpa2.Spec.MaxReplicas = hpa1.Spec.MaxReplicas
	return fedutil.ObjectMetaAndSpecEquivalent(hpa1, hpa2)
}

type replicaNums struct {
	min int32
	max int32
}

type hpaFederatedStatus struct {
	lastScaleTime *metav1.Time
	// Indicates how many clusters have hpa/replicas.
	// Used to average the cpu utilization which is
	// reflected to the federation user.
	count                             int32
	aggregateCPUUtilizationPercentage *int32
	currentReplicas                   int32
	desiredReplicas                   int32
}

type hpaSchedulingInfo struct {
	scheduleState map[string]*replicaNums
	fedStatus     hpaFederatedStatus
}

// List of cluster names.
type hpaLists struct {
	// Stores names of those clusters which can offer min.
	availableMin sets.String
	// Stores names of those clusters which can offer max.
	availableMax sets.String
	// Stores names of those clusters which do not have hpa yet.
	noHpa sets.String
}

func (a *HpaAdapter) GetSchedule(obj pkgruntime.Object, key string, clusters []*federationapi.Cluster, informer fedutil.FederatedInformer) (interface{}, error) {
	currentClusterObjs, err := getCurrentClusterObjs(informer, key, clusters)
	if err != nil {
		return nil, err
	}

	// Initialise averaged cpu utilisation for this reconcile.
	var ccup int32 = 0
	fedStatus := hpaFederatedStatus{
		aggregateCPUUtilizationPercentage: &ccup,
		count:           int32(0),
		desiredReplicas: int32(0),
		currentReplicas: int32(0),
	}
	fedHpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	// We assign the last known scale time here, which we update with
	// the latest time from among all clusters in ScheduleObject()
	if fedHpa.Status.LastScaleTime != nil {
		t := metav1.NewTime(fedHpa.Status.LastScaleTime.Time)
		fedStatus.lastScaleTime = &t
	}

	return &hpaSchedulingInfo{
		scheduleState: a.getHpaScheduleState(obj, currentClusterObjs),
		fedStatus:     fedStatus,
	}, nil
}

func getCurrentClusterObjs(informer fedutil.FederatedInformer, key string, clusters []*federationapi.Cluster) (map[string]pkgruntime.Object, error) {
	currentClusterObjs := make(map[string]pkgruntime.Object)
	for _, cluster := range clusters {
		clusterName := cluster.Name
		clusterObj, found, err := informer.GetTargetStore().GetByKey(clusterName, key)
		if err != nil {
			return nil, err
		}
		currentClusterObjs[clusterName] = nil
		if found {
			currentClusterObjs[clusterName] = clusterObj.(pkgruntime.Object)
		}
	}
	return currentClusterObjs, nil
}

// The algorithm used for scheduling is briefed as below:
//
// 1. Find clusters which can offer max and min, if any (lists.availableMax and
// lists.availableMin) in one pass on all clusters.
//
// 2. Reduce the replicas (both min and max) if needed (situation when fedHpa
// has lesser replicas then all cluster local hpa replicas totalled together).
// In this step reduce first from those hpas which already have max (and min)
// reducible. Once such clusters are over and reduction still needed, reduce
// one at a time from all clusters, randomly. This step will ensure that the
// exceeding replicas in local hpas are reduced to match the fedHpa.
// This step would ideally be a noop in most cases because its rare that fedHpa
// would have lesser replicas then the cluster local total (probably when user
// forces update if fedHpa).
//
// 3. Distribute the replicas. In this step we have replicas to distribute (which
// are fed replicas exceeding the sum total of local cluster replicas). If clusters
// already have replicas, one replica from each cluster which can offer replicas
// (both for max and min) are also added to this replicas to distribute numbers (min
// and max).
// 3a. We first do a sub-pass to distribute to clusters which need replicas, considering
// those as clusters in crucial need of replicas.
// 3b. After previous sub-pass, if we still have replicas remaining, in the sub-pass
// we distribute to those clusters which do not yet have any hpa.
// 3c. After previous if we still have more to distribute, then we distribute to all
// clusters randomly, giving replica distribution count (rdc=total-fed-replicas/no-of-clusters)
// to each at a time.
//
// The above algorithm is run to first distribute max and then distribute min to those clusters
// which get max.
func (a *HpaAdapter) getHpaScheduleState(fedObj pkgruntime.Object, currentObjs map[string]pkgruntime.Object) map[string]*replicaNums {
	fedHpa := fedObj.(*autoscalingv1.HorizontalPodAutoscaler)
	requestedMin := hpaMinReplicaDefault
	if fedHpa.Spec.MinReplicas != nil {
		requestedMin = *fedHpa.Spec.MinReplicas
	}
	requestedReplicas := replicaNums{
		min: requestedMin,
		max: fedHpa.Spec.MaxReplicas,
	}
	// replica distribution count, per cluster
	rdc := replicaNums{
		min: requestedReplicas.min / int32(len(currentObjs)),
		max: requestedReplicas.max / int32(len(currentObjs)),
	}
	if rdc.min < 1 {
		rdc.min = 1
	}
	// TODO: Is there a better way?
	// We need to cap the lowest limit of Max to 2, because in a
	// situation like both min and max become 1 (same) for all clusters,
	// no rebalancing would happen.
	if rdc.max < 2 {
		rdc.max = 2
	}

	// Pass 1: Analyse existing local hpa's if any.
	// clusterLists holds the list of those clusters which can offer
	// min and max replicas, to those which want them.
	// For example new clusters joining the federation and/or
	// those clusters which need to increase or reduce replicas
	// beyond min/max limits.
	// schedStatus currently have status of existing hpas.
	// It will eventually have desired status for this reconcile.
	clusterLists, currentReplicas, scheduleState := a.prepareForScheduling(currentObjs)

	remainingReplicas := replicaNums{
		min: requestedReplicas.min - currentReplicas.min,
		max: requestedReplicas.max - currentReplicas.max,
	}

	// Pass 2: reduction of replicas if needed ( situation that fedHpa updated replicas
	// to lesser then existing).
	// In this pass, we remain pessimistic and reduce one replica per cluster at a time.
	if remainingReplicas.min < 0 {
		excessMin := (remainingReplicas.min * int32(-1))
		remainingReplicas.min = reduceMinReplicas(excessMin, clusterLists.availableMin, scheduleState)
	}
	if remainingReplicas.max < 0 {
		excessMax := (remainingReplicas.max * int32(-1))
		remainingReplicas.max = reduceMaxReplicas(excessMax, clusterLists.availableMax, scheduleState)
	}

	toDistribute := replicaNums{
		min: remainingReplicas.min + int32(clusterLists.availableMin.Len()),
		max: remainingReplicas.max + int32(clusterLists.availableMax.Len()),
	}

	// Pass 3: Distribute Max and then Min.
	// Here we first distribute max and then (in the next loop)
	// distribute min into those clusters which already get the
	// max fixed.
	// In this process we might not meet the min limit and total of
	// min limits might remain more then the requested federated min.
	// This is partially because a min per cluster cannot be lesser
	// then 1, but min could be requested as 1 at federation.
	// Additionally we first increase replicas into those clusters
	// which already have hpa's and are in a condition to increase.
	// This will save cluster related resources for the user, such that
	// if an already existing cluster can satisfy users request why send
	// the workload to another.
	// We then go ahead to give the replicas to those which do not
	// have any hpa. In this pass however we try to ensure that all
	// our Max are consumed in this reconcile.
	a.distributeMaxReplicas(toDistribute.max, clusterLists, rdc, currentObjs, scheduleState)

	// We distribute min to those clusters which:
	// 1 - can adjust min (our increase step would be only 1)
	// 2 - which do not have this hpa and got max(increase step rdcMin)
	// We might exhaust all min replicas here, with
	// some clusters still needing them. We adjust this in finalise by
	// assigning min replicas to 1 into those clusters which got max
	// but min remains 0.
	a.distributeMinReplicas(toDistribute.min, clusterLists, rdc, currentObjs, scheduleState)

	return finaliseScheduleState(scheduleState)
}

func (a *HpaAdapter) ScheduleObject(cluster *federationapi.Cluster, clusterObj pkgruntime.Object, federationObjCopy pkgruntime.Object, schedulingInfo interface{}) (pkgruntime.Object, ScheduleAction, error) {
	// Update federated status info
	typedInfo := schedulingInfo.(*hpaSchedulingInfo)
	if clusterObj != nil {
		clusterHpa := clusterObj.(*autoscalingv1.HorizontalPodAutoscaler)
		if clusterHpa.Status.CurrentCPUUtilizationPercentage != nil {
			*typedInfo.fedStatus.aggregateCPUUtilizationPercentage +=
				(*clusterHpa.Status.CurrentCPUUtilizationPercentage * clusterHpa.Status.CurrentReplicas)
			typedInfo.fedStatus.count += clusterHpa.Status.CurrentReplicas
		}
		if clusterHpa.Status.LastScaleTime != nil {
			t := metav1.NewTime(clusterHpa.Status.LastScaleTime.Time)
			if typedInfo.fedStatus.lastScaleTime != nil &&
				t.After(typedInfo.fedStatus.lastScaleTime.Time) {
				typedInfo.fedStatus.lastScaleTime = &t
			}
		}

		typedInfo.fedStatus.currentReplicas += clusterHpa.Status.CurrentReplicas
		typedInfo.fedStatus.desiredReplicas += clusterHpa.Status.DesiredReplicas
	}

	// Update the cluster obj and the needed action on the cluster
	clusterHpaState := typedInfo.scheduleState[cluster.Name]
	desiredHpa := federationObjCopy.(*autoscalingv1.HorizontalPodAutoscaler)
	if clusterHpaState != nil {
		desiredHpa.Spec.MaxReplicas = clusterHpaState.max
		if desiredHpa.Spec.MinReplicas == nil {
			min := int32(0)
			desiredHpa.Spec.MinReplicas = &min
		}
		*desiredHpa.Spec.MinReplicas = clusterHpaState.min
	}

	var defaultAction ScheduleAction = ""
	switch {
	case clusterHpaState != nil && clusterObj != nil:
		return desiredHpa, defaultAction, nil
	case clusterHpaState != nil && clusterObj == nil:
		return desiredHpa, ActionAdd, nil
	case clusterHpaState == nil && clusterObj != nil:
		return nil, ActionDelete, nil
	}
	return nil, defaultAction, nil
}

func (a *HpaAdapter) UpdateFederatedStatus(obj pkgruntime.Object, schedulingInfo interface{}) error {
	fedHpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	needUpdate, newFedHpaStatus := updateStatus(fedHpa, schedulingInfo.(*hpaSchedulingInfo).fedStatus)
	if needUpdate {
		fedHpa.Status = newFedHpaStatus
		_, err := a.client.AutoscalingV1().HorizontalPodAutoscalers(fedHpa.Namespace).UpdateStatus(fedHpa)
		if err != nil {
			return fmt.Errorf("Error updating hpa: %s status in federation: %v", fedHpa.Name, err)
		}
	}

	if err := a.updateClusterListOnTargetObject(fedHpa, schedulingInfo.(*hpaSchedulingInfo).scheduleState); err != nil {
		return fmt.Errorf("Error updating cluster list on object targetted by hpa: %s: %v", fedHpa.Name, err)
	}
	return nil
}

func updateStatus(fedHpa *autoscalingv1.HorizontalPodAutoscaler, newStatus hpaFederatedStatus) (bool, autoscalingv1.HorizontalPodAutoscalerStatus) {
	averageCPUUtilizationPercentage := int32(0)
	// Average out the available current utilisation
	if *newStatus.aggregateCPUUtilizationPercentage != 0 && newStatus.count != 0 {
		averageCPUUtilizationPercentage = *newStatus.aggregateCPUUtilizationPercentage / newStatus.count
	}
	gen := fedHpa.Generation
	newFedHpaStatus := autoscalingv1.HorizontalPodAutoscalerStatus{ObservedGeneration: &gen}
	needUpdate := false
	if (fedHpa.Status.CurrentCPUUtilizationPercentage == nil &&
		averageCPUUtilizationPercentage != 0) ||
		(fedHpa.Status.CurrentCPUUtilizationPercentage != nil &&
			averageCPUUtilizationPercentage !=
				*fedHpa.Status.CurrentCPUUtilizationPercentage) {
		needUpdate = true
		newFedHpaStatus.CurrentCPUUtilizationPercentage = &averageCPUUtilizationPercentage
	}
	if (fedHpa.Status.LastScaleTime == nil && newStatus.lastScaleTime != nil) ||
		(fedHpa.Status.LastScaleTime != nil && newStatus.lastScaleTime == nil) ||
		((fedHpa.Status.LastScaleTime != nil && newStatus.lastScaleTime != nil) &&
			newStatus.lastScaleTime.After(fedHpa.Status.LastScaleTime.Time)) {
		needUpdate = true
		newFedHpaStatus.LastScaleTime = newStatus.lastScaleTime
	}
	if fedHpa.Status.DesiredReplicas != newStatus.desiredReplicas {
		needUpdate = true
		newFedHpaStatus.CurrentReplicas = newStatus.currentReplicas
	}
	if fedHpa.Status.CurrentReplicas != newStatus.currentReplicas {
		needUpdate = true
		newFedHpaStatus.DesiredReplicas = newStatus.desiredReplicas
	}
	return needUpdate, newFedHpaStatus
}

// prepareForScheduling prepares the lists and totals from the
// existing objs.
// currentObjs has the list of all clusters, with obj as nil
// for those clusters which do not have hpa yet.
func (a *HpaAdapter) prepareForScheduling(currentObjs map[string]pkgruntime.Object) (hpaLists, replicaNums, map[string]*replicaNums) {
	lists := hpaLists{
		availableMax: sets.NewString(),
		availableMin: sets.NewString(),
		noHpa:        sets.NewString(),
	}
	existingTotal := replicaNums{
		min: int32(0),
		max: int32(0),
	}

	scheduleState := make(map[string]*replicaNums)
	for cluster, obj := range currentObjs {
		if obj == nil {
			lists.noHpa.Insert(cluster)
			scheduleState[cluster] = nil
			continue
		}

		if a.maxReplicasReducible(obj) {
			lists.availableMax.Insert(cluster)
		}
		if a.minReplicasReducible(obj) {
			lists.availableMin.Insert(cluster)
		}

		replicas := replicaNums{min: 0, max: 0}
		scheduleState[cluster] = &replicas
		if obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas != nil {
			existingTotal.min += *obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas
			replicas.min = *obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas
		}
		existingTotal.max += obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MaxReplicas
		replicas.max = obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MaxReplicas
	}

	return lists, existingTotal, scheduleState
}

// Note: reduceMinReplicas and reduceMaxReplicas, look quite similar in flow
// and code, however there are subtle differences. They together can be made
// into 1 function with an arg governing the functionality difference and
// additional args (superset of args in both) as needed. Doing so however
// makes the logical flow quite less readable. They are thus left as 2 for
// readability.

// reduceMinReplicas reduces the min replicas from existing clusters.
// At the end of the function excessMin should be 0 and the MinList
// and the scheduledReplicas properly updated in place.
func reduceMinReplicas(excessMin int32, availableMinList sets.String, scheduled map[string]*replicaNums) int32 {
	if excessMin > 0 {
		// first we try reducing from those clusters which already offer min
		if availableMinList.Len() > 0 {
			for _, cluster := range availableMinList.List() {
				replicas := scheduled[cluster]
				if replicas.min > 1 {
					replicas.min--
					availableMinList.Delete(cluster)
					excessMin--
					if excessMin <= 0 {
						break
					}
				}
			}
		}
	}

	// If we could not get needed replicas from already offered min above
	// we abruptly start removing replicas from some/all clusters.
	// Here we might make some min to 0 signalling that this hpa might be a
	// candidate to be removed from this cluster altogether.
	for excessMin > 0 {
		for _, replicas := range scheduled {
			if replicas != nil &&
				replicas.min > 0 {
				replicas.min--
				excessMin--
				if excessMin <= 0 {
					break
				}
			}
		}
	}

	return excessMin
}

// reduceMaxReplicas reduces the max replicas from existing clusters.
// At the end of the function excessMax should be 0 and the MaxList
// and the scheduledReplicas properly updated in place.
func reduceMaxReplicas(excessMax int32, availableMaxList sets.String, scheduled map[string]*replicaNums) int32 {
	if excessMax > 0 {
		// first we try reducing from those clusters which already offer max
		if availableMaxList.Len() > 0 {
			for _, cluster := range availableMaxList.List() {
				replicas := scheduled[cluster]
				if replicas != nil && !((replicas.max - replicas.min) < 0) {
					replicas.max--
					availableMaxList.Delete(cluster)
					excessMax--
					if excessMax <= 0 {
						break
					}
				}
			}
		}
	}
	// If we could not get needed replicas to reduce from already offered
	// max above we abruptly start removing replicas from some/all clusters.
	// Here we might make some max and min to 0, signalling that this hpa be
	// removed from this cluster altogether
	for excessMax > 0 {
		for _, replicas := range scheduled {
			if replicas != nil &&
				!((replicas.max - replicas.min) < 0) {
				replicas.max--
				excessMax--
				if excessMax <= 0 {
					break
				}
			}
		}
	}

	return excessMax
}

// distributeMaxReplicas
// Takes input:
// toDistributeMax: number of replicas to distribute.
// lists: cluster name lists, which have clusters with available max,
//	available min and those with no hpas yet.
// rdc: replicadistributioncount for max and min.
// currentObjs: list of current cluster hpas.
// scheduled: schedule state which will be updated in place.
func (a *HpaAdapter) distributeMaxReplicas(toDistributeMax int32, lists hpaLists, rdc replicaNums,
	currentObjs map[string]pkgruntime.Object, scheduled map[string]*replicaNums) int32 {
	for cluster, replicas := range scheduled {
		if toDistributeMax == 0 {
			break
		}
		if replicas == nil {
			continue
		}
		if a.maxReplicasNeeded(currentObjs[cluster]) {
			replicas.max++
			if lists.availableMax.Len() > 0 {
				popped, notEmpty := lists.availableMax.PopAny()
				if notEmpty {
					// Boundary checks have happened earlier in
					// minReplicasReducible().
					scheduled[popped].max--
				}
			}
			// Any which ways utilise available map replicas
			toDistributeMax--
		}
	}

	// If we have new clusters where we can  give our replicas,
	// then give away all our replicas to the new clusters first.
	if lists.noHpa.Len() > 0 {
		for toDistributeMax > 0 {
			for _, cluster := range lists.noHpa.UnsortedList() {
				if scheduled[cluster] == nil {
					scheduled[cluster] = &replicaNums{min: 0, max: 0}
				}
				replicas := scheduled[cluster]
				// first give away max from clusters offering them
				// this case especially helps getting hpa into newly joining
				// clusters.
				if lists.availableMax.Len() > 0 {
					popped, notEmpty := lists.availableMax.PopAny()
					if notEmpty {
						// Boundary checks to reduce max have happened earlier in
						// minReplicasReducible().
						replicas.max++
						scheduled[popped].max--
						toDistributeMax--
						continue
					}
				}
				if toDistributeMax < rdc.max {
					replicas.max += toDistributeMax
					toDistributeMax = 0
					break
				}
				replicas.max += rdc.max
				toDistributeMax -= rdc.max
			}
		}
	} else { // we have no new clusters but if still have max replicas to distribute;
		// just distribute all in current clusters.
		for toDistributeMax > 0 {
			for cluster, replicas := range scheduled {
				if replicas == nil {
					replicas = &replicaNums{min: 0, max: 0}
					scheduled[cluster] = replicas
				}
				// First give away max from clusters offering them.
				// This case especially helps getting hpa into newly joining
				// clusters.
				if lists.availableMax.Len() > 0 {
					popped, notEmpty := lists.availableMax.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier in
						// minReplicasReducible().
						replicas.max++
						scheduled[popped].max--
						toDistributeMax--
						continue
					}
				}
				if toDistributeMax < rdc.max {
					replicas.max += toDistributeMax
					toDistributeMax = 0
					break
				}
				replicas.max += rdc.max
				toDistributeMax -= rdc.max
			}
		}
	}
	return toDistributeMax
}

// distributeMinReplicas
// Takes input:
// toDistributeMin: number of replicas to distribute.
// lists: cluster name lists, which have clusters with available max,
//	available min and those with no hpas yet.
// rdc: replicadistributioncount for max and min.
// currentObjs: list of current cluster hpas.
// scheduled: schedule state which will be updated in place.
func (a *HpaAdapter) distributeMinReplicas(toDistributeMin int32, lists hpaLists, rdc replicaNums,
	currentObjs map[string]pkgruntime.Object, scheduled map[string]*replicaNums) int32 {
	for cluster, replicas := range scheduled {
		if toDistributeMin == 0 {
			break
		}
		// We have distriubted Max and thus scheduled might not be nil
		// but probably current (what we got originally) is nil(no hpa)
		if replicas == nil || currentObjs[cluster] == nil {
			continue
		}
		if a.minReplicasIncreasable(currentObjs[cluster]) {
			if lists.availableMin.Len() > 0 {
				popped, notEmpty := lists.availableMin.PopAny()
				if notEmpty {
					// Boundary checks have happened earlier.
					scheduled[popped].min--
					replicas.min++
					toDistributeMin--
				}
			}
		}
	}

	if lists.noHpa.Len() > 0 {
		// TODO: can this become an infinite loop?
		for toDistributeMin > 0 {
			for _, cluster := range lists.noHpa.UnsortedList() {
				replicas := scheduled[cluster]
				if replicas == nil {
					// We did not get max here so this cluster
					// remains without hpa
					continue
				}
				var replicaNum int32 = 0
				if toDistributeMin < rdc.min {
					replicaNum = toDistributeMin
				} else {
					replicaNum = rdc.min
				}
				if (replicas.max - replicaNum) < replicas.min {
					// Cannot increase the min in this cluster
					// as it will go beyond max
					continue
				}
				if lists.availableMin.Len() > 0 {
					popped, notEmpty := lists.availableMin.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier.
						scheduled[popped].min--
						replicas.min++
						toDistributeMin--
						continue
					}
				}
				replicas.min += replicaNum
				toDistributeMin -= replicaNum
			}
		}
	} else { // we have no new clusters but if still have min replicas to distribute;
		// just distribute all in current clusters.
		for toDistributeMin > 0 {
			for _, replicas := range scheduled {
				if replicas == nil {
					// We did not get max here so this cluster
					// remains without hpa
					continue
				}
				var replicaNum int32 = 0
				if toDistributeMin < rdc.min {
					replicaNum = toDistributeMin
				} else {
					replicaNum = rdc.min
				}
				if (replicas.max - replicaNum) < replicas.min {
					// Cannot increase the min in this cluster
					// as it will go beyond max
					continue
				}
				if lists.availableMin.Len() > 0 {
					popped, notEmpty := lists.availableMin.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier.
						scheduled[popped].min--
						replicas.min++
						toDistributeMin--
						continue
					}
				}
				replicas.min += replicaNum
				toDistributeMin -= replicaNum
			}
		}
	}
	return toDistributeMin
}

// finaliseScheduleState ensures that the minReplica count is made to 1
// for those clusters which got max, but did not get min. This is because
// k8s hpa does not accept hpas with 0 min replicas.
// The replica num distribution can thus have more mins then fedHpa requested
// but its better then having all replicas go into one cluster (if fedHpa
// requested min=1 (which is the most usual case).
func finaliseScheduleState(scheduled map[string]*replicaNums) map[string]*replicaNums {
	for _, replicas := range scheduled {
		if (replicas != nil) && (replicas.min <= 0) && (replicas.max > 0) {
			// Min total does not necessarily meet the federated min limit.
			replicas.min = 1
		}
	}
	return scheduled
}

// isPristine is used to determine if so far local controller has been
// able to really determine, what should be the desired replica number for
// this cluster.
// This is used to get hpas into those clusters which might join fresh,
// and so far other cluster hpas haven't really reached anywhere.
// TODO: There is a flaw here, that a just born object would also offer its
// replicas which can also lead to fast thrashing.
// The only better way is to either ensure that object creation time stamp is set
// and can be used authoritatively; or have another field on the local object
// which is mandatorily set on creation and can be used authoritatively.
// Should we abuse annotations again for this, or this can be a proper requirement?
func isPristine(hpa *autoscalingv1.HorizontalPodAutoscaler) bool {
	if hpa.Status.LastScaleTime == nil &&
		hpa.Status.DesiredReplicas == 0 {
		return true
	}
	return false
}

// isScaleable tells if it already has been a reasonable amount of
// time since this hpa scaled. Its used to avoid fast thrashing.
func (a *HpaAdapter) isScaleable(hpa *autoscalingv1.HorizontalPodAutoscaler) bool {
	if hpa.Status.LastScaleTime == nil {
		return false
	}
	t := hpa.Status.LastScaleTime.Add(a.scaleForbiddenWindow)
	if t.After(time.Now()) {
		return false
	}
	return true
}

func (a *HpaAdapter) maxReplicasReducible(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if (hpa.Spec.MinReplicas != nil) &&
		(((hpa.Spec.MaxReplicas - 1) - *hpa.Spec.MinReplicas) < 0) {
		return false
	}
	if isPristine(hpa) {
		return true
	}
	if !a.isScaleable(hpa) {
		return false
	}
	if (hpa.Status.DesiredReplicas < hpa.Status.CurrentReplicas) ||
		((hpa.Status.DesiredReplicas == hpa.Status.CurrentReplicas) &&
			(hpa.Status.DesiredReplicas < hpa.Spec.MaxReplicas)) {
		return true
	}
	return false
}

// minReplicasReducible checks if this cluster (hpa) can offer replicas which are
// stuck here because of min limit.
// Its noteworthy, that min and max are adjusted separately, but if the replicas
// are not being used here, the max adjustment will lead it to become equal to min,
// but will not be able to scale down further and offer max to some other cluster
// which needs replicas.
func (a *HpaAdapter) minReplicasReducible(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if isPristine(hpa) && (hpa.Spec.MinReplicas != nil) &&
		(*hpa.Spec.MinReplicas > 1) &&
		(*hpa.Spec.MinReplicas <= hpa.Spec.MaxReplicas) {
		return true
	}
	if !a.isScaleable(hpa) {
		return false
	}
	if (hpa.Spec.MinReplicas != nil) &&
		(*hpa.Spec.MinReplicas > 1) &&
		(hpa.Status.DesiredReplicas == hpa.Status.CurrentReplicas) &&
		(hpa.Status.CurrentReplicas == *hpa.Spec.MinReplicas) {
		return true
	}
	return false
}

func (a *HpaAdapter) maxReplicasNeeded(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if !a.isScaleable(hpa) {
		return false
	}

	if (hpa.Status.CurrentReplicas == hpa.Status.DesiredReplicas) &&
		(hpa.Status.CurrentReplicas == hpa.Spec.MaxReplicas) {
		return true
	}
	return false
}

func (a *HpaAdapter) minReplicasIncreasable(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if !a.isScaleable(hpa) ||
		((hpa.Spec.MinReplicas != nil) &&
			(*hpa.Spec.MinReplicas) >= hpa.Spec.MaxReplicas) {
		return false
	}

	if (hpa.Spec.MinReplicas != nil) &&
		(hpa.Status.DesiredReplicas > *hpa.Spec.MinReplicas) {
		return true
	}
	return false
}

// updateClusterListOnTargetObject passes the necessary info to the target object,
// so that the corresponding controller can act on that.
// This is used because if an hpa is active on a federated object it is supposed
// to control the replicas and presence/absence of target object from federated clusters.
func (a *HpaAdapter) updateClusterListOnTargetObject(fedHpa *autoscalingv1.HorizontalPodAutoscaler, scheduleStatus map[string]*replicaNums) error {
	if len(fedHpa.Spec.ScaleTargetRef.Kind) <= 0 || len(fedHpa.Spec.ScaleTargetRef.Name) <= 0 {
		// nothing to do
		glog.Infof("Fed HPA: cluster list update on target object skipped for target obj: %s, kind: %s", fedHpa.Spec.ScaleTargetRef.Name, fedHpa.Spec.ScaleTargetRef.Kind)
		return nil
	}

	names := []string{}
	for clusterName, replicas := range scheduleStatus {
		if replicas != nil {
			names = append(names, clusterName)
		}
	}
	clusterNames := hpautil.ClusterNames{Names: names}
	qualifiedKind := extensionsinternal.Kind(fedHpa.Spec.ScaleTargetRef.Kind)
	targetObj, err := getRuntimeObjectForKind(a.client, qualifiedKind, fedHpa.Namespace, fedHpa.Spec.ScaleTargetRef.Name)
	if errors.IsNotFound(err) {
		// Nothing to do; the target object does not exist in federation.
		glog.Infof("Fed HPA: cluster list update on target object skipped for target obj: %s, kind: %s. Target object missing in federation", fedHpa.Spec.ScaleTargetRef.Name, fedHpa.Spec.ScaleTargetRef.Kind)
		return nil
	}
	if err != nil {
		return err
	}

	updatedObj := hpautil.SetHpaTargetClusterList(targetObj, clusterNames)
	_, err = updateRuntimeObjectForKind(a.client, qualifiedKind, fedHpa.Namespace, updatedObj)
	if err != nil {
		return err
	}
	return nil
}

// getRuntimeObjectForKind gets the hpa targetted object from the federation control plane.
// As of now, federation only supports "ReplicaSets" and "Deployments", which is the reason
// this function only lists these two types.
// TODO: update a similar info in federated hpa documentation.
func getRuntimeObjectForKind(c federationclientset.Interface, kind schema.GroupKind, ns, name string) (pkgruntime.Object, error) {
	switch kind {
	case extensionsinternal.Kind("ReplicaSet"):
		return c.ExtensionsV1beta1().ReplicaSets(ns).Get(name, metav1.GetOptions{})
	case extensionsinternal.Kind("Deployment"):
		return c.ExtensionsV1beta1().Deployments(ns).Get(name, metav1.GetOptions{})
	default:
		return nil, fmt.Errorf("Unsupported federated kind targeted by hpa: %v", kind)
	}
}

// updateRuntimeObjectForKind updates the hpa targetted object in the federation control plane.
// As of now, federation only supports "ReplicaSets" and "Deployments", which is the reason
// this function only lists these two types.
// TODO: update a similar info in federated hpa documentation.
func updateRuntimeObjectForKind(c federationclientset.Interface, kind schema.GroupKind, ns string, obj pkgruntime.Object) (pkgruntime.Object, error) {
	switch kind {
	case extensionsinternal.Kind("ReplicaSet"):
		return c.ExtensionsV1beta1().ReplicaSets(ns).Update(obj.(*v1beta1.ReplicaSet))
	case extensionsinternal.Kind("Deployment"):
		return c.ExtensionsV1beta1().Deployments(ns).Update(obj.(*v1beta1.Deployment))
	default:
		return nil, fmt.Errorf("Unsupported federated kind targeted by hpa: %v", kind)
	}
}

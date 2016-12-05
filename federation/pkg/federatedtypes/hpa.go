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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	pkgruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/watch"
	federationclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset"
	"k8s.io/kubernetes/federation/pkg/federation-controller/util"
	"k8s.io/kubernetes/pkg/api"
	apiv1 "k8s.io/kubernetes/pkg/api/v1"
	autoscalingv1 "k8s.io/kubernetes/pkg/apis/autoscaling/v1"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
)

const (
	HpaKind              = "horizontalpodautoscaler"
	HpaControllerName    = "horizontalpodautoscalers"
	scaleForbiddenWindow = 5 * time.Minute
)

func init() {
	RegisterFederatedType(HpaKind, HpaControllerName, []schema.GroupVersionResource{apiv1.SchemeGroupVersion.WithResource(HpaControllerName)}, NewHpaAdapter)
}

type HpaAdapter struct {
	client federationclientset.Interface
}

func NewHpaAdapter(client federationclientset.Interface) FederatedTypeAdapter {
	return &HpaAdapter{client: client}
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
		ObjectMeta: util.DeepCopyRelevantObjectMeta(hpa.ObjectMeta),
		Spec:       *(util.DeepCopyApiTypeOrPanic(hpa.Spec).(*autoscalingv1.HorizontalPodAutoscalerSpec)),
	}
}

func (a *HpaAdapter) Equivalent(obj1, obj2 pkgruntime.Object) bool {
	//hpa1 := obj1.(*autoscalingv1.HorizontalPodAutoscaler)
	//hpa2 := obj2.(*autoscalingv1.HorizontalPodAutoscaler)
	return util.ObjectMetaAndSpecEquivalent(obj1, obj2)
}

func (a *HpaAdapter) NamespacedName(obj pkgruntime.Object) types.NamespacedName {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return types.NamespacedName{Namespace: hpa.Namespace, Name: hpa.Name}
}

func (a *HpaAdapter) ObjectMeta(obj pkgruntime.Object) *metav1.ObjectMeta {
	return &obj.(*autoscalingv1.HorizontalPodAutoscaler).ObjectMeta
}

func (a *HpaAdapter) FedCreate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	secret := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(secret.Namespace).Create(secret)
}

func (a *HpaAdapter) FedDelete(namespacedName types.NamespacedName, options *metav1.DeleteOptions) error {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespacedName.Namespace).Delete(namespacedName.Name, options)
}

func (a *HpaAdapter) FedGet(namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *HpaAdapter) FedList(namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespace).List(options)
}

func (a *HpaAdapter) FedUpdate(obj pkgruntime.Object) (pkgruntime.Object, error) {
	secret := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(secret.Namespace).Update(secret)
}

func (a *HpaAdapter) FedWatch(namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return a.client.AutoscalingV1().HorizontalPodAutoscalers(namespace).Watch(options)
}

func (a *HpaAdapter) ClusterCreate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	secret := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return client.AutoscalingV1().HorizontalPodAutoscalers(secret.Namespace).Create(secret)
}

func (a *HpaAdapter) ClusterDelete(client kubeclientset.Interface, nsName types.NamespacedName, options *metav1.DeleteOptions) error {
	return client.AutoscalingV1().HorizontalPodAutoscalers(nsName.Namespace).Delete(nsName.Name, options)
}

func (a *HpaAdapter) ClusterGet(client kubeclientset.Interface, namespacedName types.NamespacedName) (pkgruntime.Object, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(namespacedName.Namespace).Get(namespacedName.Name, metav1.GetOptions{})
}

func (a *HpaAdapter) ClusterList(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (pkgruntime.Object, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(namespace).List(options)
}

func (a *HpaAdapter) ClusterUpdate(client kubeclientset.Interface, obj pkgruntime.Object) (pkgruntime.Object, error) {
	secret := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	return client.AutoscalingV1().HorizontalPodAutoscalers(secret.Namespace).Update(secret)
}

func (a *HpaAdapter) ClusterWatch(client kubeclientset.Interface, namespace string, options metav1.ListOptions) (watch.Interface, error) {
	return client.AutoscalingV1().HorizontalPodAutoscalers(namespace).Watch(options)
}

func (a *HpaAdapter) NewTestObject(namespace string) pkgruntime.Object {
	return &autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "test-hpa-",
			Namespace:    namespace,
		},
		Spec: autoscalingv1.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv1.CrossVersionObjectReference{
				Kind: "HorizontalPodAutoscaler",
				Name: "myhpa",
			},
			MaxReplicas: int32(10),
		},
	}
}

func (a *HpaAdapter) IsScheduler() bool {
	return true
}

func (a *HpaAdapter) Schedule(obj pkgruntime.Object, current map[string]pkgruntime.Object) (map[string]pkgruntime.Object, error) {
	fedHpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	var requestedMax int32 = fedHpa.Spec.MaxReplicas
	var requestedMin int32 = 1
	if fedHpa.Spec.MinReplicas != nil {
		requestedMin = *fedHpa.Spec.MinReplicas
	}

	// replica distribution count per cluster
	rdcMin := requestedMin / int32(len(current))
	if rdcMin < 1 {
		rdcMin = 1
	}
	// TODO: Is there a better way
	// We need to cap the lowest limit of Max to 2, because in a
	// situation like both min and max become 1 (same) for all clusters,
	// no rebalancing would happen.
	rdcMax := requestedMax / int32(len(current))
	if rdcMax < 2 {
		rdcMax = 2
	}

	type schedMinMax struct {
		min int32
		max int32
	}
	scheduled := make(map[string]*schedMinMax)
	// These hold the list of those clusters which can offer
	// min and max replicas, to those which want them.
	// For example new clusters joining the federation and/or
	// those clusters which need to increase or reduce replicas
	// beyond min/max limits.
	availableMaxList := sets.NewString()
	availableMinList := sets.NewString()
	noHpaList := sets.NewString()
	var existingMin int32 = 0
	var existingMax int32 = 0

	// First pass: Analyse existing local hpa's if any.
	// current has the list of all clusters, with obj as nil
	// for those clusters which do not have hpa yet.
	for cluster, obj := range current {
		if obj == nil {
			noHpaList.Insert(cluster)
			scheduled[cluster] = nil
			continue
		}
		if maxReplicasReducible(obj) {
			availableMaxList.Insert(cluster)
		}
		if minReplicasReducible(obj) {
			availableMinList.Insert(cluster)
		}

		scheduled[cluster] = &schedMinMax{min: 0, max: 0}
		if obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas != nil {
			existingMin += *obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas
			scheduled[cluster].min += *obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MinReplicas
		}
		existingMax += obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MaxReplicas
		scheduled[cluster].max += obj.(*autoscalingv1.HorizontalPodAutoscaler).Spec.MaxReplicas
	}

	var remainingMin int32 = requestedMin - existingMin
	var remainingMax int32 = requestedMax - existingMax

	// Second pass: reduction of replicas across clusters if needed.
	// In this pass, we remain pessimistic and reduce one replica per cluster at a time.

	// The requested fedHpa has requested/updated min max to
	// something lesser then current local hpas.
	if remainingMin < 0 {
		// first we try reducing from those clusters which already offer min
		if availableMinList.Len() > 0 {
			for _, cluster := range availableMinList.List() {
				minmax := scheduled[cluster]
				if (minmax.min - 1) != 0 {
					minmax.min--
					availableMinList.Delete(cluster)
					remainingMin++
					if remainingMin == 0 {
						break
					}
				}
			}
		}
	}
	// If we could not get needed replicas from already offered min above
	// we abruptly start removing replicas from some/all clusters.
	// Here we might make some min to 0 signalling that this hpa be
	// removed from this cluster altogether
	for remainingMin < 0 {
		for _, minmax := range scheduled {
			if minmax != nil &&
				!((minmax.min - 1) < 0) {
				minmax.min--
				remainingMin++
				if remainingMin == 0 {
					break
				}
			}
		}
	}

	if remainingMax < 0 {
		// first we try reducing from those clusters which already offer max
		if availableMaxList.Len() > 0 {
			for _, cluster := range availableMaxList.List() {
				minmax := scheduled[cluster]
				if minmax != nil && !((minmax.max - minmax.min) < 0) {
					minmax.max--
					availableMaxList.Delete(cluster)
					remainingMax++
					if remainingMax == 0 {
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
	for remainingMax < 0 {
		for _, minmax := range scheduled {
			if minmax != nil &&
				!((minmax.max - minmax.min) < 0) {
				minmax.max--
				remainingMax++
				if remainingMax == 0 {
					break
				}
			}
		}
	}

	// Third pass: We distribute replicas into those clusters which need them.
	toDistributeMin := remainingMin + int32(availableMinList.Len())
	toDistributeMax := remainingMax + int32(availableMaxList.Len())
	// Here we first distribute max and then (in the next loop)
	// distribute min into those clusters which already get the
	// max fixed.
	// In this process we might not meet the min limit and
	// total of min limits might remain more then the requested
	// federated min. This is partially because a min per cluster
	// cannot be lesser then 1.
	// Additionally we first increase replicas into those clusters
	// which already has hpa's and if we still have some remaining.
	// This will save cluster related resources for the user.
	// We then go ahead to give the replicas to those which do not
	// have any hpa. In this pass however we try to ensure that all
	// our Max are consumed in this reconcile.
	for cluster := range scheduled {
		if toDistributeMax == 0 {
			break
		}
		if scheduled[cluster] == nil {
			continue
		}
		if maxReplicasNeeded(current[cluster]) {
			scheduled[cluster].max++
			if availableMaxList.Len() > 0 {
				popped, notEmpty := availableMaxList.PopAny()
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

	// If we do not have any new clusters where we can just
	// give all our replicas, then we stick to increase and
	// decrease of 1 replica and continue doing the same in
	// next reconcile cycles.
	if noHpaList.Len() > 0 {
		for toDistributeMax > 0 {
			for _, cluster := range noHpaList.UnsortedList() {
				if scheduled[cluster] == nil {
					scheduled[cluster] = &schedMinMax{min: 0, max: 0}
				}
				// first give away max from clusters offering them
				// this case especially helps getting hpa into newly joining
				// clusters.
				if availableMaxList.Len() > 0 {
					popped, notEmpty := availableMaxList.PopAny()
					if notEmpty {
						// Boundary checks to reduce max have happened earlier in
						// minReplicasReducible().
						scheduled[cluster].max++
						scheduled[popped].max--
						toDistributeMax--
						continue
					}
				}
				if toDistributeMax < rdcMax {
					scheduled[cluster].max += toDistributeMax
					toDistributeMax = 0
					break
				}
				scheduled[cluster].max += rdcMax
				toDistributeMax -= rdcMax
			}
		}
	} else { // we have no new clusters but if still have max replicas to distribute;
		// just distribute all in current clusters.
		for toDistributeMax > 0 {
			for cluster, minmax := range scheduled {
				if minmax == nil {
					minmax = &schedMinMax{min: 0, max: 0}
					scheduled[cluster] = minmax
				}
				// First give away max from clusters offering them.
				// This case especially helps getting hpa into newly joining
				// clusters.
				if availableMaxList.Len() > 0 {
					popped, notEmpty := availableMaxList.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier in
						// minReplicasReducible().
						minmax.max++
						scheduled[popped].max--
						toDistributeMax--
						continue
					}
				}
				if toDistributeMax < rdcMax {
					minmax.max += toDistributeMax
					toDistributeMax = 0
					break
				}
				minmax.max += rdcMax
				toDistributeMax -= rdcMax
			}
		}
	}

	// We distribute min to those clusters which:
	// 1 - can adjust min (our increase step would be only 1)
	// 2 - which do not have this hpa (increase step rdcMin)
	// Also after this distribution, we might still have some
	// minReplicas to distribute, but we ignore that in this reconcile.
	// On the other hand we might exhaust all min replicas here, with
	// some clusters still needing them. We adjust this in next step by
	// assigning min replicas to 1 into those clusters which got max
	// but min remains 0.
	for cluster := range scheduled {
		if toDistributeMin == 0 {
			break
		}
		// We have distriubted Max and thus scheduled might not be nil
		// but probably current (what we got originally) is nil(no hpa)
		if scheduled[cluster] == nil || current[cluster] == nil {
			continue
		}
		if minReplicasAdjusteable(current[cluster]) {
			if availableMinList.Len() > 0 {
				popped, notEmpty := availableMinList.PopAny()
				if notEmpty {
					// Boundary checks have happened earlier.
					scheduled[popped].min--
					scheduled[cluster].min++
					toDistributeMin--
				}
			}
		}
	}

	if noHpaList.Len() > 0 {
		// TODO: can this become an infinite loop?
		for toDistributeMin > 0 {
			for _, cluster := range noHpaList.UnsortedList() {
				minmax := scheduled[cluster]
				if minmax == nil {
					// We did not get max here so this cluster
					// remains without hpa
					continue
				}
				var replicaNum int32 = 0
				if toDistributeMin < rdcMin {
					replicaNum = toDistributeMin
				} else {
					replicaNum = rdcMin
				}
				if (minmax.max - replicaNum) < minmax.min {
					// Cannot increase the min in this cluster
					// as it will go beyond max
					continue
				}
				if availableMinList.Len() > 0 {
					popped, notEmpty := availableMinList.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier.
						scheduled[popped].min--
						minmax.min++
						toDistributeMin--
						continue
					}
				}
				minmax.min += replicaNum
				toDistributeMin -= replicaNum
			}
		}
	} else { // we have no new clusters but if still have min replicas to distribute;
		// just distribute all in current clusters.
		for toDistributeMin > 0 {
			for _, minmax := range scheduled {
				if minmax == nil {
					// We did not get max here so this cluster
					// remains without hpa
					continue
				}
				var replicaNum int32 = 0
				if toDistributeMin < rdcMin {
					replicaNum = toDistributeMin
				} else {
					replicaNum = rdcMin
				}
				if (minmax.max - replicaNum) < minmax.min {
					// Cannot increase the min in this cluster
					// as it will go beyond max
					continue
				}
				if availableMinList.Len() > 0 {
					popped, notEmpty := availableMinList.PopAny()
					if notEmpty {
						// Boundary checks have happened earlier.
						scheduled[popped].min--
						minmax.min++
						toDistributeMin--
						continue
					}
				}
				minmax.min += replicaNum
				toDistributeMin -= replicaNum
			}
		}
	}

	schedHpas := make(map[string]pkgruntime.Object)

	// TODO: Optimisation: use schedule[string]pkgruntime.object directly
	// in above code/computation to avoid this additional computation/copy.
	// But we would any way need to scan the list again for updating
	// min of those scheduled hpas where min <= 0 and max has some proper value.
	for cluster, minmax := range scheduled {
		currentClusterHpa := current[cluster]
		if minmax == nil {
			schedHpas[cluster] = nil
			continue
		}
		if (minmax.min <= 0) && (minmax.max > 0) {
			// Min total does not necessarily meet the federated min limit
			minmax.min = 1
		}
		if minmax.max <= 0 {
			// This ideally is a case were we should remove HPA from this cluser
			// but we dont, and follow the deletion helpers method.
			minmax.max = 0
			minmax.min = 0
		}
		if current[cluster] == nil {
			// This cluster got a new hpa
			schedHpas[cluster] = getNewLocalHpa(fedHpa, minmax.min, minmax.max)
		} else {
			// we copy out the existing hpa from local cluster
			// and update the max and min if we need to, retaining the status.
			var lHpa *autoscalingv1.HorizontalPodAutoscaler
			lObj, err := api.Scheme.DeepCopy(currentClusterHpa)
			if err != nil {
				return nil, fmt.Errorf("Error copying the hpa object from cluster %s: %v", cluster, err)
			} else {
				lHpa = lObj.(*autoscalingv1.HorizontalPodAutoscaler)
			}
			if lHpa.Spec.MinReplicas == nil {
				lHpa.Spec.MinReplicas = new(int32)
			}
			*lHpa.Spec.MinReplicas = minmax.min
			lHpa.Spec.MaxReplicas = minmax.max
			schedHpas[cluster] = lHpa
		}
	}

	return schedHpas, nil
}

func (a *HpaAdapter) FedUpdateStatus(obj pkgruntime.Object, current map[string]pkgruntime.Object) error {
	copiedObj, err := api.Scheme.DeepCopy(obj)
	if err != nil {
		return fmt.Errorf("Error copying federated hpa object: %v", err)
	}
	if len(current) == 0 {
		return nil
	}

	fedHpa := copiedObj.(*autoscalingv1.HorizontalPodAutoscaler)
	gen := fedHpa.Generation
	updatedStatus := autoscalingv1.HorizontalPodAutoscalerStatus{ObservedGeneration: &gen}
	if fedHpa.Status.LastScaleTime != nil {
		// TODO: do we really need to copy value
		t := metav1.NewTime(fedHpa.Status.LastScaleTime.Time)
		updatedStatus.LastScaleTime = &t
	}
	var ccup int32 = 0
	updatedStatus.CurrentCPUUtilizationPercentage = &ccup

	count := int32(0)
	for _, lObj := range current {
		if lObj == nil {
			continue
		}
		lHpa := lObj.(*autoscalingv1.HorizontalPodAutoscaler)
		if lHpa.Status.CurrentCPUUtilizationPercentage != nil {
			*updatedStatus.CurrentCPUUtilizationPercentage += *lHpa.Status.CurrentCPUUtilizationPercentage
			count++
		}
		if updatedStatus.LastScaleTime != nil && lHpa.Status.LastScaleTime != nil &&
			updatedStatus.LastScaleTime.After(lHpa.Status.LastScaleTime.Time) {
			t := metav1.NewTime(lHpa.Status.LastScaleTime.Time)
			updatedStatus.LastScaleTime = &t
		}
		updatedStatus.CurrentReplicas += lHpa.Status.CurrentReplicas
		updatedStatus.DesiredReplicas += lHpa.Status.DesiredReplicas
	}

	// Average out the available current utilisation
	if *updatedStatus.CurrentCPUUtilizationPercentage != 0 {
		*updatedStatus.CurrentCPUUtilizationPercentage /= count
	}

	if ((fedHpa.Status.CurrentCPUUtilizationPercentage == nil &&
		*updatedStatus.CurrentCPUUtilizationPercentage != 0) ||
		(fedHpa.Status.CurrentCPUUtilizationPercentage != nil &&
			*updatedStatus.CurrentCPUUtilizationPercentage !=
				*fedHpa.Status.CurrentCPUUtilizationPercentage)) ||
		((fedHpa.Status.LastScaleTime == nil && updatedStatus.LastScaleTime != nil) ||
			(fedHpa.Status.LastScaleTime != nil && updatedStatus.LastScaleTime == nil) ||
			((fedHpa.Status.LastScaleTime != nil && updatedStatus.LastScaleTime != nil) &&
				updatedStatus.LastScaleTime.After(fedHpa.Status.LastScaleTime.Time))) ||
		fedHpa.Status.DesiredReplicas != updatedStatus.DesiredReplicas ||
		fedHpa.Status.CurrentReplicas != updatedStatus.CurrentReplicas {

		fedHpa.Status = updatedStatus
		_, err = a.client.AutoscalingV1().HorizontalPodAutoscalers(fedHpa.Namespace).UpdateStatus(fedHpa)
		if err != nil {
			return fmt.Errorf("Error updating hpa: %s status in federation: %v", fedHpa.Name, err)
		}
	}
	return nil
}

func getNewLocalHpa(fedHpa *autoscalingv1.HorizontalPodAutoscaler, min, max int32) *autoscalingv1.HorizontalPodAutoscaler {
	newHpa := autoscalingv1.HorizontalPodAutoscaler{
		ObjectMeta: util.DeepCopyRelevantObjectMeta(fedHpa.ObjectMeta),
		Spec:       util.DeepCopyApiTypeOrPanic(fedHpa.Spec).(autoscalingv1.HorizontalPodAutoscalerSpec),
	}
	if newHpa.Spec.MinReplicas == nil {
		newHpa.Spec.MinReplicas = new(int32)
	}
	*newHpa.Spec.MinReplicas = min
	newHpa.Spec.MaxReplicas = max

	return &newHpa
}

// isPristine is used to determine if so far local controller has been
// able to really determine, what should be the desired replica number for
// this cluster.
// This is used only to get hpas into those clusters which might join fresh,
// and so far other cluster hpas haven't really reached anywhere.
func isPristine(hpa *autoscalingv1.HorizontalPodAutoscaler) bool {
	if hpa.Status.LastScaleTime == nil &&
		hpa.Status.DesiredReplicas == 0 {
		return true
	}
	return false
}

// isScaleable tells if it already has been a reasonable amount of
// time since this hpa scaled. Its used to avoid fast thrashing.
func isScaleable(hpa *autoscalingv1.HorizontalPodAutoscaler) bool {
	if hpa.Status.LastScaleTime == nil {
		return false
	}
	t := hpa.Status.LastScaleTime.Add(scaleForbiddenWindow)
	if t.After(time.Now()) {
		return false
	}
	return true
}

func maxReplicasReducible(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if (hpa.Spec.MinReplicas != nil) &&
		((hpa.Spec.MaxReplicas - *hpa.Spec.MinReplicas) < 0) {
		return false
	}
	if isPristine(hpa) {
		return true
	}
	if !isScaleable(hpa) {
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
func minReplicasReducible(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if isPristine(hpa) && (hpa.Spec.MinReplicas != nil) &&
		(*hpa.Spec.MinReplicas > 1) &&
		(*hpa.Spec.MinReplicas == hpa.Spec.MaxReplicas) {
		return true
	}
	if !isScaleable(hpa) {
		return false
	}
	if (hpa.Spec.MinReplicas != nil) &&
		(*hpa.Spec.MinReplicas > 1) &&
		(*hpa.Spec.MinReplicas == hpa.Spec.MaxReplicas) {
		return true
	}
	// TODO: should we rather just check that max replica equals min and this check is useless
	if (hpa.Spec.MinReplicas != nil) &&
		(*hpa.Spec.MinReplicas > 1) &&
		(hpa.Status.DesiredReplicas == hpa.Status.CurrentReplicas) &&
		(hpa.Status.CurrentReplicas == *hpa.Spec.MinReplicas) {
		return true
	}
	return false
}

func maxReplicasNeeded(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if !isScaleable(hpa) {
		return false
	}

	if (hpa.Status.CurrentReplicas == hpa.Status.DesiredReplicas) &&
		(hpa.Status.CurrentReplicas == hpa.Spec.MaxReplicas) {
		return true
	}
	return false
}

func minReplicasAdjusteable(obj pkgruntime.Object) bool {
	hpa := obj.(*autoscalingv1.HorizontalPodAutoscaler)
	if !isScaleable(hpa) ||
		((hpa.Spec.MinReplicas != nil) &&
			(*hpa.Spec.MinReplicas+1) > hpa.Spec.MaxReplicas) {
		return false
	}

	if (hpa.Spec.MinReplicas != nil) &&
		(hpa.Status.DesiredReplicas > *hpa.Spec.MinReplicas) {
		return true
	}
	return false
}

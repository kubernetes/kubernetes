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

package statefulset

import (
	"context"
	"sort"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller/history"
	"k8s.io/kubernetes/pkg/features"
)

// StatefulSetControl implements the control logic for updating StatefulSets and their children Pods. It is implemented
// as an interface to allow for extensions that provide different semantics. Currently, there is only one implementation.
type StatefulSetControlInterface interface {
	// UpdateStatefulSet implements the control logic for Pod creation, update, and deletion, and
	// persistent volume creation, update, and deletion.
	// If an implementation returns a non-nil error, the invocation will be retried using a rate-limited strategy.
	// Implementors should sink any errors that they do not wish to trigger a retry, and they may feel free to
	// exit exceptionally at any point provided they wish the update to be re-run at a later point in time.
	UpdateStatefulSet(ctx context.Context, set *apps.StatefulSet, pods []*v1.Pod) (*apps.StatefulSetStatus, error)
	// ListRevisions returns a array of the ControllerRevisions that represent the revisions of set. If the returned
	// error is nil, the returns slice of ControllerRevisions is valid.
	ListRevisions(set *apps.StatefulSet) ([]*apps.ControllerRevision, error)
	// AdoptOrphanRevisions adopts any orphaned ControllerRevisions that match set's Selector. If all adoptions are
	// successful the returned error is nil.
	AdoptOrphanRevisions(set *apps.StatefulSet, revisions []*apps.ControllerRevision) error
}

// NewDefaultStatefulSetControl returns a new instance of the default implementation StatefulSetControlInterface that
// implements the documented semantics for StatefulSets. podControl is the PodControlInterface used to create, update,
// and delete Pods and to create PersistentVolumeClaims. statusUpdater is the StatefulSetStatusUpdaterInterface used
// to update the status of StatefulSets. You should use an instance returned from NewRealStatefulPodControl() for any
// scenario other than testing.
func NewDefaultStatefulSetControl(
	podControl *StatefulPodControl,
	statusUpdater StatefulSetStatusUpdaterInterface,
	controllerHistory history.Interface,
	recorder record.EventRecorder) StatefulSetControlInterface {
	return &defaultStatefulSetControl{podControl, statusUpdater, controllerHistory, recorder}
}

type defaultStatefulSetControl struct {
	podControl        *StatefulPodControl
	statusUpdater     StatefulSetStatusUpdaterInterface
	controllerHistory history.Interface
	recorder          record.EventRecorder
}

// UpdateStatefulSet executes the core logic loop for a stateful set, applying the predictable and
// consistent monotonic update strategy by default - scale up proceeds in ordinal order, no new pod
// is created while any pod is unhealthy, and pods are terminated in descending order. The burst
// strategy allows these constraints to be relaxed - pods will be created and deleted eagerly and
// in no particular order. Clients using the burst strategy should be careful to ensure they
// understand the consistency implications of having unpredictable numbers of pods available.
func (ssc *defaultStatefulSetControl) UpdateStatefulSet(ctx context.Context, set *apps.StatefulSet, pods []*v1.Pod) (*apps.StatefulSetStatus, error) {
	set = set.DeepCopy() // set is modified when a new revision is created in performUpdate. Make a copy now to avoid mutation errors.

	// list all revisions and sort them
	revisions, err := ssc.ListRevisions(set)
	if err != nil {
		return nil, err
	}
	history.SortControllerRevisions(revisions)

	currentRevision, updateRevision, status, err := ssc.performUpdate(ctx, set, pods, revisions)
	if err != nil {
		return nil, utilerrors.NewAggregate([]error{err, ssc.truncateHistory(set, pods, revisions, currentRevision, updateRevision)})
	}

	// maintain the set's revision history limit
	return status, ssc.truncateHistory(set, pods, revisions, currentRevision, updateRevision)
}

func (ssc *defaultStatefulSetControl) performUpdate(
	ctx context.Context, set *apps.StatefulSet, pods []*v1.Pod, revisions []*apps.ControllerRevision) (*apps.ControllerRevision, *apps.ControllerRevision, *apps.StatefulSetStatus, error) {
	var currentStatus *apps.StatefulSetStatus
	// get the current, and update revisions
	currentRevision, updateRevision, collisionCount, err := ssc.getStatefulSetRevisions(set, revisions)
	if err != nil {
		return currentRevision, updateRevision, currentStatus, err
	}

	// perform the main update function and get the status
	currentStatus, err = ssc.updateStatefulSet(ctx, set, currentRevision, updateRevision, collisionCount, pods)
	if err != nil {
		return currentRevision, updateRevision, currentStatus, err
	}

	klog.V(4).InfoS("StatefulSet revisions", "statefulSet", klog.KObj(set),
		"currentRevision", currentStatus.CurrentRevision,
		"updateRevision", currentStatus.UpdateRevision)

	return currentRevision, updateRevision, currentStatus, nil
}

func (ssc *defaultStatefulSetControl) ListRevisions(set *apps.StatefulSet) ([]*apps.ControllerRevision, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return ssc.controllerHistory.ListControllerRevisions(set, selector)
}

func (ssc *defaultStatefulSetControl) AdoptOrphanRevisions(
	set *apps.StatefulSet,
	revisions []*apps.ControllerRevision) error {
	for i := range revisions {
		adopted, err := ssc.controllerHistory.AdoptControllerRevision(set, controllerKind, revisions[i])
		if err != nil {
			return err
		}
		revisions[i] = adopted
	}
	return nil
}

// truncateHistory truncates any non-live ControllerRevisions in revisions from set's history. The UpdateRevision and
// CurrentRevision in set's Status are considered to be live. Any revisions associated with the Pods in pods are also
// considered to be live. Non-live revisions are deleted, starting with the revision with the lowest Revision, until
// only RevisionHistoryLimit revisions remain. If the returned error is nil the operation was successful. This method
// expects that revisions is sorted when supplied.
func (ssc *defaultStatefulSetControl) truncateHistory(
	set *apps.StatefulSet,
	pods []*v1.Pod,
	revisions []*apps.ControllerRevision,
	current *apps.ControllerRevision,
	update *apps.ControllerRevision) error {
	history := make([]*apps.ControllerRevision, 0, len(revisions))
	// mark all live revisions
	live := map[string]bool{}
	if current != nil {
		live[current.Name] = true
	}
	if update != nil {
		live[update.Name] = true
	}
	for i := range pods {
		live[getPodRevision(pods[i])] = true
	}
	// collect live revisions and historic revisions
	for i := range revisions {
		if !live[revisions[i].Name] {
			history = append(history, revisions[i])
		}
	}
	historyLen := len(history)
	historyLimit := int(*set.Spec.RevisionHistoryLimit)
	if historyLen <= historyLimit {
		return nil
	}
	// delete any non-live history to maintain the revision limit.
	history = history[:(historyLen - historyLimit)]
	for i := 0; i < len(history); i++ {
		if err := ssc.controllerHistory.DeleteControllerRevision(history[i]); err != nil {
			return err
		}
	}
	return nil
}

// getStatefulSetRevisions returns the current and update ControllerRevisions for set. It also
// returns a collision count that records the number of name collisions set saw when creating
// new ControllerRevisions. This count is incremented on every name collision and is used in
// building the ControllerRevision names for name collision avoidance. This method may create
// a new revision, or modify the Revision of an existing revision if an update to set is detected.
// This method expects that revisions is sorted when supplied.
func (ssc *defaultStatefulSetControl) getStatefulSetRevisions(
	set *apps.StatefulSet,
	revisions []*apps.ControllerRevision) (*apps.ControllerRevision, *apps.ControllerRevision, int32, error) {
	var currentRevision, updateRevision *apps.ControllerRevision

	revisionCount := len(revisions)
	history.SortControllerRevisions(revisions)

	// Use a local copy of set.Status.CollisionCount to avoid modifying set.Status directly.
	// This copy is returned so the value gets carried over to set.Status in updateStatefulSet.
	var collisionCount int32
	if set.Status.CollisionCount != nil {
		collisionCount = *set.Status.CollisionCount
	}

	// create a new revision from the current set
	updateRevision, err := newRevision(set, nextRevision(revisions), &collisionCount)
	if err != nil {
		return nil, nil, collisionCount, err
	}

	// find any equivalent revisions
	equalRevisions := history.FindEqualRevisions(revisions, updateRevision)
	equalCount := len(equalRevisions)

	if equalCount > 0 && history.EqualRevision(revisions[revisionCount-1], equalRevisions[equalCount-1]) {
		// if the equivalent revision is immediately prior the update revision has not changed
		updateRevision = revisions[revisionCount-1]
	} else if equalCount > 0 {
		// if the equivalent revision is not immediately prior we will roll back by incrementing the
		// Revision of the equivalent revision
		updateRevision, err = ssc.controllerHistory.UpdateControllerRevision(
			equalRevisions[equalCount-1],
			updateRevision.Revision)
		if err != nil {
			return nil, nil, collisionCount, err
		}
	} else {
		//if there is no equivalent revision we create a new one
		updateRevision, err = ssc.controllerHistory.CreateControllerRevision(set, updateRevision, &collisionCount)
		if err != nil {
			return nil, nil, collisionCount, err
		}
	}

	// attempt to find the revision that corresponds to the current revision
	for i := range revisions {
		if revisions[i].Name == set.Status.CurrentRevision {
			currentRevision = revisions[i]
			break
		}
	}

	// if the current revision is nil we initialize the history by setting it to the update revision
	if currentRevision == nil {
		currentRevision = updateRevision
	}

	return currentRevision, updateRevision, collisionCount, nil
}

// updateStatefulSet performs the update function for a StatefulSet. This method creates, updates, and deletes Pods in
// the set in order to conform the system to the target state for the set. The target state always contains
// set.Spec.Replicas Pods with a Ready Condition. If the UpdateStrategy.Type for the set is
// RollingUpdateStatefulSetStrategyType then all Pods in the set must be at set.Status.CurrentRevision.
// If the UpdateStrategy.Type for the set is OnDeleteStatefulSetStrategyType, the target state implies nothing about
// the revisions of Pods in the set. If the UpdateStrategy.Type for the set is PartitionStatefulSetStrategyType, then
// all Pods with ordinal less than UpdateStrategy.Partition.Ordinal must be at Status.CurrentRevision and all other
// Pods must be at Status.UpdateRevision. If the returned error is nil, the returned StatefulSetStatus is valid and the
// update must be recorded. If the error is not nil, the method should be retried until successful.
func (ssc *defaultStatefulSetControl) updateStatefulSet(
	ctx context.Context,
	set *apps.StatefulSet,
	currentRevision *apps.ControllerRevision,
	updateRevision *apps.ControllerRevision,
	collisionCount int32,
	pods []*v1.Pod) (statefulSetStatus *apps.StatefulSetStatus, updateErr error) {
	// get the current and update revisions of the set.
	currentSet, err := ApplyRevision(set, currentRevision)
	if err != nil {
		return nil, err
	}
	updateSet, err := ApplyRevision(set, updateRevision)
	if err != nil {
		return nil, err
	}

	// set the generation, and revisions in the returned status
	status := apps.StatefulSetStatus{}
	status.ObservedGeneration = set.Generation
	status.CurrentRevision = currentRevision.Name
	status.UpdateRevision = updateRevision.Name
	status.CollisionCount = new(int32)
	*status.CollisionCount = collisionCount

	replicaCount := int(*set.Spec.Replicas)
	// slice that will contain all Pods such that 0 <= getOrdinal(pod) < set.Spec.Replicas
	replicas := make([]*v1.Pod, replicaCount)
	// slice that will contain all Pods such that set.Spec.Replicas <= getOrdinal(pod)
	condemned := make([]*v1.Pod, 0, len(pods))
	unhealthy := 0
	var firstUnhealthyPod *v1.Pod

	// First we partition pods into two lists valid replicas and condemned Pods
	for i := range pods {
		status.Replicas++

		// count the number of running and ready replicas
		if isRunningAndReady(pods[i]) {
			status.ReadyReplicas++
			// count the number of running and available replicas
			if isRunningAndAvailable(pods[i], set.Spec.MinReadySeconds) {
				status.AvailableReplicas++
			}

		}

		// count the number of current and update replicas
		if isCreated(pods[i]) && !isTerminating(pods[i]) {
			if getPodRevision(pods[i]) == currentRevision.Name {
				status.CurrentReplicas++
			}
			if getPodRevision(pods[i]) == updateRevision.Name {
				status.UpdatedReplicas++
			}
		}

		if ord := getOrdinal(pods[i]); 0 <= ord && ord < replicaCount {
			// if the ordinal of the pod is within the range of the current number of replicas,
			// insert it at the indirection of its ordinal
			replicas[ord] = pods[i]
		} else if ord >= replicaCount {
			// if the ordinal is greater than the number of replicas add it to the condemned list
			condemned = append(condemned, pods[i])
		}
		// If the ordinal could not be parsed (ord < 0), ignore the Pod.
	}

	// make sure to update the latest status even if there is an error later
	defer func() {
		// update the set's status
		statusErr := ssc.updateStatefulSetStatus(ctx, set, &status)
		if statusErr == nil {
			klog.V(4).InfoS("Updated status", "statefulSet", klog.KObj(set),
				"replicas", status.Replicas,
				"readyReplicas", status.ReadyReplicas,
				"currentReplicas", status.CurrentReplicas,
				"updatedReplicas", status.UpdatedReplicas)
		} else if updateErr == nil {
			updateErr = statusErr
		} else {
			klog.V(4).InfoS("Could not update status", "statefulSet", klog.KObj(set), "err", statusErr)
		}
	}()

	// for any empty indices in the sequence [0,set.Spec.Replicas) create a new Pod at the correct revision
	for ord := 0; ord < replicaCount; ord++ {
		if replicas[ord] == nil {
			replicas[ord] = newVersionedStatefulSetPod(
				currentSet,
				updateSet,
				currentRevision.Name,
				updateRevision.Name, ord)
		}
	}

	// sort the condemned Pods by their ordinals
	sort.Sort(ascendingOrdinal(condemned))

	// find the first unhealthy Pod
	for i := range replicas {
		if !isHealthy(replicas[i]) {
			unhealthy++
			if firstUnhealthyPod == nil {
				firstUnhealthyPod = replicas[i]
			}
		}
	}

	for i := range condemned {
		if !isHealthy(condemned[i]) {
			unhealthy++
			if firstUnhealthyPod == nil {
				firstUnhealthyPod = condemned[i]
			}
		}
	}

	if unhealthy > 0 {
		klog.V(4).InfoS("StatefulSet has unhealthy Pods", "statefulSet", klog.KObj(set), "unhealthyReplicas", unhealthy, "pod", klog.KObj(firstUnhealthyPod))
	}

	// If the StatefulSet is being deleted, don't do anything other than updating
	// status.
	if set.DeletionTimestamp != nil {
		return &status, nil
	}

	monotonic := !allowsBurst(set)

	// Examine each replica with respect to its ordinal
	for i := range replicas {
		// delete and recreate failed pods
		if isFailed(replicas[i]) {
			ssc.recorder.Eventf(set, v1.EventTypeWarning, "RecreatingFailedPod",
				"StatefulSet %s/%s is recreating failed Pod %s",
				set.Namespace,
				set.Name,
				replicas[i].Name)
			if err := ssc.podControl.DeleteStatefulPod(set, replicas[i]); err != nil {
				return &status, err
			}
			if getPodRevision(replicas[i]) == currentRevision.Name {
				status.CurrentReplicas--
			}
			if getPodRevision(replicas[i]) == updateRevision.Name {
				status.UpdatedReplicas--
			}
			status.Replicas--
			replicas[i] = newVersionedStatefulSetPod(
				currentSet,
				updateSet,
				currentRevision.Name,
				updateRevision.Name,
				i)
		}
		// If we find a Pod that has not been created we create the Pod
		if !isCreated(replicas[i]) {
			if utilfeature.DefaultFeatureGate.Enabled(features.StatefulSetAutoDeletePVC) {
				if isStale, err := ssc.podControl.PodClaimIsStale(set, replicas[i]); err != nil {
					return &status, err
				} else if isStale {
					// If a pod has a stale PVC, no more work can be done this round.
					return &status, err
				}
			}
			if err := ssc.podControl.CreateStatefulPod(ctx, set, replicas[i]); err != nil {
				return &status, err
			}
			status.Replicas++
			if getPodRevision(replicas[i]) == currentRevision.Name {
				status.CurrentReplicas++
			}
			if getPodRevision(replicas[i]) == updateRevision.Name {
				status.UpdatedReplicas++
			}
			// if the set does not allow bursting, return immediately
			if monotonic {
				return &status, nil
			}
			// pod created, no more work possible for this round
			continue
		}
		// If we find a Pod that is currently terminating, we must wait until graceful deletion
		// completes before we continue to make progress.
		if isTerminating(replicas[i]) && monotonic {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to Terminate",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(replicas[i]))
			return &status, nil
		}
		// If we have a Pod that has been created but is not running and ready we can not make progress.
		// We must ensure that all for each Pod, when we create it, all of its predecessors, with respect to its
		// ordinal, are Running and Ready.
		if !isRunningAndReady(replicas[i]) && monotonic {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to be Running and Ready",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(replicas[i]))
			return &status, nil
		}
		// If we have a Pod that has been created but is not available we can not make progress.
		// We must ensure that all for each Pod, when we create it, all of its predecessors, with respect to its
		// ordinal, are Available.
		if !isRunningAndAvailable(replicas[i], set.Spec.MinReadySeconds) && monotonic {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to be Available",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(replicas[i]))
			return &status, nil
		}
		// Enforce the StatefulSet invariants
		retentionMatch := true
		if utilfeature.DefaultFeatureGate.Enabled(features.StatefulSetAutoDeletePVC) {
			var err error
			retentionMatch, err = ssc.podControl.ClaimsMatchRetentionPolicy(updateSet, replicas[i])
			// An error is expected if the pod is not yet fully updated, and so return is treated as matching.
			if err != nil {
				retentionMatch = true
			}
		}
		if identityMatches(set, replicas[i]) && storageMatches(set, replicas[i]) && retentionMatch {
			continue
		}
		// Make a deep copy so we don't mutate the shared cache
		replica := replicas[i].DeepCopy()
		if err := ssc.podControl.UpdateStatefulPod(updateSet, replica); err != nil {
			return &status, err
		}
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.StatefulSetAutoDeletePVC) {
		// Ensure ownerRefs are set correctly for the condemned pods.
		for i := range condemned {
			if matchPolicy, err := ssc.podControl.ClaimsMatchRetentionPolicy(updateSet, condemned[i]); err != nil {
				return &status, err
			} else if !matchPolicy {
				if err := ssc.podControl.UpdatePodClaimForRetentionPolicy(updateSet, condemned[i]); err != nil {
					return &status, err
				}
			}
		}
	}

	// At this point, all of the current Replicas are Running, Ready and Available, we can consider termination.
	// We will wait for all predecessors to be Running and Ready prior to attempting a deletion.
	// We will terminate Pods in a monotonically decreasing order over [len(pods),set.Spec.Replicas).
	// Note that we do not resurrect Pods in this interval. Also note that scaling will take precedence over
	// updates.
	for target := len(condemned) - 1; target >= 0; target-- {
		// wait for terminating pods to expire
		if isTerminating(condemned[target]) {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to Terminate prior to scale down",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(condemned[target]))
			// block if we are in monotonic mode
			if monotonic {
				return &status, nil
			}
			continue
		}
		// if we are in monotonic mode and the condemned target is not the first unhealthy Pod block
		if !isRunningAndReady(condemned[target]) && monotonic && condemned[target] != firstUnhealthyPod {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to be Running and Ready prior to scale down",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(firstUnhealthyPod))
			return &status, nil
		}
		// if we are in monotonic mode and the condemned target is not the first unhealthy Pod, block.
		if !isRunningAndAvailable(condemned[target], set.Spec.MinReadySeconds) && monotonic && condemned[target] != firstUnhealthyPod {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to be Available prior to scale down",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(firstUnhealthyPod))
			return &status, nil
		}
		klog.V(2).InfoS("Pod of StatefulSet is terminating for scale down",
			"statefulSet", klog.KObj(set), "pod", klog.KObj(condemned[target]))

		if err := ssc.podControl.DeleteStatefulPod(set, condemned[target]); err != nil {
			return &status, err
		}
		if getPodRevision(condemned[target]) == currentRevision.Name {
			status.CurrentReplicas--
		}
		if getPodRevision(condemned[target]) == updateRevision.Name {
			status.UpdatedReplicas--
		}
		if monotonic {
			return &status, nil
		}
	}

	// for the OnDelete strategy we short circuit. Pods will be updated when they are manually deleted.
	if set.Spec.UpdateStrategy.Type == apps.OnDeleteStatefulSetStrategyType {
		return &status, nil
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.MaxUnavailableStatefulSet) {
		return updateStatefulSetAfterInvariantEstablished(ctx,
			ssc,
			set,
			replicas,
			updateRevision,
			status,
		)
	}

	// we compute the minimum ordinal of the target sequence for a destructive update based on the strategy.
	updateMin := 0
	if set.Spec.UpdateStrategy.RollingUpdate != nil {
		updateMin = int(*set.Spec.UpdateStrategy.RollingUpdate.Partition)
	}
	// we terminate the Pod with the largest ordinal that does not match the update revision.
	for target := len(replicas) - 1; target >= updateMin; target-- {

		// delete the Pod if it is not already terminating and does not match the update revision.
		if getPodRevision(replicas[target]) != updateRevision.Name && !isTerminating(replicas[target]) {
			klog.V(2).InfoS("Pod of StatefulSet is terminating for update",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(replicas[target]))
			err := ssc.podControl.DeleteStatefulPod(set, replicas[target])
			status.CurrentReplicas--
			return &status, err
		}

		// wait for unhealthy Pods on update
		if !isHealthy(replicas[target]) {
			klog.V(4).InfoS("StatefulSet is waiting for Pod to update",
				"statefulSet", klog.KObj(set), "pod", klog.KObj(replicas[target]))
			return &status, nil
		}

	}
	return &status, nil
}

func updateStatefulSetAfterInvariantEstablished(
	ctx context.Context,
	ssc *defaultStatefulSetControl,
	set *apps.StatefulSet,
	replicas []*v1.Pod,
	updateRevision *apps.ControllerRevision,
	status apps.StatefulSetStatus,
) (*apps.StatefulSetStatus, error) {

	replicaCount := int(*set.Spec.Replicas)

	// we compute the minimum ordinal of the target sequence for a destructive update based on the strategy.
	updateMin := 0
	maxUnavailable := 1
	if set.Spec.UpdateStrategy.RollingUpdate != nil {
		updateMin = int(*set.Spec.UpdateStrategy.RollingUpdate.Partition)

		// if the feature was enabled and then later disabled, MaxUnavailable may have a value
		// more than 1. Ignore the passed in value and Use maxUnavailable as 1 to enforce
		// expected behavior when feature gate is not enabled.
		var err error
		maxUnavailable, err = getStatefulSetMaxUnavailable(set.Spec.UpdateStrategy.RollingUpdate.MaxUnavailable, replicaCount)
		if err != nil {
			return &status, err
		}
	}

	// Collect all targets in the range between the 0 and Spec.Replicas. Count any targets in that range
	// that are unhealthy i.e. terminated or not running and ready as unavailable). Select the
	// (MaxUnavailable - Unavailable) Pods, in order with respect to their ordinal for termination. Delete
	// those pods and count the successful deletions. Update the status with the correct number of deletions.
	unavailablePods := 0
	for target := len(replicas) - 1; target >= 0; target-- {
		if !isHealthy(replicas[target]) {
			unavailablePods++
		}
	}

	if unavailablePods >= maxUnavailable {
		klog.V(2).Infof("StatefulSet %s/%s found unavailablePods %v, more than or equal to allowed maxUnavailable %v",
			set.Namespace,
			set.Name,
			unavailablePods,
			maxUnavailable)
		return &status, nil
	}

	// Now we need to delete MaxUnavailable- unavailablePods
	// start deleting one by one starting from the highest ordinal first
	podsToDelete := maxUnavailable - unavailablePods

	deletedPods := 0
	for target := len(replicas) - 1; target >= updateMin && deletedPods < podsToDelete; target-- {

		// delete the Pod if it is healthy and the revision doesnt match the target
		if getPodRevision(replicas[target]) != updateRevision.Name && !isTerminating(replicas[target]) {
			// delete the Pod if it is healthy and the revision doesnt match the target
			klog.V(2).Infof("StatefulSet %s/%s terminating Pod %s for update",
				set.Namespace,
				set.Name,
				replicas[target].Name)
			if err := ssc.podControl.DeleteStatefulPod(set, replicas[target]); err != nil {
				if !errors.IsNotFound(err) {
					return nil, err
				}
			}
			deletedPods++
			status.CurrentReplicas--
		}
	}
	return &status, nil
}

// updateStatefulSetStatus updates set's Status to be equal to status. If status indicates a complete update, it is
// mutated to indicate completion. If status is semantically equivalent to set's Status no update is performed. If the
// returned error is nil, the update is successful.
func (ssc *defaultStatefulSetControl) updateStatefulSetStatus(
	ctx context.Context,
	set *apps.StatefulSet,
	status *apps.StatefulSetStatus) error {
	// complete any in progress rolling update if necessary
	completeRollingUpdate(set, status)

	// if the status is not inconsistent do not perform an update
	if !inconsistentStatus(set, status) {
		return nil
	}

	// copy set and update its status
	set = set.DeepCopy()
	if err := ssc.statusUpdater.UpdateStatefulSetStatus(ctx, set, status); err != nil {
		return err
	}

	return nil
}

var _ StatefulSetControlInterface = &defaultStatefulSetControl{}

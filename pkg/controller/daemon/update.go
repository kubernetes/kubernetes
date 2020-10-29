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

package daemon

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"sort"

	"k8s.io/klog/v2"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/daemon/util"
	labelsutil "k8s.io/kubernetes/pkg/util/labels"
)

// rollingUpdate identifies the set of old pods to delete, or additional pods to create on nodes,
// remaining within the constraints imposed by the update strategy.
func (dsc *DaemonSetsController) rollingUpdate(ctx context.Context, ds *apps.DaemonSet, nodeList []*v1.Node, hash string) error {
	logger := klog.FromContext(ctx)
	nodeToDaemonPods, err := dsc.getNodesToDaemonPods(ctx, ds, false)
	if err != nil {
		return fmt.Errorf("couldn't get node to daemon pod mapping for daemon set %q: %v", ds.Name, err)
	}
	maxSurge, maxUnavailable, desiredNumberScheduled, err := dsc.updatedDesiredNodeCounts(ctx, ds, nodeList, nodeToDaemonPods)
	if err != nil {
		return fmt.Errorf("couldn't get unavailable numbers: %v", err)
	}

	now := dsc.failedPodsBackoff.Clock.Now()

	// When not surging, we delete just enough pods to stay under the maxUnavailable limit, if any
	// are necessary, and let the core loop create new instances on those nodes.
	//
	// Assumptions:
	// * Expect manage loop to allow no more than one pod per node
	// * Expect manage loop will create new pods
	// * Expect manage loop will handle failed pods
	// * Deleted pods do not count as unavailable so that updates make progress when nodes are down
	// Invariants:
	// * The number of new pods that are unavailable must be less than maxUnavailable
	// * A node with an available old pod is a candidate for deletion if it does not violate other invariants
	//
	if maxSurge == 0 {
		var numUnavailable int
		var allowedReplacementPods []string
		var candidatePodsToDelete []string
		for nodeName, pods := range nodeToDaemonPods {
			newPod, oldPod, ok := findUpdatedPodsOnNode(ds, pods, hash)
			if !ok {
				// let the manage loop clean up this node, and treat it as an unavailable node
				logger.V(3).Info("DaemonSet has excess pods on node, skipping to allow the core loop to process", "daemonset", klog.KObj(ds), "node", klog.KRef("", nodeName))
				numUnavailable++
				continue
			}
			switch {
			case oldPod == nil && newPod == nil, oldPod != nil && newPod != nil:
				// the manage loop will handle creating or deleting the appropriate pod, consider this unavailable
				numUnavailable++
			case newPod != nil:
				// this pod is up to date, check its availability
				if !podutil.IsPodAvailable(newPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}) {
					// an unavailable new pod is counted against maxUnavailable
					numUnavailable++
				}
			default:
				// this pod is old, it is an update candidate
				switch {
				case !podutil.IsPodAvailable(oldPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}):
					// the old pod isn't available, so it needs to be replaced
					logger.V(5).Info("DaemonSet pod on node is out of date and not available, allowing replacement", "daemonset", klog.KObj(ds), "pod", klog.KObj(oldPod), "node", klog.KRef("", nodeName))
					// record the replacement
					if allowedReplacementPods == nil {
						allowedReplacementPods = make([]string, 0, len(nodeToDaemonPods))
					}
					allowedReplacementPods = append(allowedReplacementPods, oldPod.Name)
					numUnavailable++
				case numUnavailable >= maxUnavailable:
					// no point considering any other candidates
					continue
				default:
					logger.V(5).Info("DaemonSet pod on node is out of date, this is a candidate to replace", "daemonset", klog.KObj(ds), "pod", klog.KObj(oldPod), "node", klog.KRef("", nodeName))
					// record the candidate
					if candidatePodsToDelete == nil {
						candidatePodsToDelete = make([]string, 0, maxUnavailable)
					}
					candidatePodsToDelete = append(candidatePodsToDelete, oldPod.Name)
				}
			}
		}

		// use any of the candidates we can, including the allowedReplacementPods
		logger.V(5).Info("DaemonSet allowing replacements", "daemonset", klog.KObj(ds), "replacements", len(allowedReplacementPods), "maxUnavailable", maxUnavailable, "numUnavailable", numUnavailable, "candidates", len(candidatePodsToDelete))
		remainingUnavailable := maxUnavailable - numUnavailable
		if remainingUnavailable < 0 {
			remainingUnavailable = 0
		}
		if max := len(candidatePodsToDelete); remainingUnavailable > max {
			remainingUnavailable = max
		}
		oldPodsToDelete := append(allowedReplacementPods, candidatePodsToDelete[:remainingUnavailable]...)

		return dsc.syncNodes(ctx, ds, oldPodsToDelete, nil, hash)
	}

	// When surging, we create new pods whenever an old pod is unavailable, and we can create up
	// to maxSurge extra pods
	//
	// Assumptions:
	// * Expect manage loop to allow no more than two pods per node, one old, one new
	// * Expect manage loop will create new pods if there are no pods on node
	// * Expect manage loop will handle failed pods
	// * Deleted pods do not count as unavailable so that updates make progress when nodes are down
	// Invariants:
	// * A node with an unavailable old pod is a candidate for immediate new pod creation
	// * An old available pod is deleted if a new pod is available
	// * No more than maxSurge new pods are created for old available pods at any one time
	//
	var oldPodsToDelete []string          // these pods are already updated or unavailable on sunsetted node
	var shouldNotRunPodsToDelete []string // candidate pods to be deleted on sunsetted nodes
	var candidateNewNodes []string
	var allowedNewNodes []string
	var numSurge int
	var numAvailable int

	for nodeName, pods := range nodeToDaemonPods {
		newPod, oldPod, ok := findUpdatedPodsOnNode(ds, pods, hash)
		if !ok {
			// let the manage loop clean up this node, and treat it as a surge node
			logger.V(3).Info("DaemonSet has excess pods on node, skipping to allow the core loop to process", "daemonset", klog.KObj(ds), "node", klog.KRef("", nodeName))
			numSurge++
			continue
		}

		// first count availability for all the nodes (even the ones that we are sunsetting due to scheduling constraints)
		if oldPod != nil {
			if podutil.IsPodAvailable(oldPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}) {
				numAvailable++
			}
		} else if newPod != nil {
			if podutil.IsPodAvailable(newPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}) {
				numAvailable++
			}
		}

		switch {
		case oldPod == nil:
			// we don't need to do anything to this node, the manage loop will handle it
		case newPod == nil:
			// this is a surge candidate
			switch {
			case !podutil.IsPodAvailable(oldPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}):
				node, err := dsc.nodeLister.Get(nodeName)
				if err != nil {
					return fmt.Errorf("couldn't get node for nodeName %q: %v", nodeName, err)
				}
				if shouldRun, _ := dsc.nodeShouldRunDaemonPod(logger, node, ds); !shouldRun {
					logger.V(5).Info("DaemonSet pod on node is not available and does not match scheduling constraints, remove old pod", "daemonset", klog.KObj(ds), "node", nodeName, "oldPod", klog.KObj(oldPod))
					oldPodsToDelete = append(oldPodsToDelete, oldPod.Name)
					continue
				}
				// the old pod isn't available, allow it to become a replacement
				logger.V(5).Info("Pod on node is out of date and not available, allowing replacement", "daemonset", klog.KObj(ds), "pod", klog.KObj(oldPod), "node", klog.KRef("", nodeName))
				// record the replacement
				if allowedNewNodes == nil {
					allowedNewNodes = make([]string, 0, len(nodeToDaemonPods))
				}
				allowedNewNodes = append(allowedNewNodes, nodeName)
			default:
				node, err := dsc.nodeLister.Get(nodeName)
				if err != nil {
					return fmt.Errorf("couldn't get node for nodeName %q: %v", nodeName, err)
				}
				if shouldRun, _ := dsc.nodeShouldRunDaemonPod(logger, node, ds); !shouldRun {
					shouldNotRunPodsToDelete = append(shouldNotRunPodsToDelete, oldPod.Name)
					continue
				}
				if numSurge >= maxSurge {
					// no point considering any other candidates
					continue
				}
				logger.V(5).Info("DaemonSet pod on node is out of date, this is a surge candidate", "daemonset", klog.KObj(ds), "pod", klog.KObj(oldPod), "node", klog.KRef("", nodeName))
				// record the candidate
				if candidateNewNodes == nil {
					candidateNewNodes = make([]string, 0, maxSurge)
				}
				candidateNewNodes = append(candidateNewNodes, nodeName)
			}
		default:
			// we have already surged onto this node, determine our state
			if !podutil.IsPodAvailable(newPod, ds.Spec.MinReadySeconds, metav1.Time{Time: now}) {
				// we're waiting to go available here
				numSurge++
				continue
			}
			// we're available, delete the old pod
			logger.V(5).Info("DaemonSet pod on node is available, remove old pod", "daemonset", klog.KObj(ds), "newPod", klog.KObj(newPod), "node", nodeName, "oldPod", klog.KObj(oldPod))
			oldPodsToDelete = append(oldPodsToDelete, oldPod.Name)
		}
	}

	// use any of the candidates we can, including the allowedNewNodes
	logger.V(5).Info("DaemonSet allowing replacements", "daemonset", klog.KObj(ds), "replacements", len(allowedNewNodes), "maxSurge", maxSurge, "numSurge", numSurge, "candidates", len(candidateNewNodes))
	remainingSurge := maxSurge - numSurge

	// With maxSurge, the application owner expects 100% availability.
	// When the scheduling constraint change from node A to node B, we do not want the application to stay
	// without any available pods. Only delete a pod on node A when a pod on node B becomes available.
	if deletablePodsNumber := numAvailable - desiredNumberScheduled; deletablePodsNumber > 0 {
		if shouldNotRunPodsToDeleteNumber := len(shouldNotRunPodsToDelete); deletablePodsNumber > shouldNotRunPodsToDeleteNumber {
			deletablePodsNumber = shouldNotRunPodsToDeleteNumber
		}
		for _, podToDeleteName := range shouldNotRunPodsToDelete[:deletablePodsNumber] {
			podToDelete, err := dsc.podLister.Pods(ds.Namespace).Get(podToDeleteName)
			if err != nil {
				if errors.IsNotFound(err) {
					continue
				}
				return fmt.Errorf("couldn't get pod which should be deleted due to scheduling constraints %q: %v", podToDeleteName, err)
			}
			logger.V(5).Info("DaemonSet pod on node should be deleted due to scheduling constraints", "daemonset", klog.KObj(ds), "pod", klog.KObj(podToDelete), "node", podToDelete.Spec.NodeName)
			oldPodsToDelete = append(oldPodsToDelete, podToDeleteName)
		}
	}

	if remainingSurge < 0 {
		remainingSurge = 0
	}
	if max := len(candidateNewNodes); remainingSurge > max {
		remainingSurge = max
	}
	newNodesToCreate := append(allowedNewNodes, candidateNewNodes[:remainingSurge]...)

	return dsc.syncNodes(ctx, ds, oldPodsToDelete, newNodesToCreate, hash)
}

// findUpdatedPodsOnNode looks at non-deleted pods on a given node and returns true if there
// is at most one of each old and new pods, or false if there are multiples. We can skip
// processing the particular node in those scenarios and let the manage loop prune the
// excess pods for our next time around.
func findUpdatedPodsOnNode(ds *apps.DaemonSet, podsOnNode []*v1.Pod, hash string) (newPod, oldPod *v1.Pod, ok bool) {
	for _, pod := range podsOnNode {
		if pod.DeletionTimestamp != nil {
			continue
		}
		generation, err := util.GetTemplateGeneration(ds)
		if err != nil {
			generation = nil
		}
		if util.IsPodUpdated(pod, hash, generation) {
			if newPod != nil {
				return nil, nil, false
			}
			newPod = pod
		} else {
			if oldPod != nil {
				return nil, nil, false
			}
			oldPod = pod
		}
	}
	return newPod, oldPod, true
}

// constructHistory finds all histories controlled by the given DaemonSet, and
// update current history revision number, or create current history if need to.
// It also deduplicates current history, and adds missing unique labels to existing histories.
func (dsc *DaemonSetsController) constructHistory(ctx context.Context, ds *apps.DaemonSet) (cur *apps.ControllerRevision, old []*apps.ControllerRevision, err error) {
	var histories []*apps.ControllerRevision
	var currentHistories []*apps.ControllerRevision
	histories, err = dsc.controlledHistories(ctx, ds)
	if err != nil {
		return nil, nil, err
	}
	for _, history := range histories {
		// Add the unique label if it's not already added to the history
		// We use history name instead of computing hash, so that we don't need to worry about hash collision
		if _, ok := history.Labels[apps.DefaultDaemonSetUniqueLabelKey]; !ok {
			toUpdate := history.DeepCopy()
			toUpdate.Labels[apps.DefaultDaemonSetUniqueLabelKey] = toUpdate.Name
			history, err = dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Update(ctx, toUpdate, metav1.UpdateOptions{})
			if err != nil {
				return nil, nil, err
			}
		}
		// Compare histories with ds to separate cur and old history
		found := false
		found, err = Match(ds, history)
		if err != nil {
			return nil, nil, err
		}
		if found {
			currentHistories = append(currentHistories, history)
		} else {
			old = append(old, history)
		}
	}

	currRevision := maxRevision(old) + 1
	switch len(currentHistories) {
	case 0:
		// Create a new history if the current one isn't found
		cur, err = dsc.snapshot(ctx, ds, currRevision)
		if err != nil {
			return nil, nil, err
		}
	default:
		cur, err = dsc.dedupCurHistories(ctx, ds, currentHistories)
		if err != nil {
			return nil, nil, err
		}
		// Update revision number if necessary
		if cur.Revision < currRevision {
			toUpdate := cur.DeepCopy()
			toUpdate.Revision = currRevision
			_, err = dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Update(ctx, toUpdate, metav1.UpdateOptions{})
			if err != nil {
				return nil, nil, err
			}
		}
	}
	return cur, old, err
}

func (dsc *DaemonSetsController) cleanupHistory(ctx context.Context, ds *apps.DaemonSet, old []*apps.ControllerRevision) error {
	// Include deleted terminal pods when maintaining history.
	nodesToDaemonPods, err := dsc.getNodesToDaemonPods(ctx, ds, true)
	if err != nil {
		return fmt.Errorf("couldn't get node to daemon pod mapping for daemon set %q: %v", ds.Name, err)
	}

	toKeep := int(*ds.Spec.RevisionHistoryLimit)
	toKill := len(old) - toKeep
	if toKill <= 0 {
		return nil
	}

	// Find all hashes of live pods
	liveHashes := make(map[string]bool)
	for _, pods := range nodesToDaemonPods {
		for _, pod := range pods {
			if hash := pod.Labels[apps.DefaultDaemonSetUniqueLabelKey]; len(hash) > 0 {
				liveHashes[hash] = true
			}
		}
	}

	// Clean up old history from smallest to highest revision (from oldest to newest)
	sort.Sort(historiesByRevision(old))
	for _, history := range old {
		if toKill <= 0 {
			break
		}
		if hash := history.Labels[apps.DefaultDaemonSetUniqueLabelKey]; liveHashes[hash] {
			continue
		}
		// Clean up
		err := dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Delete(ctx, history.Name, metav1.DeleteOptions{})
		if err != nil {
			return err
		}
		toKill--
	}
	return nil
}

// maxRevision returns the max revision number of the given list of histories
func maxRevision(histories []*apps.ControllerRevision) int64 {
	max := int64(0)
	for _, history := range histories {
		if history.Revision > max {
			max = history.Revision
		}
	}
	return max
}

func (dsc *DaemonSetsController) dedupCurHistories(ctx context.Context, ds *apps.DaemonSet, curHistories []*apps.ControllerRevision) (*apps.ControllerRevision, error) {
	if len(curHistories) == 1 {
		return curHistories[0], nil
	}
	var maxRevision int64
	var keepCur *apps.ControllerRevision
	for _, cur := range curHistories {
		if cur.Revision >= maxRevision {
			keepCur = cur
			maxRevision = cur.Revision
		}
	}
	// Relabel pods before dedup
	pods, err := dsc.getDaemonPods(ctx, ds)
	if err != nil {
		return nil, err
	}
	for _, pod := range pods {
		if pod.Labels[apps.DefaultDaemonSetUniqueLabelKey] != keepCur.Labels[apps.DefaultDaemonSetUniqueLabelKey] {
			patchRaw := map[string]interface{}{
				"metadata": map[string]interface{}{
					"labels": map[string]interface{}{
						apps.DefaultDaemonSetUniqueLabelKey: keepCur.Labels[apps.DefaultDaemonSetUniqueLabelKey],
					},
				},
			}
			patchJSON, err := json.Marshal(patchRaw)
			if err != nil {
				return nil, err
			}
			_, err = dsc.kubeClient.CoreV1().Pods(ds.Namespace).Patch(ctx, pod.Name, types.MergePatchType, patchJSON, metav1.PatchOptions{})
			if err != nil {
				return nil, err
			}
		}
	}
	// Clean up duplicates
	for _, cur := range curHistories {
		if cur.Name == keepCur.Name {
			continue
		}
		// Remove duplicates
		err = dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Delete(ctx, cur.Name, metav1.DeleteOptions{})
		if err != nil {
			return nil, err
		}
	}
	return keepCur, nil
}

// controlledHistories returns all ControllerRevisions controlled by the given DaemonSet.
// This also reconciles ControllerRef by adopting/orphaning.
// Note that returned histories are pointers to objects in the cache.
// If you want to modify one, you need to deep-copy it first.
func (dsc *DaemonSetsController) controlledHistories(ctx context.Context, ds *apps.DaemonSet) ([]*apps.ControllerRevision, error) {
	selector, err := metav1.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		return nil, err
	}

	// List all histories to include those that don't match the selector anymore
	// but have a ControllerRef pointing to the controller.
	histories, err := dsc.historyLister.ControllerRevisions(ds.Namespace).List(labels.Everything())
	if err != nil {
		return nil, err
	}
	// If any adoptions are attempted, we should first recheck for deletion with
	// an uncached quorum read sometime after listing Pods (see #42639).
	canAdoptFunc := controller.RecheckDeletionTimestamp(func(ctx context.Context) (metav1.Object, error) {
		fresh, err := dsc.kubeClient.AppsV1().DaemonSets(ds.Namespace).Get(ctx, ds.Name, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		if fresh.UID != ds.UID {
			return nil, fmt.Errorf("original DaemonSet %v/%v is gone: got uid %v, wanted %v", ds.Namespace, ds.Name, fresh.UID, ds.UID)
		}
		return fresh, nil
	})
	// Use ControllerRefManager to adopt/orphan as needed.
	cm := controller.NewControllerRevisionControllerRefManager(dsc.crControl, ds, selector, controllerKind, canAdoptFunc)
	return cm.ClaimControllerRevisions(ctx, histories)
}

// Match check if the given DaemonSet's template matches the template stored in the given history.
func Match(ds *apps.DaemonSet, history *apps.ControllerRevision) (bool, error) {
	patch, err := getPatch(ds)
	if err != nil {
		return false, err
	}
	return bytes.Equal(patch, history.Data.Raw), nil
}

// getPatch returns a strategic merge patch that can be applied to restore a Daemonset to a
// previous version. If the returned error is nil the patch is valid. The current state that we save is just the
// PodSpecTemplate. We can modify this later to encompass more state (or less) and remain compatible with previously
// recorded patches.
func getPatch(ds *apps.DaemonSet) ([]byte, error) {
	dsBytes, err := json.Marshal(ds)
	if err != nil {
		return nil, err
	}
	var raw map[string]interface{}
	err = json.Unmarshal(dsBytes, &raw)
	if err != nil {
		return nil, err
	}
	objCopy := make(map[string]interface{})
	specCopy := make(map[string]interface{})

	// Create a patch of the DaemonSet that replaces spec.template
	spec := raw["spec"].(map[string]interface{})
	template := spec["template"].(map[string]interface{})
	specCopy["template"] = template
	template["$patch"] = "replace"
	objCopy["spec"] = specCopy
	patch, err := json.Marshal(objCopy)
	return patch, err
}

func (dsc *DaemonSetsController) snapshot(ctx context.Context, ds *apps.DaemonSet, revision int64) (*apps.ControllerRevision, error) {
	patch, err := getPatch(ds)
	if err != nil {
		return nil, err
	}
	hash := controller.ComputeHash(&ds.Spec.Template, ds.Status.CollisionCount)
	name := ds.Name + "-" + hash
	history := &apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       ds.Namespace,
			Labels:          labelsutil.CloneAndAddLabel(ds.Spec.Template.Labels, apps.DefaultDaemonSetUniqueLabelKey, hash),
			Annotations:     ds.Annotations,
			OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(ds, controllerKind)},
		},
		Data:     runtime.RawExtension{Raw: patch},
		Revision: revision,
	}

	history, err = dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Create(ctx, history, metav1.CreateOptions{})
	if outerErr := err; errors.IsAlreadyExists(outerErr) {
		logger := klog.FromContext(ctx)
		// TODO: Is it okay to get from historyLister?
		existedHistory, getErr := dsc.kubeClient.AppsV1().ControllerRevisions(ds.Namespace).Get(ctx, name, metav1.GetOptions{})
		if getErr != nil {
			return nil, getErr
		}
		// Check if we already created it
		done, matchErr := Match(ds, existedHistory)
		if matchErr != nil {
			return nil, matchErr
		}
		if done {
			return existedHistory, nil
		}

		// Handle name collisions between different history
		// Get the latest DaemonSet from the API server to make sure collision count is only increased when necessary
		currDS, getErr := dsc.kubeClient.AppsV1().DaemonSets(ds.Namespace).Get(ctx, ds.Name, metav1.GetOptions{})
		if getErr != nil {
			return nil, getErr
		}
		// If the collision count used to compute hash was in fact stale, there's no need to bump collision count; retry again
		if !reflect.DeepEqual(currDS.Status.CollisionCount, ds.Status.CollisionCount) {
			return nil, fmt.Errorf("found a stale collision count (%d, expected %d) of DaemonSet %q while processing; will retry until it is updated", ds.Status.CollisionCount, currDS.Status.CollisionCount, ds.Name)
		}
		if currDS.Status.CollisionCount == nil {
			currDS.Status.CollisionCount = new(int32)
		}
		*currDS.Status.CollisionCount++
		_, updateErr := dsc.kubeClient.AppsV1().DaemonSets(ds.Namespace).UpdateStatus(ctx, currDS, metav1.UpdateOptions{})
		if updateErr != nil {
			return nil, updateErr
		}
		logger.V(2).Info("Found a hash collision for DaemonSet - bumping collisionCount to resolve it", "daemonset", klog.KObj(ds), "collisionCount", *currDS.Status.CollisionCount)
		return nil, outerErr
	}
	return history, err
}

// updatedDesiredNodeCounts calculates the true number of allowed surge, unavailable or desired scheduled pods and
// updates the nodeToDaemonPods array to include an empty array for every node that is not scheduled.
func (dsc *DaemonSetsController) updatedDesiredNodeCounts(ctx context.Context, ds *apps.DaemonSet, nodeList []*v1.Node, nodeToDaemonPods map[string][]*v1.Pod) (int, int, int, error) {
	var desiredNumberScheduled int
	logger := klog.FromContext(ctx)
	for i := range nodeList {
		node := nodeList[i]
		wantToRun, _ := dsc.nodeShouldRunDaemonPod(logger, node, ds)
		if !wantToRun {
			continue
		}
		desiredNumberScheduled++

		if _, exists := nodeToDaemonPods[node.Name]; !exists {
			nodeToDaemonPods[node.Name] = nil
		}
	}

	maxUnavailable, err := util.UnavailableCount(ds, desiredNumberScheduled)
	if err != nil {
		return -1, -1, -1, fmt.Errorf("invalid value for MaxUnavailable: %v", err)
	}

	maxSurge, err := util.SurgeCount(ds, desiredNumberScheduled)
	if err != nil {
		return -1, -1, -1, fmt.Errorf("invalid value for MaxSurge: %v", err)
	}

	// if the daemonset returned with an impossible configuration, obey the default of unavailable=1 (in the
	// event the apiserver returns 0 for both surge and unavailability)
	if desiredNumberScheduled > 0 && maxUnavailable == 0 && maxSurge == 0 {
		logger.Info("DaemonSet is not configured for surge or unavailability, defaulting to accepting unavailability", "daemonset", klog.KObj(ds))
		maxUnavailable = 1
	}
	logger.V(5).Info("DaemonSet with maxSurge and maxUnavailable", "daemonset", klog.KObj(ds), "maxSurge", maxSurge, "maxUnavailable", maxUnavailable)
	return maxSurge, maxUnavailable, desiredNumberScheduled, nil
}

type historiesByRevision []*apps.ControllerRevision

func (h historiesByRevision) Len() int      { return len(h) }
func (h historiesByRevision) Swap(i, j int) { h[i], h[j] = h[j], h[i] }
func (h historiesByRevision) Less(i, j int) bool {
	return h[i].Revision < h[j].Revision
}

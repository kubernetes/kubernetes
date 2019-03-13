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

package persistentvolume

import (
	"fmt"
	"sort"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage/etcd"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	"k8s.io/klog"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// SchedulerVolumeBinder is used by the scheduler to handle PVC/PV binding
// and dynamic provisioning.  The binding decisions are integrated into the pod scheduling
// workflow so that the PV NodeAffinity is also considered along with the pod's other
// scheduling requirements.
//
// This integrates into the existing default scheduler workflow as follows:
// 1. The scheduler takes a Pod off the scheduler queue and processes it serially:
//    a. Invokes all predicate functions, parallelized across nodes.  FindPodVolumes() is invoked here.
//    b. Invokes all priority functions.  Future/TBD
//    c. Selects the best node for the Pod.
//    d. Cache the node selection for the Pod. AssumePodVolumes() is invoked here.
//       i.  If PVC binding is required, cache in-memory only:
//           * For manual binding: update PV objects for prebinding to the corresponding PVCs.
//           * For dynamic provisioning: update PVC object with a selected node from c)
//           * For the pod, which PVCs and PVs need API updates.
//       ii. Afterwards, the main scheduler caches the Pod->Node binding in the scheduler's pod cache,
//           This is handled in the scheduler and not here.
//    e. Asynchronously bind volumes and pod in a separate goroutine
//        i.  BindPodVolumes() is called first. It makes all the necessary API updates and waits for
//            PV controller to fully bind and provision the PVCs. If binding fails, the Pod is sent
//            back through the scheduler.
//        ii. After BindPodVolumes() is complete, then the scheduler does the final Pod->Node binding.
// 2. Once all the assume operations are done in d), the scheduler processes the next Pod in the scheduler queue
//    while the actual binding operation occurs in the background.
type SchedulerVolumeBinder interface {
	// FindPodVolumes checks if all of a Pod's PVCs can be satisfied by the node.
	//
	// If a PVC is bound, it checks if the PV's NodeAffinity matches the Node.
	// Otherwise, it tries to find an available PV to bind to the PVC.
	//
	// It returns true if all of the Pod's PVCs have matching PVs or can be dynamic provisioned,
	// and returns true if bound volumes satisfy the PV NodeAffinity.
	//
	// This function is called by the volume binding scheduler predicate and can be called in parallel
	FindPodVolumes(pod *v1.Pod, node *v1.Node) (unboundVolumesSatisified, boundVolumesSatisfied bool, err error)

	// AssumePodVolumes will:
	// 1. Take the PV matches for unbound PVCs and update the PV cache assuming
	// that the PV is prebound to the PVC.
	// 2. Take the PVCs that need provisioning and update the PVC cache with related
	// annotations set.
	//
	// It returns true if all volumes are fully bound
	//
	// This function will modify assumedPod with the node name.
	// This function is called serially.
	AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (allFullyBound bool, err error)

	// BindPodVolumes will:
	// 1. Initiate the volume binding by making the API call to prebind the PV
	// to its matching PVC.
	// 2. Trigger the volume provisioning by making the API call to set related
	// annotations on the PVC
	// 3. Wait for PVCs to be completely bound by the PV controller
	//
	// This function can be called in parallel.
	BindPodVolumes(assumedPod *v1.Pod) error

	// GetBindingsCache returns the cache used (if any) to store volume binding decisions.
	GetBindingsCache() PodBindingCache
}

type volumeBinder struct {
	kubeClient  clientset.Interface
	classLister storagelisters.StorageClassLister

	nodeInformer coreinformers.NodeInformer
	pvcCache     PVCAssumeCache
	pvCache      PVAssumeCache

	// Stores binding decisions that were made in FindPodVolumes for use in AssumePodVolumes.
	// AssumePodVolumes modifies the bindings again for use in BindPodVolumes.
	podBindingCache PodBindingCache

	// Amount of time to wait for the bind operation to succeed
	bindTimeout time.Duration
}

// NewVolumeBinder sets up all the caches needed for the scheduler to make volume binding decisions.
func NewVolumeBinder(
	kubeClient clientset.Interface,
	nodeInformer coreinformers.NodeInformer,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	storageClassInformer storageinformers.StorageClassInformer,
	bindTimeout time.Duration) SchedulerVolumeBinder {

	b := &volumeBinder{
		kubeClient:      kubeClient,
		classLister:     storageClassInformer.Lister(),
		nodeInformer:    nodeInformer,
		pvcCache:        NewPVCAssumeCache(pvcInformer.Informer()),
		pvCache:         NewPVAssumeCache(pvInformer.Informer()),
		podBindingCache: NewPodBindingCache(),
		bindTimeout:     bindTimeout,
	}

	return b
}

func (b *volumeBinder) GetBindingsCache() PodBindingCache {
	return b.podBindingCache
}

func podHasClaims(pod *v1.Pod) bool {
	for _, vol := range pod.Spec.Volumes {
		if vol.PersistentVolumeClaim != nil {
			return true
		}
	}
	return false
}

// FindPodVolumes caches the matching PVs and PVCs to provision per node in podBindingCache.
// This method intentionally takes in a *v1.Node object instead of using volumebinder.nodeInformer.
// That's necessary because some operations will need to pass in to the predicate fake node objects.
func (b *volumeBinder) FindPodVolumes(pod *v1.Pod, node *v1.Node) (unboundVolumesSatisfied, boundVolumesSatisfied bool, err error) {
	podName := getPodName(pod)

	// Warning: Below log needs high verbosity as it can be printed several times (#60933).
	klog.V(5).Infof("FindPodVolumes for pod %q, node %q", podName, node.Name)

	// Initialize to true for pods that don't have volumes
	unboundVolumesSatisfied = true
	boundVolumesSatisfied = true
	start := time.Now()
	defer func() {
		VolumeSchedulingStageLatency.WithLabelValues("predicate").Observe(time.Since(start).Seconds())
		if err != nil {
			VolumeSchedulingStageFailed.WithLabelValues("predicate").Inc()
		}
	}()

	if !podHasClaims(pod) {
		// Fast path
		return unboundVolumesSatisfied, boundVolumesSatisfied, nil
	}

	var (
		matchedClaims     []*bindingInfo
		provisionedClaims []*v1.PersistentVolumeClaim
	)
	defer func() {
		// We recreate bindings for each new schedule loop.
		if len(matchedClaims) == 0 && len(provisionedClaims) == 0 {
			// Clear cache if no claims to bind or provision for this node.
			b.podBindingCache.ClearBindings(pod, node.Name)
			return
		}
		// Although we do not distinguish nil from empty in this function, for
		// easier testing, we normalize empty to nil.
		if len(matchedClaims) == 0 {
			matchedClaims = nil
		}
		if len(provisionedClaims) == 0 {
			provisionedClaims = nil
		}
		// Mark cache with all matched and provisioned claims for this node
		b.podBindingCache.UpdateBindings(pod, node.Name, matchedClaims, provisionedClaims)
	}()

	// The pod's volumes need to be processed in one call to avoid the race condition where
	// volumes can get bound/provisioned in between calls.
	boundClaims, claimsToBind, unboundClaimsImmediate, err := b.getPodVolumes(pod)
	if err != nil {
		return false, false, err
	}

	// Immediate claims should be bound
	if len(unboundClaimsImmediate) > 0 {
		return false, false, fmt.Errorf("pod has unbound immediate PersistentVolumeClaims")
	}

	// Check PV node affinity on bound volumes
	if len(boundClaims) > 0 {
		boundVolumesSatisfied, err = b.checkBoundClaims(boundClaims, node, podName)
		if err != nil {
			return false, false, err
		}
	}

	// Find matching volumes and node for unbound claims
	if len(claimsToBind) > 0 {
		var (
			claimsToFindMatching []*v1.PersistentVolumeClaim
			claimsToProvision    []*v1.PersistentVolumeClaim
		)

		// Filter out claims to provision
		for _, claim := range claimsToBind {
			if selectedNode, ok := claim.Annotations[annSelectedNode]; ok {
				if selectedNode != node.Name {
					// Fast path, skip unmatched node
					return false, boundVolumesSatisfied, nil
				}
				claimsToProvision = append(claimsToProvision, claim)
			} else {
				claimsToFindMatching = append(claimsToFindMatching, claim)
			}
		}

		// Find matching volumes
		if len(claimsToFindMatching) > 0 {
			var unboundClaims []*v1.PersistentVolumeClaim
			unboundVolumesSatisfied, matchedClaims, unboundClaims, err = b.findMatchingVolumes(pod, claimsToFindMatching, node)
			if err != nil {
				return false, false, err
			}
			claimsToProvision = append(claimsToProvision, unboundClaims...)
		}

		// Check for claims to provision
		if len(claimsToProvision) > 0 {
			unboundVolumesSatisfied, provisionedClaims, err = b.checkVolumeProvisions(pod, claimsToProvision, node)
			if err != nil {
				return false, false, err
			}
		}
	}

	return unboundVolumesSatisfied, boundVolumesSatisfied, nil
}

// AssumePodVolumes will take the cached matching PVs and PVCs to provision
// in podBindingCache for the chosen node, and:
// 1. Update the pvCache with the new prebound PV.
// 2. Update the pvcCache with the new PVCs with annotations set
// 3. Update podBindingCache again with cached API updates for PVs and PVCs.
func (b *volumeBinder) AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (allFullyBound bool, err error) {
	podName := getPodName(assumedPod)

	klog.V(4).Infof("AssumePodVolumes for pod %q, node %q", podName, nodeName)
	start := time.Now()
	defer func() {
		VolumeSchedulingStageLatency.WithLabelValues("assume").Observe(time.Since(start).Seconds())
		if err != nil {
			VolumeSchedulingStageFailed.WithLabelValues("assume").Inc()
		}
	}()

	if allBound := b.arePodVolumesBound(assumedPod); allBound {
		klog.V(4).Infof("AssumePodVolumes for pod %q, node %q: all PVCs bound and nothing to do", podName, nodeName)
		return true, nil
	}

	assumedPod.Spec.NodeName = nodeName

	claimsToBind := b.podBindingCache.GetBindings(assumedPod, nodeName)
	claimsToProvision := b.podBindingCache.GetProvisionedPVCs(assumedPod, nodeName)

	// Assume PV
	newBindings := []*bindingInfo{}
	for _, binding := range claimsToBind {
		newPV, dirty, err := GetBindVolumeToClaim(binding.pv, binding.pvc)
		klog.V(5).Infof("AssumePodVolumes: GetBindVolumeToClaim for pod %q, PV %q, PVC %q.  newPV %p, dirty %v, err: %v",
			podName,
			binding.pv.Name,
			binding.pvc.Name,
			newPV,
			dirty,
			err)
		if err != nil {
			b.revertAssumedPVs(newBindings)
			return false, err
		}
		// TODO: can we assume everytime?
		if dirty {
			err = b.pvCache.Assume(newPV)
			if err != nil {
				b.revertAssumedPVs(newBindings)
				return false, err
			}
		}
		newBindings = append(newBindings, &bindingInfo{pv: newPV, pvc: binding.pvc})
	}

	// Assume PVCs
	newProvisionedPVCs := []*v1.PersistentVolumeClaim{}
	for _, claim := range claimsToProvision {
		// The claims from method args can be pointing to watcher cache. We must not
		// modify these, therefore create a copy.
		claimClone := claim.DeepCopy()
		metav1.SetMetaDataAnnotation(&claimClone.ObjectMeta, annSelectedNode, nodeName)
		err = b.pvcCache.Assume(claimClone)
		if err != nil {
			b.revertAssumedPVs(newBindings)
			b.revertAssumedPVCs(newProvisionedPVCs)
			return
		}

		newProvisionedPVCs = append(newProvisionedPVCs, claimClone)
	}

	// Update cache with the assumed pvcs and pvs
	// Even if length is zero, update the cache with an empty slice to indicate that no
	// operations are needed
	b.podBindingCache.UpdateBindings(assumedPod, nodeName, newBindings, newProvisionedPVCs)

	return
}

// BindPodVolumes gets the cached bindings and PVCs to provision in podBindingCache,
// makes the API update for those PVs/PVCs, and waits for the PVCs to be completely bound
// by the PV controller.
func (b *volumeBinder) BindPodVolumes(assumedPod *v1.Pod) (err error) {
	podName := getPodName(assumedPod)
	klog.V(4).Infof("BindPodVolumes for pod %q, node %q", podName, assumedPod.Spec.NodeName)

	start := time.Now()
	defer func() {
		VolumeSchedulingStageLatency.WithLabelValues("bind").Observe(time.Since(start).Seconds())
		if err != nil {
			VolumeSchedulingStageFailed.WithLabelValues("bind").Inc()
		}
	}()

	bindings := b.podBindingCache.GetBindings(assumedPod, assumedPod.Spec.NodeName)
	claimsToProvision := b.podBindingCache.GetProvisionedPVCs(assumedPod, assumedPod.Spec.NodeName)

	// Start API operations
	err = b.bindAPIUpdate(podName, bindings, claimsToProvision)
	if err != nil {
		return err
	}

	return wait.Poll(time.Second, b.bindTimeout, func() (bool, error) {
		b, err := b.checkBindings(assumedPod, bindings, claimsToProvision)
		return b, err
	})
}

func getPodName(pod *v1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

func getPVCName(pvc *v1.PersistentVolumeClaim) string {
	return pvc.Namespace + "/" + pvc.Name
}

// bindAPIUpdate gets the cached bindings and PVCs to provision in podBindingCache
// and makes the API update for those PVs/PVCs.
func (b *volumeBinder) bindAPIUpdate(podName string, bindings []*bindingInfo, claimsToProvision []*v1.PersistentVolumeClaim) error {
	if bindings == nil {
		return fmt.Errorf("failed to get cached bindings for pod %q", podName)
	}
	if claimsToProvision == nil {
		return fmt.Errorf("failed to get cached claims to provision for pod %q", podName)
	}

	lastProcessedBinding := 0
	lastProcessedProvisioning := 0
	defer func() {
		// only revert assumed cached updates for volumes we haven't successfully bound
		if lastProcessedBinding < len(bindings) {
			b.revertAssumedPVs(bindings[lastProcessedBinding:])
		}
		// only revert assumed cached updates for claims we haven't updated,
		if lastProcessedProvisioning < len(claimsToProvision) {
			b.revertAssumedPVCs(claimsToProvision[lastProcessedProvisioning:])
		}
	}()

	var (
		binding *bindingInfo
		i       int
		claim   *v1.PersistentVolumeClaim
	)

	// Do the actual prebinding. Let the PV controller take care of the rest
	// There is no API rollback if the actual binding fails
	for _, binding = range bindings {
		klog.V(5).Infof("bindAPIUpdate: Pod %q, binding PV %q to PVC %q", podName, binding.pv.Name, binding.pvc.Name)
		// TODO: does it hurt if we make an api call and nothing needs to be updated?
		claimKey := claimToClaimKey(binding.pvc)
		klog.V(2).Infof("claim %q bound to volume %q", claimKey, binding.pv.Name)
		if newPV, err := b.kubeClient.CoreV1().PersistentVolumes().Update(binding.pv); err != nil {
			klog.V(4).Infof("updating PersistentVolume[%s]: binding to %q failed: %v", binding.pv.Name, claimKey, err)
			return err
		} else {
			klog.V(4).Infof("updating PersistentVolume[%s]: bound to %q", binding.pv.Name, claimKey)
			// Save updated object from apiserver for later checking.
			binding.pv = newPV
		}
		lastProcessedBinding++
	}

	// Update claims objects to trigger volume provisioning. Let the PV controller take care of the rest
	// PV controller is expect to signal back by removing related annotations if actual provisioning fails
	for i, claim = range claimsToProvision {
		klog.V(5).Infof("bindAPIUpdate: Pod %q, PVC %q", podName, getPVCName(claim))
		if newClaim, err := b.kubeClient.CoreV1().PersistentVolumeClaims(claim.Namespace).Update(claim); err != nil {
			return err
		} else {
			// Save updated object from apiserver for later checking.
			claimsToProvision[i] = newClaim
		}
		lastProcessedProvisioning++
	}

	return nil
}

var (
	versioner = etcd.APIObjectVersioner{}
)

// checkBindings runs through all the PVCs in the Pod and checks:
// * if the PVC is fully bound
// * if there are any conditions that require binding to fail and be retried
//
// It returns true when all of the Pod's PVCs are fully bound, and error if
// binding (and scheduling) needs to be retried
// Note that it checks on API objects not PV/PVC cache, this is because
// PV/PVC cache can be assumed again in main scheduler loop, we must check
// latest state in API server which are shared with PV controller and
// provisioners
func (b *volumeBinder) checkBindings(pod *v1.Pod, bindings []*bindingInfo, claimsToProvision []*v1.PersistentVolumeClaim) (bool, error) {
	podName := getPodName(pod)
	if bindings == nil {
		return false, fmt.Errorf("failed to get cached bindings for pod %q", podName)
	}
	if claimsToProvision == nil {
		return false, fmt.Errorf("failed to get cached claims to provision for pod %q", podName)
	}

	node, err := b.nodeInformer.Lister().Get(pod.Spec.NodeName)
	if err != nil {
		return false, fmt.Errorf("failed to get node %q: %v", pod.Spec.NodeName, err)
	}

	// Check for any conditions that might require scheduling retry

	// When pod is removed from scheduling queue because of deletion or any
	// other reasons, binding operation should be cancelled. There is no need
	// to check PV/PVC bindings any more.
	// We check pod binding cache here which will be cleared when pod is
	// removed from scheduling queue.
	if b.podBindingCache.GetDecisions(pod) == nil {
		return false, fmt.Errorf("pod %q does not exist any more", podName)
	}

	for _, binding := range bindings {
		pv, err := b.pvCache.GetAPIPV(binding.pv.Name)
		if err != nil {
			return false, fmt.Errorf("failed to check binding: %v", err)
		}

		pvc, err := b.pvcCache.GetAPIPVC(getPVCName(binding.pvc))
		if err != nil {
			return false, fmt.Errorf("failed to check binding: %v", err)
		}

		// Because we updated PV in apiserver, skip if API object is older
		// and wait for new API object propagated from apiserver.
		if versioner.CompareResourceVersion(binding.pv, pv) > 0 {
			return false, nil
		}

		// Check PV's node affinity (the node might not have the proper label)
		if err := volumeutil.CheckNodeAffinity(pv, node.Labels); err != nil {
			return false, fmt.Errorf("pv %q node affinity doesn't match node %q: %v", pv.Name, node.Name, err)
		}

		// Check if pv.ClaimRef got dropped by unbindVolume()
		if pv.Spec.ClaimRef == nil || pv.Spec.ClaimRef.UID == "" {
			return false, fmt.Errorf("ClaimRef got reset for pv %q", pv.Name)
		}

		// Check if pvc is fully bound
		if !b.isPVCFullyBound(pvc) {
			return false, nil
		}
	}

	for _, claim := range claimsToProvision {
		pvc, err := b.pvcCache.GetAPIPVC(getPVCName(claim))
		if err != nil {
			return false, fmt.Errorf("failed to check provisioning pvc: %v", err)
		}

		// Because we updated PVC in apiserver, skip if API object is older
		// and wait for new API object propagated from apiserver.
		if versioner.CompareResourceVersion(claim, pvc) > 0 {
			return false, nil
		}

		// Check if selectedNode annotation is still set
		if pvc.Annotations == nil {
			return false, fmt.Errorf("selectedNode annotation reset for PVC %q", pvc.Name)
		}
		selectedNode := pvc.Annotations[annSelectedNode]
		if selectedNode != pod.Spec.NodeName {
			return false, fmt.Errorf("selectedNode annotation value %q not set to scheduled node %q", selectedNode, pod.Spec.NodeName)
		}

		// If the PVC is bound to a PV, check its node affinity
		if pvc.Spec.VolumeName != "" {
			pv, err := b.pvCache.GetAPIPV(pvc.Spec.VolumeName)
			if err != nil {
				if _, ok := err.(*errNotFound); ok {
					// We tolerate NotFound error here, because PV is possibly
					// not found because of API delay, we can check next time.
					// And if PV does not exist because it's deleted, PVC will
					// be unbound eventually.
					return false, nil
				} else {
					return false, fmt.Errorf("failed to get pv %q from cache: %v", pvc.Spec.VolumeName, err)
				}
			}
			if err := volumeutil.CheckNodeAffinity(pv, node.Labels); err != nil {
				return false, fmt.Errorf("pv %q node affinity doesn't match node %q: %v", pv.Name, node.Name, err)
			}
		}

		// Check if pvc is fully bound
		if !b.isPVCFullyBound(pvc) {
			return false, nil
		}
	}

	// All pvs and pvcs that we operated on are bound
	klog.V(4).Infof("All PVCs for pod %q are bound", podName)
	return true, nil
}

func (b *volumeBinder) isVolumeBound(namespace string, vol *v1.Volume) (bool, *v1.PersistentVolumeClaim, error) {
	if vol.PersistentVolumeClaim == nil {
		return true, nil, nil
	}

	pvcName := vol.PersistentVolumeClaim.ClaimName
	return b.isPVCBound(namespace, pvcName)
}

func (b *volumeBinder) isPVCBound(namespace, pvcName string) (bool, *v1.PersistentVolumeClaim, error) {
	claim := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pvcName,
			Namespace: namespace,
		},
	}
	pvcKey := getPVCName(claim)
	pvc, err := b.pvcCache.GetPVC(pvcKey)
	if err != nil || pvc == nil {
		return false, nil, fmt.Errorf("error getting PVC %q: %v", pvcKey, err)
	}

	fullyBound := b.isPVCFullyBound(pvc)
	if fullyBound {
		klog.V(5).Infof("PVC %q is fully bound to PV %q", pvcKey, pvc.Spec.VolumeName)
	} else {
		if pvc.Spec.VolumeName != "" {
			klog.V(5).Infof("PVC %q is not fully bound to PV %q", pvcKey, pvc.Spec.VolumeName)
		} else {
			klog.V(5).Infof("PVC %q is not bound", pvcKey)
		}
	}
	return fullyBound, pvc, nil
}

func (b *volumeBinder) isPVCFullyBound(pvc *v1.PersistentVolumeClaim) bool {
	return pvc.Spec.VolumeName != "" && metav1.HasAnnotation(pvc.ObjectMeta, annBindCompleted)
}

// arePodVolumesBound returns true if all volumes are fully bound
func (b *volumeBinder) arePodVolumesBound(pod *v1.Pod) bool {
	for _, vol := range pod.Spec.Volumes {
		if isBound, _, _ := b.isVolumeBound(pod.Namespace, &vol); !isBound {
			// Pod has at least one PVC that needs binding
			return false
		}
	}
	return true
}

// getPodVolumes returns a pod's PVCs separated into bound, unbound with delayed binding (including provisioning)
// and unbound with immediate binding (including prebound)
func (b *volumeBinder) getPodVolumes(pod *v1.Pod) (boundClaims []*v1.PersistentVolumeClaim, unboundClaims []*v1.PersistentVolumeClaim, unboundClaimsImmediate []*v1.PersistentVolumeClaim, err error) {
	boundClaims = []*v1.PersistentVolumeClaim{}
	unboundClaimsImmediate = []*v1.PersistentVolumeClaim{}
	unboundClaims = []*v1.PersistentVolumeClaim{}

	for _, vol := range pod.Spec.Volumes {
		volumeBound, pvc, err := b.isVolumeBound(pod.Namespace, &vol)
		if err != nil {
			return nil, nil, nil, err
		}
		if pvc == nil {
			continue
		}
		if volumeBound {
			boundClaims = append(boundClaims, pvc)
		} else {
			delayBindingMode, err := IsDelayBindingMode(pvc, b.classLister)
			if err != nil {
				return nil, nil, nil, err
			}
			// Prebound PVCs are treated as unbound immediate binding
			if delayBindingMode && pvc.Spec.VolumeName == "" {
				// Scheduler path
				unboundClaims = append(unboundClaims, pvc)
			} else {
				// !delayBindingMode || pvc.Spec.VolumeName != ""
				// Immediate binding should have already been bound
				unboundClaimsImmediate = append(unboundClaimsImmediate, pvc)
			}
		}
	}
	return boundClaims, unboundClaims, unboundClaimsImmediate, nil
}

func (b *volumeBinder) checkBoundClaims(claims []*v1.PersistentVolumeClaim, node *v1.Node, podName string) (bool, error) {
	for _, pvc := range claims {
		pvName := pvc.Spec.VolumeName
		pv, err := b.pvCache.GetPV(pvName)
		if err != nil {
			return false, err
		}

		err = volumeutil.CheckNodeAffinity(pv, node.Labels)
		if err != nil {
			klog.V(4).Infof("PersistentVolume %q, Node %q mismatch for Pod %q: %v", pvName, node.Name, podName, err)
			return false, nil
		}
		klog.V(5).Infof("PersistentVolume %q, Node %q matches for Pod %q", pvName, node.Name, podName)
	}

	klog.V(4).Infof("All bound volumes for Pod %q match with Node %q", podName, node.Name)
	return true, nil
}

// findMatchingVolumes tries to find matching volumes for given claims,
// and return unbound claims for further provision.
func (b *volumeBinder) findMatchingVolumes(pod *v1.Pod, claimsToBind []*v1.PersistentVolumeClaim, node *v1.Node) (foundMatches bool, matchedClaims []*bindingInfo, unboundClaims []*v1.PersistentVolumeClaim, err error) {
	podName := getPodName(pod)
	// Sort all the claims by increasing size request to get the smallest fits
	sort.Sort(byPVCSize(claimsToBind))

	chosenPVs := map[string]*v1.PersistentVolume{}

	foundMatches = true
	matchedClaims = []*bindingInfo{}

	for _, pvc := range claimsToBind {
		// Get storage class name from each PVC
		storageClassName := ""
		storageClass := pvc.Spec.StorageClassName
		if storageClass != nil {
			storageClassName = *storageClass
		}
		allPVs := b.pvCache.ListPVs(storageClassName)
		pvcName := getPVCName(pvc)

		// Find a matching PV
		pv, err := findMatchingVolume(pvc, allPVs, node, chosenPVs, true)
		if err != nil {
			return false, nil, nil, err
		}
		if pv == nil {
			klog.V(4).Infof("No matching volumes for Pod %q, PVC %q on node %q", podName, pvcName, node.Name)
			unboundClaims = append(unboundClaims, pvc)
			foundMatches = false
			continue
		}

		// matching PV needs to be excluded so we don't select it again
		chosenPVs[pv.Name] = pv
		matchedClaims = append(matchedClaims, &bindingInfo{pv: pv, pvc: pvc})
		klog.V(5).Infof("Found matching PV %q for PVC %q on node %q for pod %q", pv.Name, pvcName, node.Name, podName)
	}

	if foundMatches {
		klog.V(4).Infof("Found matching volumes for pod %q on node %q", podName, node.Name)
	}

	return
}

// checkVolumeProvisions checks given unbound claims (the claims have gone through func
// findMatchingVolumes, and do not have matching volumes for binding), and return true
// if all of the claims are eligible for dynamic provision.
func (b *volumeBinder) checkVolumeProvisions(pod *v1.Pod, claimsToProvision []*v1.PersistentVolumeClaim, node *v1.Node) (provisionSatisfied bool, provisionedClaims []*v1.PersistentVolumeClaim, err error) {
	podName := getPodName(pod)
	provisionedClaims = []*v1.PersistentVolumeClaim{}

	for _, claim := range claimsToProvision {
		pvcName := getPVCName(claim)
		className := v1helper.GetPersistentVolumeClaimClass(claim)
		if className == "" {
			return false, nil, fmt.Errorf("no class for claim %q", pvcName)
		}

		class, err := b.classLister.Get(className)
		if err != nil {
			return false, nil, fmt.Errorf("failed to find storage class %q", className)
		}
		provisioner := class.Provisioner
		if provisioner == "" || provisioner == notSupportedProvisioner {
			klog.V(4).Infof("storage class %q of claim %q does not support dynamic provisioning", className, pvcName)
			return false, nil, nil
		}

		// Check if the node can satisfy the topology requirement in the class
		if !v1helper.MatchTopologySelectorTerms(class.AllowedTopologies, labels.Set(node.Labels)) {
			klog.V(4).Infof("Node %q cannot satisfy provisioning topology requirements of claim %q", node.Name, pvcName)
			return false, nil, nil
		}

		// TODO: Check if capacity of the node domain in the storage class
		// can satisfy resource requirement of given claim

		provisionedClaims = append(provisionedClaims, claim)

	}
	klog.V(4).Infof("Provisioning for claims of pod %q that has no matching volumes on node %q ...", podName, node.Name)

	return true, provisionedClaims, nil
}

func (b *volumeBinder) revertAssumedPVs(bindings []*bindingInfo) {
	for _, bindingInfo := range bindings {
		b.pvCache.Restore(bindingInfo.pv.Name)
	}
}

func (b *volumeBinder) revertAssumedPVCs(claims []*v1.PersistentVolumeClaim) {
	for _, claim := range claims {
		b.pvcCache.Restore(getPVCName(claim))
	}
}

type bindingInfo struct {
	// Claim that needs to be bound
	pvc *v1.PersistentVolumeClaim

	// Proposed PV to bind to this claim
	pv *v1.PersistentVolume
}

type byPVCSize []*v1.PersistentVolumeClaim

func (a byPVCSize) Len() int {
	return len(a)
}

func (a byPVCSize) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a byPVCSize) Less(i, j int) bool {
	iSize := a[i].Spec.Resources.Requests[v1.ResourceStorage]
	jSize := a[j].Spec.Resources.Requests[v1.ResourceStorage]
	// return true if iSize is less than jSize
	return iSize.Cmp(jSize) == -1
}

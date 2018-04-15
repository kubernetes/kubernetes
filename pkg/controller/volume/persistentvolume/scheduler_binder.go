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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
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
//    d. Cache the node selection for the Pod. (Assume phase)
//       i.  If PVC binding is required, cache in-memory only:
//           * Updated PV objects for prebinding to the corresponding PVCs.
//           * For the pod, which PVs need API updates.
//           AssumePodVolumes() is invoked here.  Then BindPodVolumes() is called asynchronously by the
//           scheduler.  After BindPodVolumes() is complete, the Pod is added back to the scheduler queue
//           to be processed again until all PVCs are bound.
//       ii. If PVC binding is not required, cache the Pod->Node binding in the scheduler's pod cache,
//           and asynchronously bind the Pod to the Node.  This is handled in the scheduler and not here.
// 2. Once the assume operation is done, the scheduler processes the next Pod in the scheduler queue
//    while the actual binding operation occurs in the background.
type SchedulerVolumeBinder interface {
	// FindPodVolumes checks if all of a Pod's PVCs can be satisfied by the node.
	//
	// If a PVC is bound, it checks if the PV's NodeAffinity matches the Node.
	// Otherwise, it tries to find an available PV to bind to the PVC.
	//
	// It returns true if there are matching PVs that can satisfy all of the Pod's PVCs, and returns true
	// if bound volumes satisfy the PV NodeAffinity.
	//
	// This function is called by the volume binding scheduler predicate and can be called in parallel
	FindPodVolumes(pod *v1.Pod, node *v1.Node) (unboundVolumesSatisified, boundVolumesSatisfied bool, err error)

	// AssumePodVolumes will take the PV matches for unbound PVCs and update the PV cache assuming
	// that the PV is prebound to the PVC.
	//
	// It returns true if all volumes are fully bound, and returns true if any volume binding API operation needs
	// to be done afterwards.
	//
	// This function will modify assumedPod with the node name.
	// This function is called serially.
	AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (allFullyBound bool, bindingRequired bool, err error)

	// BindPodVolumes will initiate the volume binding by making the API call to prebind the PV
	// to its matching PVC.
	//
	// This function can be called in parallel.
	BindPodVolumes(assumedPod *v1.Pod) error

	// GetBindingsCache returns the cache used (if any) to store volume binding decisions.
	GetBindingsCache() PodBindingCache
}

type volumeBinder struct {
	ctrl *PersistentVolumeController

	// TODO: Need AssumeCache for PVC for dynamic provisioning
	pvcCache corelisters.PersistentVolumeClaimLister
	pvCache  PVAssumeCache

	// Stores binding decisions that were made in FindPodVolumes for use in AssumePodVolumes.
	// AssumePodVolumes modifies the bindings again for use in BindPodVolumes.
	podBindingCache PodBindingCache
}

// NewVolumeBinder sets up all the caches needed for the scheduler to make volume binding decisions.
func NewVolumeBinder(
	kubeClient clientset.Interface,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	storageClassInformer storageinformers.StorageClassInformer) SchedulerVolumeBinder {

	// TODO: find better way...
	ctrl := &PersistentVolumeController{
		kubeClient:  kubeClient,
		classLister: storageClassInformer.Lister(),
	}

	b := &volumeBinder{
		ctrl:            ctrl,
		pvcCache:        pvcInformer.Lister(),
		pvCache:         NewPVAssumeCache(pvInformer.Informer()),
		podBindingCache: NewPodBindingCache(),
	}

	return b
}

func (b *volumeBinder) GetBindingsCache() PodBindingCache {
	return b.podBindingCache
}

// FindPodVolumes caches the matching PVs per node in podBindingCache
func (b *volumeBinder) FindPodVolumes(pod *v1.Pod, node *v1.Node) (unboundVolumesSatisfied, boundVolumesSatisfied bool, err error) {
	podName := getPodName(pod)

	// Warning: Below log needs high verbosity as it can be printed several times (#60933).
	glog.V(5).Infof("FindPodVolumes for pod %q, node %q", podName, node.Name)

	// Initialize to true for pods that don't have volumes
	unboundVolumesSatisfied = true
	boundVolumesSatisfied = true

	// The pod's volumes need to be processed in one call to avoid the race condition where
	// volumes can get bound in between calls.
	boundClaims, unboundClaims, unboundClaimsImmediate, err := b.getPodVolumes(pod)
	if err != nil {
		return false, false, err
	}

	// Immediate claims should be bound
	if len(unboundClaimsImmediate) > 0 {
		return false, false, fmt.Errorf("pod has unbound PersistentVolumeClaims")
	}

	// Check PV node affinity on bound volumes
	if len(boundClaims) > 0 {
		boundVolumesSatisfied, err = b.checkBoundClaims(boundClaims, node, podName)
		if err != nil {
			return false, false, err
		}
	}

	// Find PVs for unbound volumes
	if len(unboundClaims) > 0 {
		unboundVolumesSatisfied, err = b.findMatchingVolumes(pod, unboundClaims, node)
		if err != nil {
			return false, false, err
		}
	}

	return unboundVolumesSatisfied, boundVolumesSatisfied, nil
}

// AssumePodVolumes will take the cached matching PVs in podBindingCache for the chosen node
// and update the pvCache with the new prebound PV.  It will update podBindingCache again
// with the PVs that need an API update.
func (b *volumeBinder) AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (allFullyBound, bindingRequired bool, err error) {
	podName := getPodName(assumedPod)

	glog.V(4).Infof("AssumePodVolumes for pod %q, node %q", podName, nodeName)

	if allBound := b.arePodVolumesBound(assumedPod); allBound {
		glog.V(4).Infof("AssumePodVolumes: all PVCs bound and nothing to do")
		return true, false, nil
	}

	assumedPod.Spec.NodeName = nodeName
	claimsToBind := b.podBindingCache.GetBindings(assumedPod, nodeName)
	newBindings := []*bindingInfo{}

	for _, binding := range claimsToBind {
		newPV, dirty, err := b.ctrl.getBindVolumeToClaim(binding.pv, binding.pvc)
		glog.V(5).Infof("AssumePodVolumes: getBindVolumeToClaim for PV %q, PVC %q.  newPV %p, dirty %v, err: %v",
			binding.pv.Name,
			binding.pvc.Name,
			newPV,
			dirty,
			err)
		if err != nil {
			b.revertAssumedPVs(newBindings)
			return false, true, err
		}
		if dirty {
			err = b.pvCache.Assume(newPV)
			if err != nil {
				b.revertAssumedPVs(newBindings)
				return false, true, err
			}

			newBindings = append(newBindings, &bindingInfo{pv: newPV, pvc: binding.pvc})
		}
	}

	if len(newBindings) == 0 {
		// Don't update cached bindings if no API updates are needed.  This can happen if we
		// previously updated the PV object and are waiting for the PV controller to finish binding.
		glog.V(4).Infof("AssumePodVolumes: PVs already assumed")
		return false, false, nil
	}
	b.podBindingCache.UpdateBindings(assumedPod, nodeName, newBindings)

	return false, true, nil
}

// BindPodVolumes gets the cached bindings in podBindingCache and makes the API update for those PVs.
func (b *volumeBinder) BindPodVolumes(assumedPod *v1.Pod) error {
	glog.V(4).Infof("BindPodVolumes for pod %q", getPodName(assumedPod))

	bindings := b.podBindingCache.GetBindings(assumedPod, assumedPod.Spec.NodeName)

	// Do the actual prebinding. Let the PV controller take care of the rest
	// There is no API rollback if the actual binding fails
	for i, bindingInfo := range bindings {
		_, err := b.ctrl.updateBindVolumeToClaim(bindingInfo.pv, bindingInfo.pvc, false)
		if err != nil {
			// only revert assumed cached updates for volumes we haven't successfully bound
			b.revertAssumedPVs(bindings[i:])
			return err
		}
	}

	return nil
}

func getPodName(pod *v1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

func getPVCName(pvc *v1.PersistentVolumeClaim) string {
	return pvc.Namespace + "/" + pvc.Name
}

func (b *volumeBinder) isVolumeBound(namespace string, vol *v1.Volume, checkFullyBound bool) (bool, *v1.PersistentVolumeClaim, error) {
	if vol.PersistentVolumeClaim == nil {
		return true, nil, nil
	}

	pvcName := vol.PersistentVolumeClaim.ClaimName
	pvc, err := b.pvcCache.PersistentVolumeClaims(namespace).Get(pvcName)
	if err != nil || pvc == nil {
		return false, nil, fmt.Errorf("error getting PVC %q: %v", pvcName, err)
	}

	pvName := pvc.Spec.VolumeName
	if pvName != "" {
		if checkFullyBound {
			if metav1.HasAnnotation(pvc.ObjectMeta, annBindCompleted) {
				glog.V(5).Infof("PVC %q is fully bound to PV %q", getPVCName(pvc), pvName)
				return true, pvc, nil
			} else {
				glog.V(5).Infof("PVC %q is not fully bound to PV %q", getPVCName(pvc), pvName)
				return false, pvc, nil
			}
		}
		glog.V(5).Infof("PVC %q is bound or prebound to PV %q", getPVCName(pvc), pvName)
		return true, pvc, nil
	}

	glog.V(5).Infof("PVC %q is not bound", getPVCName(pvc))
	return false, pvc, nil
}

// arePodVolumesBound returns true if all volumes are fully bound
func (b *volumeBinder) arePodVolumesBound(pod *v1.Pod) bool {
	for _, vol := range pod.Spec.Volumes {
		if isBound, _, _ := b.isVolumeBound(pod.Namespace, &vol, true); !isBound {
			// Pod has at least one PVC that needs binding
			return false
		}
	}
	return true
}

// getPodVolumes returns a pod's PVCs separated into bound (including prebound), unbound with delayed binding,
// and unbound with immediate binding
func (b *volumeBinder) getPodVolumes(pod *v1.Pod) (boundClaims []*v1.PersistentVolumeClaim, unboundClaims []*bindingInfo, unboundClaimsImmediate []*v1.PersistentVolumeClaim, err error) {
	boundClaims = []*v1.PersistentVolumeClaim{}
	unboundClaimsImmediate = []*v1.PersistentVolumeClaim{}
	unboundClaims = []*bindingInfo{}

	for _, vol := range pod.Spec.Volumes {
		volumeBound, pvc, err := b.isVolumeBound(pod.Namespace, &vol, false)
		if err != nil {
			return nil, nil, nil, err
		}
		if pvc == nil {
			continue
		}
		if volumeBound {
			boundClaims = append(boundClaims, pvc)
		} else {
			delayBinding, err := b.ctrl.shouldDelayBinding(pvc)
			if err != nil {
				return nil, nil, nil, err
			}
			if delayBinding {
				// Scheduler path
				unboundClaims = append(unboundClaims, &bindingInfo{pvc: pvc})
			} else {
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
			glog.V(4).Infof("PersistentVolume %q, Node %q mismatch for Pod %q: %v", pvName, node.Name, err.Error(), podName)
			return false, nil
		}
		glog.V(5).Infof("PersistentVolume %q, Node %q matches for Pod %q", pvName, node.Name, podName)
	}

	glog.V(4).Infof("All volumes for Pod %q match with Node %q", podName, node.Name)
	return true, nil
}

func (b *volumeBinder) findMatchingVolumes(pod *v1.Pod, claimsToBind []*bindingInfo, node *v1.Node) (foundMatches bool, err error) {
	// Sort all the claims by increasing size request to get the smallest fits
	sort.Sort(byPVCSize(claimsToBind))

	chosenPVs := map[string]*v1.PersistentVolume{}

	for _, bindingInfo := range claimsToBind {
		// Get storage class name from each PVC
		storageClassName := ""
		storageClass := bindingInfo.pvc.Spec.StorageClassName
		if storageClass != nil {
			storageClassName = *storageClass
		}
		allPVs := b.pvCache.ListPVs(storageClassName)

		// Find a matching PV
		bindingInfo.pv, err = findMatchingVolume(bindingInfo.pvc, allPVs, node, chosenPVs, true)
		if err != nil {
			return false, err
		}
		if bindingInfo.pv == nil {
			glog.V(4).Infof("No matching volumes for PVC %q on node %q", getPVCName(bindingInfo.pvc), node.Name)
			return false, nil
		}

		// matching PV needs to be excluded so we don't select it again
		chosenPVs[bindingInfo.pv.Name] = bindingInfo.pv
	}

	// Mark cache with all the matches for each PVC for this node
	b.podBindingCache.UpdateBindings(pod, node.Name, claimsToBind)
	glog.V(4).Infof("Found matching volumes on node %q", node.Name)

	return true, nil
}

func (b *volumeBinder) revertAssumedPVs(bindings []*bindingInfo) {
	for _, bindingInfo := range bindings {
		b.pvCache.Restore(bindingInfo.pv.Name)
	}
}

type bindingInfo struct {
	// Claim that needs to be bound
	pvc *v1.PersistentVolumeClaim

	// Proposed PV to bind to this claim
	pv *v1.PersistentVolume
}

// Used in unit test errors
func (b bindingInfo) String() string {
	pvcName := ""
	pvName := ""
	if b.pvc != nil {
		pvcName = getPVCName(b.pvc)
	}
	if b.pv != nil {
		pvName = b.pv.Name
	}
	return fmt.Sprintf("[PVC %q, PV %q]", pvcName, pvName)
}

type byPVCSize []*bindingInfo

func (a byPVCSize) Len() int {
	return len(a)
}

func (a byPVCSize) Swap(i, j int) {
	a[i], a[j] = a[j], a[i]
}

func (a byPVCSize) Less(i, j int) bool {
	iSize := a[i].pvc.Spec.Resources.Requests[v1.ResourceStorage]
	jSize := a[j].pvc.Spec.Resources.Requests[v1.ResourceStorage]
	// return true if iSize is less than jSize
	return iSize.Cmp(jSize) == -1
}

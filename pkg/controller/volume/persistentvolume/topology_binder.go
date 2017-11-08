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
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	storageinformers "k8s.io/client-go/informers/storage/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

// Used in AssumePodVolumes to temporarily mark in the Pod which PVs need binding.
const annPVsToBind string = "pv.kubernetes.io/pvsToBind"

// TopologyAwareVolumeBinder is used by the scheduler to handle PVC/PV binding
// and dynamic provisioning.  The binding decisions are integrated into the pod scheduling
// workflow so that the topology of volumes are also considered along with the pod's other
// scheduling requirements.
type TopologyAwareVolumeBinder interface {
	// FindPodVolumes checks if a Pod's PVCs can be satisfied by the node.
	//
	// If a PVC is bound, it checks if the PV's NodeAffinity matches the Node.
	// Otherwise, it tries to find an available PV to bind to the PVC.
	//
	// It returns true if any of the PVCs need to be bound, and true if there are matching
	// PVs that can satisfy all of the Pod's PVCs.
	FindPodVolumes(pod *v1.Pod, nodeName string) (needsBinding, foundPVs bool, err error)

	// AssumePodVolumes will take the PV matches for unbound PVCs and update the PV cache assuming
	// that the PV is prebound to the PVC.
	//
	// It returns true if any volume binding API operation needs to be done afterwards.
	//
	// This function will modify assumedPod temporarily so assumedPod should not be a shared pointer.
	AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (bindingRequired bool, err error)

	// BindPodVolumes will initiate the volume binding by making the API call to prebind the PV
	// to its matching PVC.
	//
	// It returns true if any volume binding operation needs to be completed.
	//
	// This function will modify assumedPod temporarily so assumedPod should not be a shared pointer.
	BindPodVolumes(assumedPod *v1.Pod) (bindingRequired bool, err error)

	// InitTmpData needs to be called at the beginning of processing a new Pod to initialize temporary
	// state for the current Pod and clear any temporary data that was cached for the previous Pod
	InitTmpData(pod *v1.Pod)
}

type topologyVolumeBinder struct {
	ctrl *PersistentVolumeController
	// TODO: Need TmpCache for PVC for dynamic provisioning
	pvcCache  corelisters.PersistentVolumeClaimLister
	nodeCache corelisters.NodeLister
	pvCache   PVTmpCache

	// tmpData is not re-entrant and has to be cleared at the beginning of processing a new Pod
	tmpData *tmpData
}

// NewTopologyAwareVolumeBinder sets up all the caches needed for the scheduler to make
// topology-aware volume binding decisions.
func NewTopologyAwareVolumeBinder(
	kubeClient clientset.Interface,
	pvcInformer coreinformers.PersistentVolumeClaimInformer,
	pvInformer coreinformers.PersistentVolumeInformer,
	nodeInformer coreinformers.NodeInformer,
	storageClassInformer storageinformers.StorageClassInformer) TopologyAwareVolumeBinder {

	// TODO: find better way...
	ctrl := &PersistentVolumeController{
		kubeClient:  kubeClient,
		classLister: storageClassInformer.Lister(),
	}

	b := &topologyVolumeBinder{
		ctrl:      ctrl,
		pvcCache:  pvcInformer.Lister(),
		nodeCache: nodeInformer.Lister(),
		pvCache:   newPVTmpCache(pvInformer.Informer()),
		tmpData:   newTmpData(),
	}

	return b
}

func getPodName(pod *v1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

func getPVCName(pvc *v1.PersistentVolumeClaim) string {
	return pvc.Namespace + "/" + pvc.Name
}

func (b *topologyVolumeBinder) isVolumeBound(namespace string, vol *v1.Volume, checkFullyBound bool) (bool, *v1.PersistentVolumeClaim, error) {
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

	glog.V(5).Infof("PVC %v/%v is not bound", namespace, pvc.Name)
	return false, pvc, nil
}

// arePodVolumesBound returns true if all volumes are fully bound
func (b *topologyVolumeBinder) arePodVolumesBound(pod *v1.Pod) bool {
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
func (b *topologyVolumeBinder) getPodVolumes(pod *v1.Pod) (boundClaims []*v1.PersistentVolumeClaim, unboundClaims []*bindingInfo, unboundClaimsImmediate []*v1.PersistentVolumeClaim, err error) {

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
			delayBinding := b.ctrl.shouldDelayBinding(pvc)
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

// FindPodVolumes internally caches the matching PVs per node in tmpData.nodeBindings
func (b *topologyVolumeBinder) FindPodVolumes(pod *v1.Pod, nodeName string) (needsBinding bool, foundMatches bool, err error) {
	glog.V(4).Infof("FindVolumesToBind for pod %q, node %q", getPodName(pod), nodeName)

	node, err := b.nodeCache.Get(nodeName)
	if node == nil || err != nil {
		return false, false, fmt.Errorf("error getting node %q: %v", nodeName, err)
	}

	boundClaims, unboundClaims, unboundClaimsImmediate, err := b.getPodVolumes(pod)
	if err != nil {
		return false, false, err
	}

	// Immediate claims should be bound
	if len(unboundClaimsImmediate) > 0 {
		return true, false, fmt.Errorf("pod has unbound PersistentVolumeClaims")
	}

	// Check PV node affinity on bound volumes
	if len(boundClaims) > 0 {
		affinityMatches, err := b.checkBoundClaims(boundClaims, node, getPodName(pod))
		if err != nil || !affinityMatches {
			return false, affinityMatches, err
		}
	}

	// Find PVs for unbound volumes
	if len(unboundClaims) > 0 {
		foundMatches, err := b.findMatchingVolumes(unboundClaims, node)
		return true, foundMatches, err
	}

	// Pod has no PVC volumes, return success
	return false, true, nil
}

func (b *topologyVolumeBinder) checkBoundClaims(claims []*v1.PersistentVolumeClaim, node *v1.Node, podName string) (bool, error) {
	for _, pvc := range claims {
		pvName := pvc.Spec.VolumeName
		pv := b.pvCache.GetPV(pvName)
		if pv == nil {
			return false, fmt.Errorf("PersistentVolume %q not found", pvName)
		}

		err := volumeutil.CheckNodeAffinity(pv, node.Labels)
		if err != nil {
			glog.V(4).Infof("PersistentVolume %q, Node %q mismatch for Pod %q: %v", pvName, node.Name, err.Error(), podName)
			return false, nil
		}
		glog.V(5).Infof("PersistentVolume %q, Node %q matches for Pod %q", pvName, node.Name, podName)
	}

	glog.V(4).Infof("All volumes for Pod %q match with Node %q", podName, node.Name)
	return true, nil
}

func (b *topologyVolumeBinder) findMatchingVolumes(claimsToBind []*bindingInfo, node *v1.Node) (foundMatches bool, err error) {
	// Sort all the claims by increasing size request to get the smallest fits
	sort.Sort(byPVCSize(claimsToBind))

	allPVs := b.pvCache.ListPVs()
	chosenPVs := map[string]*v1.PersistentVolume{}

	for _, bindingInfo := range claimsToBind {
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
	b.tmpData.nodeBindings[node.Name] = claimsToBind
	glog.V(4).Infof("Found matching volumes on node %q", node.Name)

	return true, nil
}

// TODO: think about the situation where multiple pods share the same PVC
// one bind routine could revert a PV that another routine is in the middle of binding
// maybe instead of a go routine per pod, assume should queue up all the pods to one
// binder routine
func (b *topologyVolumeBinder) revertAssumedPVs(pvs []string) {
	for _, pvName := range pvs {
		b.pvCache.Restore(pvName)
	}
}

// AssumePodVolumes will take the cached matching PVs in tmpData.nodeBindings for the chosen node
// and update the PV cache with the new prebound PV.  It will set the pvsToBind annotation in
// assumedPod with a list of all the PV names that need an API update.
func (b *topologyVolumeBinder) AssumePodVolumes(assumedPod *v1.Pod, nodeName string) (bool, error) {
	glog.V(4).Infof("AssumePodVolumes for pod %q, node %q", getPodName(assumedPod), nodeName)

	claimsToBind, _ := b.tmpData.nodeBindings[nodeName]
	if len(claimsToBind) == 0 {
		if !b.tmpData.allPVCsBound {
			// Set annotation even if pvsToBind is empty because it means that we haven't gone
			// through a full scheduler cycle where all the PVCs were fully bound.
			//
			// This can also happen if the volume binding predicate succeeded and was stored in
			// the equivalence cache, but other predicates failed, causing the Pod to go through
			// another scheduling cycle. Then tmpData is cleared, but the volume binding predicate
			// is not called again since it's in the equivalence cache.  This is ok because the
			// volume binding predicate will be invalidated after this.
			//
			// TODO: can this be optimized so that we don't have to spawn a go routine for binding
			// that will exit immediately?
			glog.V(4).Infof("AssumePodVolumes: all PVCs bound, but not since the beginning of scheduling this pod")
			setPVsToBind(assumedPod, nil)
			return true, nil
		}
		glog.V(4).Infof("AssumePodVolumes: all PVCs bound and nothing to do")
		return false, nil
	}

	pvsToBind := []string{}
	for _, bindingInfo := range claimsToBind {
		newPV, dirty, err := b.ctrl.getBindVolumeToClaim(bindingInfo.pv, bindingInfo.pvc)
		glog.V(5).Infof("AssumePodVolumes: getBindVolumeToClaim for PV %q, PVC %q.  newPV %p, dirty %v, err: %v",
			bindingInfo.pv.Name,
			bindingInfo.pvc.Name,
			newPV,
			dirty,
			err)
		if err != nil {
			b.revertAssumedPVs(pvsToBind)
			return true, err
		}
		if dirty {
			err = b.pvCache.TmpUpdate(newPV)
			if err != nil {
				b.revertAssumedPVs(pvsToBind)
				return true, err
			}

			pvsToBind = append(pvsToBind, bindingInfo.pv.Name)
		}
	}

	// Set annotation even if pvsToBind is empty because it means that we haven't gone
	// through a full scheduler cycle where all the PVCs were fully bound
	// TODO: can this be optimized so that we don't have to spawn a go routine for binding
	// that will exit immediately?
	setPVsToBind(assumedPod, pvsToBind)

	return true, nil
}

func setPVsToBind(assumedPod *v1.Pod, pvsToBind []string) {
	// Set annotation in assumed Pod object
	if assumedPod.Annotations == nil {
		assumedPod.Annotations = map[string]string{}
	}
	assumedPod.Annotations[annPVsToBind] = strings.Join(pvsToBind, ",")
}

func removePVsToBind(assumedPod *v1.Pod) {
	delete(assumedPod.Annotations, annPVsToBind)
}

func getPVsToBind(assumedPod *v1.Pod) ([]string, bool) {
	pvsToBindStr, ok := assumedPod.Annotations["pv.kubernetes.io/pvsToBind"]
	if len(pvsToBindStr) == 0 {
		return nil, ok
	}
	return strings.Split(pvsToBindStr, ","), ok
}

// BindPodVolumes checks the assumedPod's pvsToBind annotation and makes the API
// update for those PVs.
func (b *topologyVolumeBinder) BindPodVolumes(assumedPod *v1.Pod) (bool, error) {
	glog.V(4).Infof("BindPodVolumes for pod %q", getPodName(assumedPod))

	var err error

	defer removePVsToBind(assumedPod)

	pvsToBind, ok := getPVsToBind(assumedPod)
	if !ok {
		glog.V(4).Infof("BindPodVolumes: binding not required for pod %q", getPodName(assumedPod))
		return false, nil
	}
	if pvsToBind == nil {
		glog.V(4).Infof("BindPodVolumes: binding required but no PV updates needed for pod %q", getPodName(assumedPod))
		return true, nil
	}

	bindings := []*bindingInfo{}
	for _, pvName := range pvsToBind {
		binding := &bindingInfo{}
		binding.pv = b.pvCache.GetPV(pvName)
		if binding.pv == nil {
			b.revertAssumedPVs(pvsToBind)
			return true, fmt.Errorf("couldn't get PV %q", pvName)
		}
		claimRef := binding.pv.Spec.ClaimRef
		if claimRef == nil {
			b.revertAssumedPVs(pvsToBind)
			return true, fmt.Errorf("claimRef is nil for PV %q", pvName)
		}
		pvcName := claimRef.Namespace + "/" + claimRef.Name
		binding.pvc, err = b.pvcCache.PersistentVolumeClaims(claimRef.Namespace).Get(claimRef.Name)
		if err != nil || binding.pvc == nil {
			b.revertAssumedPVs(pvsToBind)
			return true, fmt.Errorf("couldn't get PVC %q", pvcName)
		}
		bindings = append(bindings, binding)
	}

	// Do the actual prebinding. Let the PV controller take care of the rest
	// There is no API rollback if the actual binding fails
	for i, bindingInfo := range bindings {
		_, err := b.ctrl.updateBindVolumeToClaim(bindingInfo.pv, bindingInfo.pvc, false)
		if err != nil {
			// only revert assumed cached updates for volumes we haven't successfully bound
			// this assumes pvsToBind and bindings are the same length and same order
			b.revertAssumedPVs(pvsToBind[i:])
			return true, err
		}
	}

	return true, nil
}

type bindingInfo struct {
	// Claim that needs to be bound
	pvc *v1.PersistentVolumeClaim

	// Proposed PV to bind to this claim
	pv *v1.PersistentVolume
}

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

// tmpInfo has to be cleared at the start of every pod scheduling loop
type tmpData struct {
	// The Pod that is currently being processed
	podName string

	// True if all PVCs are bound at the start
	allPVCsBound bool

	// Key = nodeName
	// Value = array of bindingInfo
	nodeBindings map[string][]*bindingInfo
}

func newTmpData() *tmpData {
	return &tmpData{nodeBindings: map[string][]*bindingInfo{}}
}

func (b *topologyVolumeBinder) InitTmpData(pod *v1.Pod) {
	b.tmpData = newTmpData()
	b.tmpData.podName = pod.Namespace + "/" + pod.Name
	b.tmpData.allPVCsBound = b.arePodVolumesBound(pod)
}

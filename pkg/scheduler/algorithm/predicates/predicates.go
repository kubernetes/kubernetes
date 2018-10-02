/*
Copyright 2014 The Kubernetes Authors.

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

package predicates

import (
	"errors"
	"fmt"
	"os"
	"regexp"
	"strconv"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	priorityutil "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities/util"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// MatchInterPodAffinityPred defines the name of predicate MatchInterPodAffinity.
	MatchInterPodAffinityPred = "MatchInterPodAffinity"
	// CheckVolumeBindingPred defines the name of predicate CheckVolumeBinding.
	CheckVolumeBindingPred = "CheckVolumeBinding"
	// CheckNodeConditionPred defines the name of predicate CheckNodeCondition.
	CheckNodeConditionPred = "CheckNodeCondition"
	// GeneralPred defines the name of predicate GeneralPredicates.
	GeneralPred = "GeneralPredicates"
	// HostNamePred defines the name of predicate HostName.
	HostNamePred = "HostName"
	// PodFitsHostPortsPred defines the name of predicate PodFitsHostPorts.
	PodFitsHostPortsPred = "PodFitsHostPorts"
	// MatchNodeSelectorPred defines the name of predicate MatchNodeSelector.
	MatchNodeSelectorPred = "MatchNodeSelector"
	// PodFitsResourcesPred defines the name of predicate PodFitsResources.
	PodFitsResourcesPred = "PodFitsResources"
	// NoDiskConflictPred defines the name of predicate NoDiskConflict.
	NoDiskConflictPred = "NoDiskConflict"
	// PodToleratesNodeTaintsPred defines the name of predicate PodToleratesNodeTaints.
	PodToleratesNodeTaintsPred = "PodToleratesNodeTaints"
	// CheckNodeUnschedulablePred defines the name of predicate CheckNodeUnschedulablePredicate.
	CheckNodeUnschedulablePred = "CheckNodeUnschedulable"
	// PodToleratesNodeNoExecuteTaintsPred defines the name of predicate PodToleratesNodeNoExecuteTaints.
	PodToleratesNodeNoExecuteTaintsPred = "PodToleratesNodeNoExecuteTaints"
	// CheckNodeLabelPresencePred defines the name of predicate CheckNodeLabelPresence.
	CheckNodeLabelPresencePred = "CheckNodeLabelPresence"
	// CheckServiceAffinityPred defines the name of predicate checkServiceAffinity.
	CheckServiceAffinityPred = "CheckServiceAffinity"
	// MaxEBSVolumeCountPred defines the name of predicate MaxEBSVolumeCount.
	MaxEBSVolumeCountPred = "MaxEBSVolumeCount"
	// MaxGCEPDVolumeCountPred defines the name of predicate MaxGCEPDVolumeCount.
	MaxGCEPDVolumeCountPred = "MaxGCEPDVolumeCount"
	// MaxAzureDiskVolumeCountPred defines the name of predicate MaxAzureDiskVolumeCount.
	MaxAzureDiskVolumeCountPred = "MaxAzureDiskVolumeCount"
	// MaxCSIVolumeCountPred defines the predicate that decides how many CSI volumes should be attached
	MaxCSIVolumeCountPred = "MaxCSIVolumeCountPred"
	// NoVolumeZoneConflictPred defines the name of predicate NoVolumeZoneConflict.
	NoVolumeZoneConflictPred = "NoVolumeZoneConflict"
	// CheckNodeMemoryPressurePred defines the name of predicate CheckNodeMemoryPressure.
	CheckNodeMemoryPressurePred = "CheckNodeMemoryPressure"
	// CheckNodeDiskPressurePred defines the name of predicate CheckNodeDiskPressure.
	CheckNodeDiskPressurePred = "CheckNodeDiskPressure"
	// CheckNodePIDPressurePred defines the name of predicate CheckNodePIDPressure.
	CheckNodePIDPressurePred = "CheckNodePIDPressure"

	// DefaultMaxGCEPDVolumes defines the maximum number of PD Volumes for GCE
	// GCE instances can have up to 16 PD volumes attached.
	DefaultMaxGCEPDVolumes = 16
	// DefaultMaxAzureDiskVolumes defines the maximum number of PD Volumes for Azure
	// Larger Azure VMs can actually have much more disks attached.
	// TODO We should determine the max based on VM size
	DefaultMaxAzureDiskVolumes = 16

	// KubeMaxPDVols defines the maximum number of PD Volumes per kubelet
	KubeMaxPDVols = "KUBE_MAX_PD_VOLS"

	// EBSVolumeFilterType defines the filter name for EBSVolumeFilter.
	EBSVolumeFilterType = "EBS"
	// GCEPDVolumeFilterType defines the filter name for GCEPDVolumeFilter.
	GCEPDVolumeFilterType = "GCE"
	// AzureDiskVolumeFilterType defines the filter name for AzureDiskVolumeFilter.
	AzureDiskVolumeFilterType = "AzureDisk"
)

// IMPORTANT NOTE for predicate developers:
// We are using cached predicate result for pods belonging to the same equivalence class.
// So when updating an existing predicate, you should consider whether your change will introduce new
// dependency to attributes of any API object like Pod, Node, Service etc.
// If yes, you are expected to invalidate the cached predicate result for related API object change.
// For example:
// https://github.com/kubernetes/kubernetes/blob/36a218e/plugin/pkg/scheduler/factory/factory.go#L422

// IMPORTANT NOTE: this list contains the ordering of the predicates, if you develop a new predicate
// it is mandatory to add its name to this list.
// Otherwise it won't be processed, see generic_scheduler#podFitsOnNode().
// The order is based on the restrictiveness & complexity of predicates.
// Design doc: https://github.com/kubernetes/community/blob/master/contributors/design-proposals/scheduling/predicates-ordering.md
var (
	predicatesOrdering = []string{CheckNodeConditionPred, CheckNodeUnschedulablePred,
		GeneralPred, HostNamePred, PodFitsHostPortsPred,
		MatchNodeSelectorPred, PodFitsResourcesPred, NoDiskConflictPred,
		PodToleratesNodeTaintsPred, PodToleratesNodeNoExecuteTaintsPred, CheckNodeLabelPresencePred,
		CheckServiceAffinityPred, MaxEBSVolumeCountPred, MaxGCEPDVolumeCountPred, MaxCSIVolumeCountPred,
		MaxAzureDiskVolumeCountPred, CheckVolumeBindingPred, NoVolumeZoneConflictPred,
		CheckNodeMemoryPressurePred, CheckNodePIDPressurePred, CheckNodeDiskPressurePred, MatchInterPodAffinityPred}
)

// NodeInfo interface represents anything that can get node object from node ID.
type NodeInfo interface {
	GetNodeInfo(nodeID string) (*v1.Node, error)
}

// PersistentVolumeInfo interface represents anything that can get persistent volume object by PV ID.
type PersistentVolumeInfo interface {
	GetPersistentVolumeInfo(pvID string) (*v1.PersistentVolume, error)
}

// CachedPersistentVolumeInfo implements PersistentVolumeInfo
type CachedPersistentVolumeInfo struct {
	corelisters.PersistentVolumeLister
}

// Ordering returns the ordering of predicates.
func Ordering() []string {
	return predicatesOrdering
}

// SetPredicatesOrdering sets the ordering of predicates.
func SetPredicatesOrdering(names []string) {
	predicatesOrdering = names
}

// GetPersistentVolumeInfo returns a persistent volume object by PV ID.
func (c *CachedPersistentVolumeInfo) GetPersistentVolumeInfo(pvID string) (*v1.PersistentVolume, error) {
	return c.Get(pvID)
}

// PersistentVolumeClaimInfo interface represents anything that can get a PVC object in
// specified namespace with specified name.
type PersistentVolumeClaimInfo interface {
	GetPersistentVolumeClaimInfo(namespace string, name string) (*v1.PersistentVolumeClaim, error)
}

// CachedPersistentVolumeClaimInfo implements PersistentVolumeClaimInfo
type CachedPersistentVolumeClaimInfo struct {
	corelisters.PersistentVolumeClaimLister
}

// GetPersistentVolumeClaimInfo fetches the claim in specified namespace with specified name
func (c *CachedPersistentVolumeClaimInfo) GetPersistentVolumeClaimInfo(namespace string, name string) (*v1.PersistentVolumeClaim, error) {
	return c.PersistentVolumeClaims(namespace).Get(name)
}

// CachedNodeInfo implements NodeInfo
type CachedNodeInfo struct {
	corelisters.NodeLister
}

// GetNodeInfo returns cached data for the node 'id'.
func (c *CachedNodeInfo) GetNodeInfo(id string) (*v1.Node, error) {
	node, err := c.Get(id)

	if apierrors.IsNotFound(err) {
		return nil, err
	}

	if err != nil {
		return nil, fmt.Errorf("error retrieving node '%v' from cache: %v", id, err)
	}

	return node, nil
}

// StorageClassInfo interface represents anything that can get a storage class object by class name.
type StorageClassInfo interface {
	GetStorageClassInfo(className string) (*storagev1.StorageClass, error)
}

// CachedStorageClassInfo implements StorageClassInfo
type CachedStorageClassInfo struct {
	storagelisters.StorageClassLister
}

// GetStorageClassInfo get StorageClass by class name.
func (c *CachedStorageClassInfo) GetStorageClassInfo(className string) (*storagev1.StorageClass, error) {
	return c.Get(className)
}

func isVolumeConflict(volume v1.Volume, pod *v1.Pod) bool {
	// fast path if there is no conflict checking targets.
	if volume.GCEPersistentDisk == nil && volume.AWSElasticBlockStore == nil && volume.RBD == nil && volume.ISCSI == nil {
		return false
	}

	for _, existingVolume := range pod.Spec.Volumes {
		// Same GCE disk mounted by multiple pods conflicts unless all pods mount it read-only.
		if volume.GCEPersistentDisk != nil && existingVolume.GCEPersistentDisk != nil {
			disk, existingDisk := volume.GCEPersistentDisk, existingVolume.GCEPersistentDisk
			if disk.PDName == existingDisk.PDName && !(disk.ReadOnly && existingDisk.ReadOnly) {
				return true
			}
		}

		if volume.AWSElasticBlockStore != nil && existingVolume.AWSElasticBlockStore != nil {
			if volume.AWSElasticBlockStore.VolumeID == existingVolume.AWSElasticBlockStore.VolumeID {
				return true
			}
		}

		if volume.ISCSI != nil && existingVolume.ISCSI != nil {
			iqn := volume.ISCSI.IQN
			eiqn := existingVolume.ISCSI.IQN
			// two ISCSI volumes are same, if they share the same iqn. As iscsi volumes are of type
			// RWO or ROX, we could permit only one RW mount. Same iscsi volume mounted by multiple Pods
			// conflict unless all other pods mount as read only.
			if iqn == eiqn && !(volume.ISCSI.ReadOnly && existingVolume.ISCSI.ReadOnly) {
				return true
			}
		}

		if volume.RBD != nil && existingVolume.RBD != nil {
			mon, pool, image := volume.RBD.CephMonitors, volume.RBD.RBDPool, volume.RBD.RBDImage
			emon, epool, eimage := existingVolume.RBD.CephMonitors, existingVolume.RBD.RBDPool, existingVolume.RBD.RBDImage
			// two RBDs images are the same if they share the same Ceph monitor, are in the same RADOS Pool, and have the same image name
			// only one read-write mount is permitted for the same RBD image.
			// same RBD image mounted by multiple Pods conflicts unless all Pods mount the image read-only
			if haveOverlap(mon, emon) && pool == epool && image == eimage && !(volume.RBD.ReadOnly && existingVolume.RBD.ReadOnly) {
				return true
			}
		}
	}

	return false
}

// NoDiskConflict evaluates if a pod can fit due to the volumes it requests, and those that
// are already mounted. If there is already a volume mounted on that node, another pod that uses the same volume
// can't be scheduled there.
// This is GCE, Amazon EBS, and Ceph RBD specific for now:
// - GCE PD allows multiple mounts as long as they're all read-only
// - AWS EBS forbids any two pods mounting the same volume ID
// - Ceph RBD forbids if any two pods share at least same monitor, and match pool and image.
// - ISCSI forbids if any two pods share at least same IQN, LUN and Target
// TODO: migrate this into some per-volume specific code?
func NoDiskConflict(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	for _, v := range pod.Spec.Volumes {
		for _, ev := range nodeInfo.Pods() {
			if isVolumeConflict(v, ev) {
				return false, []algorithm.PredicateFailureReason{ErrDiskConflict}, nil
			}
		}
	}
	return true, nil, nil
}

// MaxPDVolumeCountChecker contains information to check the max number of volumes for a predicate.
type MaxPDVolumeCountChecker struct {
	filter         VolumeFilter
	volumeLimitKey v1.ResourceName
	maxVolumeFunc  func(node *v1.Node) int
	pvInfo         PersistentVolumeInfo
	pvcInfo        PersistentVolumeClaimInfo

	// The string below is generated randomly during the struct's initialization.
	// It is used to prefix volumeID generated inside the predicate() method to
	// avoid conflicts with any real volume.
	randomVolumeIDPrefix string
}

// VolumeFilter contains information on how to filter PD Volumes when checking PD Volume caps
type VolumeFilter struct {
	// Filter normal volumes
	FilterVolume           func(vol *v1.Volume) (id string, relevant bool)
	FilterPersistentVolume func(pv *v1.PersistentVolume) (id string, relevant bool)
}

// NewMaxPDVolumeCountPredicate creates a predicate which evaluates whether a pod can fit based on the
// number of volumes which match a filter that it requests, and those that are already present.
//
// The predicate looks for both volumes used directly, as well as PVC volumes that are backed by relevant volume
// types, counts the number of unique volumes, and rejects the new pod if it would place the total count over
// the maximum.
func NewMaxPDVolumeCountPredicate(
	filterName string, pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo) algorithm.FitPredicate {
	var filter VolumeFilter
	var volumeLimitKey v1.ResourceName

	switch filterName {

	case EBSVolumeFilterType:
		filter = EBSVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.EBSVolumeLimitKey)
	case GCEPDVolumeFilterType:
		filter = GCEPDVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.GCEVolumeLimitKey)
	case AzureDiskVolumeFilterType:
		filter = AzureDiskVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.AzureVolumeLimitKey)
	default:
		glog.Fatalf("Wrong filterName, Only Support %v %v %v ", EBSVolumeFilterType,
			GCEPDVolumeFilterType, AzureDiskVolumeFilterType)
		return nil

	}
	c := &MaxPDVolumeCountChecker{
		filter:               filter,
		volumeLimitKey:       volumeLimitKey,
		maxVolumeFunc:        getMaxVolumeFunc(filterName),
		pvInfo:               pvInfo,
		pvcInfo:              pvcInfo,
		randomVolumeIDPrefix: rand.String(32),
	}

	return c.predicate
}

func getMaxVolumeFunc(filterName string) func(node *v1.Node) int {
	return func(node *v1.Node) int {
		maxVolumesFromEnv := getMaxVolLimitFromEnv()
		if maxVolumesFromEnv > 0 {
			return maxVolumesFromEnv
		}

		var nodeInstanceType string
		for k, v := range node.ObjectMeta.Labels {
			if k == kubeletapis.LabelInstanceType {
				nodeInstanceType = v
			}
		}
		switch filterName {
		case EBSVolumeFilterType:
			return getMaxEBSVolume(nodeInstanceType)
		case GCEPDVolumeFilterType:
			return DefaultMaxGCEPDVolumes
		case AzureDiskVolumeFilterType:
			return DefaultMaxAzureDiskVolumes
		default:
			return -1
		}
	}
}

func getMaxEBSVolume(nodeInstanceType string) int {
	if ok, _ := regexp.MatchString(volumeutil.EBSNitroLimitRegex, nodeInstanceType); ok {
		return volumeutil.DefaultMaxEBSNitroVolumeLimit
	}
	return volumeutil.DefaultMaxEBSVolumes
}

// getMaxVolLimitFromEnv checks the max PD volumes environment variable, otherwise returning a default value
func getMaxVolLimitFromEnv() int {
	if rawMaxVols := os.Getenv(KubeMaxPDVols); rawMaxVols != "" {
		if parsedMaxVols, err := strconv.Atoi(rawMaxVols); err != nil {
			glog.Errorf("Unable to parse maximum PD volumes value, using default: %v", err)
		} else if parsedMaxVols <= 0 {
			glog.Errorf("Maximum PD volumes must be a positive value, using default ")
		} else {
			return parsedMaxVols
		}
	}

	return -1
}

func (c *MaxPDVolumeCountChecker) filterVolumes(volumes []v1.Volume, namespace string, filteredVolumes map[string]bool) error {
	for i := range volumes {
		vol := &volumes[i]
		if id, ok := c.filter.FilterVolume(vol); ok {
			filteredVolumes[id] = true
		} else if vol.PersistentVolumeClaim != nil {
			pvcName := vol.PersistentVolumeClaim.ClaimName
			if pvcName == "" {
				return fmt.Errorf("PersistentVolumeClaim had no name")
			}

			// Until we know real ID of the volume use namespace/pvcName as substitute
			// with a random prefix (calculated and stored inside 'c' during initialization)
			// to avoid conflicts with existing volume IDs.
			pvID := fmt.Sprintf("%s-%s/%s", c.randomVolumeIDPrefix, namespace, pvcName)

			pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
			if err != nil || pvc == nil {
				// if the PVC is not found, log the error and count the PV towards the PV limit
				glog.V(4).Infof("Unable to look up PVC info for %s/%s, assuming PVC matches predicate when counting limits: %v", namespace, pvcName, err)
				filteredVolumes[pvID] = true
				continue
			}

			pvName := pvc.Spec.VolumeName
			if pvName == "" {
				// PVC is not bound. It was either deleted and created again or
				// it was forcefully unbound by admin. The pod can still use the
				// original PV where it was bound to -> log the error and count
				// the PV towards the PV limit
				glog.V(4).Infof("PVC %s/%s is not bound, assuming PVC matches predicate when counting limits", namespace, pvcName)
				filteredVolumes[pvID] = true
				continue
			}

			pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
			if err != nil || pv == nil {
				// if the PV is not found, log the error
				// and count the PV towards the PV limit
				glog.V(4).Infof("Unable to look up PV info for %s/%s/%s, assuming PV matches predicate when counting limits: %v", namespace, pvcName, pvName, err)
				filteredVolumes[pvID] = true
				continue
			}

			if id, ok := c.filter.FilterPersistentVolume(pv); ok {
				filteredVolumes[id] = true
			}
		}
	}

	return nil
}

func (c *MaxPDVolumeCountChecker) predicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	newVolumes := make(map[string]bool)
	if err := c.filterVolumes(pod.Spec.Volumes, pod.Namespace, newVolumes); err != nil {
		return false, nil, err
	}

	// quick return
	if len(newVolumes) == 0 {
		return true, nil, nil
	}

	// count unique volumes
	existingVolumes := make(map[string]bool)
	for _, existingPod := range nodeInfo.Pods() {
		if err := c.filterVolumes(existingPod.Spec.Volumes, existingPod.Namespace, existingVolumes); err != nil {
			return false, nil, err
		}
	}
	numExistingVolumes := len(existingVolumes)

	// filter out already-mounted volumes
	for k := range existingVolumes {
		if _, ok := newVolumes[k]; ok {
			delete(newVolumes, k)
		}
	}

	numNewVolumes := len(newVolumes)
	maxAttachLimit := c.maxVolumeFunc(nodeInfo.Node())

	if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		volumeLimits := nodeInfo.VolumeLimits()
		if maxAttachLimitFromAllocatable, ok := volumeLimits[c.volumeLimitKey]; ok {
			maxAttachLimit = int(maxAttachLimitFromAllocatable)
		}
	}

	if numExistingVolumes+numNewVolumes > maxAttachLimit {
		// violates MaxEBSVolumeCount or MaxGCEPDVolumeCount
		return false, []algorithm.PredicateFailureReason{ErrMaxVolumeCountExceeded}, nil
	}
	if nodeInfo != nil && nodeInfo.TransientInfo != nil && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) {
		nodeInfo.TransientInfo.TransientLock.Lock()
		defer nodeInfo.TransientInfo.TransientLock.Unlock()
		nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount = maxAttachLimit - numExistingVolumes
		nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes = numNewVolumes
	}
	return true, nil, nil
}

// EBSVolumeFilter is a VolumeFilter for filtering AWS ElasticBlockStore Volumes
var EBSVolumeFilter = VolumeFilter{
	FilterVolume: func(vol *v1.Volume) (string, bool) {
		if vol.AWSElasticBlockStore != nil {
			return vol.AWSElasticBlockStore.VolumeID, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *v1.PersistentVolume) (string, bool) {
		if pv.Spec.AWSElasticBlockStore != nil {
			return pv.Spec.AWSElasticBlockStore.VolumeID, true
		}
		return "", false
	},
}

// GCEPDVolumeFilter is a VolumeFilter for filtering GCE PersistentDisk Volumes
var GCEPDVolumeFilter = VolumeFilter{
	FilterVolume: func(vol *v1.Volume) (string, bool) {
		if vol.GCEPersistentDisk != nil {
			return vol.GCEPersistentDisk.PDName, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *v1.PersistentVolume) (string, bool) {
		if pv.Spec.GCEPersistentDisk != nil {
			return pv.Spec.GCEPersistentDisk.PDName, true
		}
		return "", false
	},
}

// AzureDiskVolumeFilter is a VolumeFilter for filtering Azure Disk Volumes
var AzureDiskVolumeFilter = VolumeFilter{
	FilterVolume: func(vol *v1.Volume) (string, bool) {
		if vol.AzureDisk != nil {
			return vol.AzureDisk.DiskName, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *v1.PersistentVolume) (string, bool) {
		if pv.Spec.AzureDisk != nil {
			return pv.Spec.AzureDisk.DiskName, true
		}
		return "", false
	},
}

// VolumeZoneChecker contains information to check the volume zone for a predicate.
type VolumeZoneChecker struct {
	pvInfo    PersistentVolumeInfo
	pvcInfo   PersistentVolumeClaimInfo
	classInfo StorageClassInfo
}

// NewVolumeZonePredicate evaluates if a pod can fit due to the volumes it requests, given
// that some volumes may have zone scheduling constraints.  The requirement is that any
// volume zone-labels must match the equivalent zone-labels on the node.  It is OK for
// the node to have more zone-label constraints (for example, a hypothetical replicated
// volume might allow region-wide access)
//
// Currently this is only supported with PersistentVolumeClaims, and looks to the labels
// only on the bound PersistentVolume.
//
// Working with volumes declared inline in the pod specification (i.e. not
// using a PersistentVolume) is likely to be harder, as it would require
// determining the zone of a volume during scheduling, and that is likely to
// require calling out to the cloud provider.  It seems that we are moving away
// from inline volume declarations anyway.
func NewVolumeZonePredicate(pvInfo PersistentVolumeInfo, pvcInfo PersistentVolumeClaimInfo, classInfo StorageClassInfo) algorithm.FitPredicate {
	c := &VolumeZoneChecker{
		pvInfo:    pvInfo,
		pvcInfo:   pvcInfo,
		classInfo: classInfo,
	}
	return c.predicate
}

func (c *VolumeZoneChecker) predicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return true, nil, nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	nodeConstraints := make(map[string]string)
	for k, v := range node.ObjectMeta.Labels {
		if k != kubeletapis.LabelZoneFailureDomain && k != kubeletapis.LabelZoneRegion {
			continue
		}
		nodeConstraints[k] = v
	}

	if len(nodeConstraints) == 0 {
		// The node has no zone constraints, so we're OK to schedule.
		// In practice, when using zones, all nodes must be labeled with zone labels.
		// We want to fast-path this case though.
		return true, nil, nil
	}

	namespace := pod.Namespace
	manifest := &(pod.Spec)
	for i := range manifest.Volumes {
		volume := &manifest.Volumes[i]
		if volume.PersistentVolumeClaim != nil {
			pvcName := volume.PersistentVolumeClaim.ClaimName
			if pvcName == "" {
				return false, nil, fmt.Errorf("PersistentVolumeClaim had no name")
			}
			pvc, err := c.pvcInfo.GetPersistentVolumeClaimInfo(namespace, pvcName)
			if err != nil {
				return false, nil, err
			}

			if pvc == nil {
				return false, nil, fmt.Errorf("PersistentVolumeClaim was not found: %q", pvcName)
			}

			pvName := pvc.Spec.VolumeName
			if pvName == "" {
				if utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
					scName := v1helper.GetPersistentVolumeClaimClass(pvc)
					if len(scName) > 0 {
						class, _ := c.classInfo.GetStorageClassInfo(scName)
						if class != nil {
							if class.VolumeBindingMode == nil {
								return false, nil, fmt.Errorf("VolumeBindingMode not set for StorageClass %q", scName)
							}
							if *class.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
								// Skip unbound volumes
								continue
							}
						}
					}
				}
				return false, nil, fmt.Errorf("PersistentVolumeClaim is not bound: %q", pvcName)
			}

			pv, err := c.pvInfo.GetPersistentVolumeInfo(pvName)
			if err != nil {
				return false, nil, err
			}

			if pv == nil {
				return false, nil, fmt.Errorf("PersistentVolume not found: %q", pvName)
			}

			for k, v := range pv.ObjectMeta.Labels {
				if k != kubeletapis.LabelZoneFailureDomain && k != kubeletapis.LabelZoneRegion {
					continue
				}
				nodeV, _ := nodeConstraints[k]
				volumeVSet, err := volumeutil.LabelZonesToSet(v)
				if err != nil {
					glog.Warningf("Failed to parse label for %q: %q. Ignoring the label. err=%v. ", k, v, err)
					continue
				}

				if !volumeVSet.Has(nodeV) {
					glog.V(10).Infof("Won't schedule pod %q onto node %q due to volume %q (mismatch on %q)", pod.Name, node.Name, pvName, k)
					return false, []algorithm.PredicateFailureReason{ErrVolumeZoneConflict}, nil
				}
			}
		}
	}

	return true, nil, nil
}

// GetResourceRequest returns a *schedulercache.Resource that covers the largest
// width in each resource dimension. Because init-containers run sequentially, we collect
// the max in each dimension iteratively. In contrast, we sum the resource vectors for
// regular containers since they run simultaneously.
//
// Example:
//
// Pod:
//   InitContainers
//     IC1:
//       CPU: 2
//       Memory: 1G
//     IC2:
//       CPU: 2
//       Memory: 3G
//   Containers
//     C1:
//       CPU: 2
//       Memory: 1G
//     C2:
//       CPU: 1
//       Memory: 1G
//
// Result: CPU: 3, Memory: 3G
func GetResourceRequest(pod *v1.Pod) *schedulercache.Resource {
	result := &schedulercache.Resource{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Requests)
	}

	// take max_resource(sum_pod, any_init_container)
	for _, container := range pod.Spec.InitContainers {
		result.SetMaxResource(container.Resources.Requests)
	}

	return result
}

func podName(pod *v1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

// PodFitsResources checks if a node has sufficient resources, such as cpu, memory, gpu, opaque int resources etc to run a pod.
// First return value indicates whether a node has sufficient resources to run a pod while the second return value indicates the
// predicate failure reasons if the node has insufficient resources to run the pod.
func PodFitsResources(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	var predicateFails []algorithm.PredicateFailureReason
	allowedPodNumber := nodeInfo.AllowedPodNumber()
	if len(nodeInfo.Pods())+1 > allowedPodNumber {
		predicateFails = append(predicateFails, NewInsufficientResourceError(v1.ResourcePods, 1, int64(len(nodeInfo.Pods())), int64(allowedPodNumber)))
	}

	// No extended resources should be ignored by default.
	ignoredExtendedResources := sets.NewString()

	var podRequest *schedulercache.Resource
	if predicateMeta, ok := meta.(*predicateMetadata); ok {
		podRequest = predicateMeta.podRequest
		if predicateMeta.ignoredExtendedResources != nil {
			ignoredExtendedResources = predicateMeta.ignoredExtendedResources
		}
	} else {
		// We couldn't parse metadata - fallback to computing it.
		podRequest = GetResourceRequest(pod)
	}
	if podRequest.MilliCPU == 0 &&
		podRequest.Memory == 0 &&
		podRequest.EphemeralStorage == 0 &&
		len(podRequest.ScalarResources) == 0 {
		return len(predicateFails) == 0, predicateFails, nil
	}

	allocatable := nodeInfo.AllocatableResource()
	if allocatable.MilliCPU < podRequest.MilliCPU+nodeInfo.RequestedResource().MilliCPU {
		predicateFails = append(predicateFails, NewInsufficientResourceError(v1.ResourceCPU, podRequest.MilliCPU, nodeInfo.RequestedResource().MilliCPU, allocatable.MilliCPU))
	}
	if allocatable.Memory < podRequest.Memory+nodeInfo.RequestedResource().Memory {
		predicateFails = append(predicateFails, NewInsufficientResourceError(v1.ResourceMemory, podRequest.Memory, nodeInfo.RequestedResource().Memory, allocatable.Memory))
	}
	if allocatable.EphemeralStorage < podRequest.EphemeralStorage+nodeInfo.RequestedResource().EphemeralStorage {
		predicateFails = append(predicateFails, NewInsufficientResourceError(v1.ResourceEphemeralStorage, podRequest.EphemeralStorage, nodeInfo.RequestedResource().EphemeralStorage, allocatable.EphemeralStorage))
	}

	for rName, rQuant := range podRequest.ScalarResources {
		if v1helper.IsExtendedResourceName(rName) {
			// If this resource is one of the extended resources that should be
			// ignored, we will skip checking it.
			if ignoredExtendedResources.Has(string(rName)) {
				continue
			}
		}
		if allocatable.ScalarResources[rName] < rQuant+nodeInfo.RequestedResource().ScalarResources[rName] {
			predicateFails = append(predicateFails, NewInsufficientResourceError(rName, podRequest.ScalarResources[rName], nodeInfo.RequestedResource().ScalarResources[rName], allocatable.ScalarResources[rName]))
		}
	}

	if glog.V(10) {
		if len(predicateFails) == 0 {
			// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
			// not logged. There is visible performance gain from it.
			glog.Infof("Schedule Pod %+v on Node %+v is allowed, Node is running only %v out of %v Pods.",
				podName(pod), node.Name, len(nodeInfo.Pods()), allowedPodNumber)
		}
	}
	return len(predicateFails) == 0, predicateFails, nil
}

// nodeMatchesNodeSelectorTerms checks if a node's labels satisfy a list of node selector terms,
// terms are ORed, and an empty list of terms will match nothing.
func nodeMatchesNodeSelectorTerms(node *v1.Node, nodeSelectorTerms []v1.NodeSelectorTerm) bool {
	nodeFields := map[string]string{}
	for k, f := range algorithm.NodeFieldSelectorKeys {
		nodeFields[k] = f(node)
	}
	return v1helper.MatchNodeSelectorTerms(nodeSelectorTerms, labels.Set(node.Labels), fields.Set(nodeFields))
}

// podMatchesNodeSelectorAndAffinityTerms checks whether the pod is schedulable onto nodes according to
// the requirements in both NodeAffinity and nodeSelector.
func podMatchesNodeSelectorAndAffinityTerms(pod *v1.Pod, node *v1.Node) bool {
	// Check if node.Labels match pod.Spec.NodeSelector.
	if len(pod.Spec.NodeSelector) > 0 {
		selector := labels.SelectorFromSet(pod.Spec.NodeSelector)
		if !selector.Matches(labels.Set(node.Labels)) {
			return false
		}
	}

	// 1. nil NodeSelector matches all nodes (i.e. does not filter out any nodes)
	// 2. nil []NodeSelectorTerm (equivalent to non-nil empty NodeSelector) matches no nodes
	// 3. zero-length non-nil []NodeSelectorTerm matches no nodes also, just for simplicity
	// 4. nil []NodeSelectorRequirement (equivalent to non-nil empty NodeSelectorTerm) matches no nodes
	// 5. zero-length non-nil []NodeSelectorRequirement matches no nodes also, just for simplicity
	// 6. non-nil empty NodeSelectorRequirement is not allowed
	nodeAffinityMatches := true
	affinity := pod.Spec.Affinity
	if affinity != nil && affinity.NodeAffinity != nil {
		nodeAffinity := affinity.NodeAffinity
		// if no required NodeAffinity requirements, will do no-op, means select all nodes.
		// TODO: Replace next line with subsequent commented-out line when implement RequiredDuringSchedulingRequiredDuringExecution.
		if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
			// if nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution == nil && nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution == nil {
			return true
		}

		// Match node selector for requiredDuringSchedulingRequiredDuringExecution.
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		// if nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution != nil {
		// 	nodeSelectorTerms := nodeAffinity.RequiredDuringSchedulingRequiredDuringExecution.NodeSelectorTerms
		// 	glog.V(10).Infof("Match for RequiredDuringSchedulingRequiredDuringExecution node selector terms %+v", nodeSelectorTerms)
		// 	nodeAffinityMatches = nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		// }

		// Match node selector for requiredDuringSchedulingIgnoredDuringExecution.
		if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			nodeSelectorTerms := nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
			glog.V(10).Infof("Match for RequiredDuringSchedulingIgnoredDuringExecution node selector terms %+v", nodeSelectorTerms)
			nodeAffinityMatches = nodeAffinityMatches && nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		}

	}
	return nodeAffinityMatches
}

// PodMatchNodeSelector checks if a pod node selector matches the node label.
func PodMatchNodeSelector(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if podMatchesNodeSelectorAndAffinityTerms(pod, node) {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{ErrNodeSelectorNotMatch}, nil
}

// PodFitsHost checks if a pod spec node name matches the current node.
func PodFitsHost(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if len(pod.Spec.NodeName) == 0 {
		return true, nil, nil
	}
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if pod.Spec.NodeName == node.Name {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{ErrPodNotMatchHostName}, nil
}

// NodeLabelChecker contains information to check node labels for a predicate.
type NodeLabelChecker struct {
	labels   []string
	presence bool
}

// NewNodeLabelPredicate creates a predicate which evaluates whether a pod can fit based on the
// node labels which match a filter that it requests.
func NewNodeLabelPredicate(labels []string, presence bool) algorithm.FitPredicate {
	labelChecker := &NodeLabelChecker{
		labels:   labels,
		presence: presence,
	}
	return labelChecker.CheckNodeLabelPresence
}

// CheckNodeLabelPresence checks whether all of the specified labels exists on a node or not, regardless of their value
// If "presence" is false, then returns false if any of the requested labels matches any of the node's labels,
// otherwise returns true.
// If "presence" is true, then returns false if any of the requested labels does not match any of the node's labels,
// otherwise returns true.
//
// Consider the cases where the nodes are placed in regions/zones/racks and these are identified by labels
// In some cases, it is required that only nodes that are part of ANY of the defined regions/zones/racks be selected
//
// Alternately, eliminating nodes that have a certain label, regardless of value, is also useful
// A node may have a label with "retiring" as key and the date as the value
// and it may be desirable to avoid scheduling new pods on this node
func (n *NodeLabelChecker) CheckNodeLabelPresence(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	var exists bool
	nodeLabels := labels.Set(node.Labels)
	for _, label := range n.labels {
		exists = nodeLabels.Has(label)
		if (exists && !n.presence) || (!exists && n.presence) {
			return false, []algorithm.PredicateFailureReason{ErrNodeLabelPresenceViolated}, nil
		}
	}
	return true, nil, nil
}

// ServiceAffinity defines a struct used for create service affinity predicates.
type ServiceAffinity struct {
	podLister     algorithm.PodLister
	serviceLister algorithm.ServiceLister
	nodeInfo      NodeInfo
	labels        []string
}

// serviceAffinityMetadataProducer should be run once by the scheduler before looping through the Predicate.  It is a helper function that
// only should be referenced by NewServiceAffinityPredicate.
func (s *ServiceAffinity) serviceAffinityMetadataProducer(pm *predicateMetadata) {
	if pm.pod == nil {
		glog.Errorf("Cannot precompute service affinity, a pod is required to calculate service affinity.")
		return
	}
	pm.serviceAffinityInUse = true
	var errSvc, errList error
	// Store services which match the pod.
	pm.serviceAffinityMatchingPodServices, errSvc = s.serviceLister.GetPodServices(pm.pod)
	selector := CreateSelectorFromLabels(pm.pod.Labels)
	allMatches, errList := s.podLister.List(selector)

	// In the future maybe we will return them as part of the function.
	if errSvc != nil || errList != nil {
		glog.Errorf("Some Error were found while precomputing svc affinity: \nservices:%v , \npods:%v", errSvc, errList)
	}
	// consider only the pods that belong to the same namespace
	pm.serviceAffinityMatchingPodList = FilterPodsByNamespace(allMatches, pm.pod.Namespace)
}

// NewServiceAffinityPredicate creates a ServiceAffinity.
func NewServiceAffinityPredicate(podLister algorithm.PodLister, serviceLister algorithm.ServiceLister, nodeInfo NodeInfo, labels []string) (algorithm.FitPredicate, PredicateMetadataProducer) {
	affinity := &ServiceAffinity{
		podLister:     podLister,
		serviceLister: serviceLister,
		nodeInfo:      nodeInfo,
		labels:        labels,
	}
	return affinity.checkServiceAffinity, affinity.serviceAffinityMetadataProducer
}

// checkServiceAffinity is a predicate which matches nodes in such a way to force that
// ServiceAffinity.labels are homogenous for pods that are scheduled to a node.
// (i.e. it returns true IFF this pod can be added to this node such that all other pods in
// the same service are running on nodes with the exact same ServiceAffinity.label values).
//
// For example:
// If the first pod of a service was scheduled to a node with label "region=foo",
// all the other subsequent pods belong to the same service will be schedule on
// nodes with the same "region=foo" label.
//
// Details:
//
// If (the svc affinity labels are not a subset of pod's label selectors )
// 	The pod has all information necessary to check affinity, the pod's label selector is sufficient to calculate
// 	the match.
// Otherwise:
// 	Create an "implicit selector" which guarantees pods will land on nodes with similar values
// 	for the affinity labels.
//
// 	To do this, we "reverse engineer" a selector by introspecting existing pods running under the same service+namespace.
//	These backfilled labels in the selector "L" are defined like so:
// 		- L is a label that the ServiceAffinity object needs as a matching constraints.
// 		- L is not defined in the pod itself already.
// 		- and SOME pod, from a service, in the same namespace, ALREADY scheduled onto a node, has a matching value.
//
// WARNING: This Predicate is NOT guaranteed to work if some of the predicateMetadata data isn't precomputed...
// For that reason it is not exported, i.e. it is highly coupled to the implementation of the FitPredicate construction.
func (s *ServiceAffinity) checkServiceAffinity(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var services []*v1.Service
	var pods []*v1.Pod
	if pm, ok := meta.(*predicateMetadata); ok && (pm.serviceAffinityMatchingPodList != nil || pm.serviceAffinityMatchingPodServices != nil) {
		services = pm.serviceAffinityMatchingPodServices
		pods = pm.serviceAffinityMatchingPodList
	} else {
		// Make the predicate resilient in case metadata is missing.
		pm = &predicateMetadata{pod: pod}
		s.serviceAffinityMetadataProducer(pm)
		pods, services = pm.serviceAffinityMatchingPodList, pm.serviceAffinityMatchingPodServices
	}
	filteredPods := nodeInfo.FilterOutPods(pods)
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	// check if the pod being scheduled has the affinity labels specified in its NodeSelector
	affinityLabels := FindLabelsInSet(s.labels, labels.Set(pod.Spec.NodeSelector))
	// Step 1: If we don't have all constraints, introspect nodes to find the missing constraints.
	if len(s.labels) > len(affinityLabels) {
		if len(services) > 0 {
			if len(filteredPods) > 0 {
				nodeWithAffinityLabels, err := s.nodeInfo.GetNodeInfo(filteredPods[0].Spec.NodeName)
				if err != nil {
					return false, nil, err
				}
				AddUnsetLabelsToMap(affinityLabels, s.labels, labels.Set(nodeWithAffinityLabels.Labels))
			}
		}
	}
	// Step 2: Finally complete the affinity predicate based on whatever set of predicates we were able to find.
	if CreateSelectorFromLabels(affinityLabels).Matches(labels.Set(node.Labels)) {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{ErrServiceAffinityViolated}, nil
}

// PodFitsHostPorts checks if a node has free ports for the requested pod ports.
func PodFitsHostPorts(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var wantPorts []*v1.ContainerPort
	if predicateMeta, ok := meta.(*predicateMetadata); ok {
		wantPorts = predicateMeta.podPorts
	} else {
		// We couldn't parse metadata - fallback to computing it.
		wantPorts = schedutil.GetContainerPorts(pod)
	}
	if len(wantPorts) == 0 {
		return true, nil, nil
	}

	existingPorts := nodeInfo.UsedPorts()

	// try to see whether existingPorts and  wantPorts will conflict or not
	if portsConflict(existingPorts, wantPorts) {
		return false, []algorithm.PredicateFailureReason{ErrPodNotFitsHostPorts}, nil
	}

	return true, nil, nil
}

// search two arrays and return true if they have at least one common element; return false otherwise
func haveOverlap(a1, a2 []string) bool {
	m := map[string]bool{}

	for _, val := range a1 {
		m[val] = true
	}
	for _, val := range a2 {
		if _, ok := m[val]; ok {
			return true
		}
	}

	return false
}

// GeneralPredicates checks whether noncriticalPredicates and EssentialPredicates pass. noncriticalPredicates are the predicates
// that only non-critical pods need and EssentialPredicates are the predicates that all pods, including critical pods, need
func GeneralPredicates(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var predicateFails []algorithm.PredicateFailureReason
	fit, reasons, err := noncriticalPredicates(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	fit, reasons, err = EssentialPredicates(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	return len(predicateFails) == 0, predicateFails, nil
}

// noncriticalPredicates are the predicates that only non-critical pods need
func noncriticalPredicates(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var predicateFails []algorithm.PredicateFailureReason
	fit, reasons, err := PodFitsResources(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	return len(predicateFails) == 0, predicateFails, nil
}

// EssentialPredicates are the predicates that all pods, including critical pods, need
func EssentialPredicates(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var predicateFails []algorithm.PredicateFailureReason
	fit, reasons, err := PodFitsHost(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	// TODO: PodFitsHostPorts is essential for now, but kubelet should ideally
	//       preempt pods to free up host ports too
	fit, reasons, err = PodFitsHostPorts(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}

	fit, reasons, err = PodMatchNodeSelector(pod, meta, nodeInfo)
	if err != nil {
		return false, predicateFails, err
	}
	if !fit {
		predicateFails = append(predicateFails, reasons...)
	}
	return len(predicateFails) == 0, predicateFails, nil
}

// PodAffinityChecker contains information to check pod affinity.
type PodAffinityChecker struct {
	info      NodeInfo
	podLister algorithm.PodLister
}

// NewPodAffinityPredicate creates a PodAffinityChecker.
func NewPodAffinityPredicate(info NodeInfo, podLister algorithm.PodLister) algorithm.FitPredicate {
	checker := &PodAffinityChecker{
		info:      info,
		podLister: podLister,
	}
	return checker.InterPodAffinityMatches
}

// InterPodAffinityMatches checks if a pod can be scheduled on the specified node with pod affinity/anti-affinity configuration.
// First return value indicates whether a pod can be scheduled on the specified node while the second return value indicates the
// predicate failure reasons if the pod cannot be scheduled on the specified node.
func (c *PodAffinityChecker) InterPodAffinityMatches(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if failedPredicates, error := c.satisfiesExistingPodsAntiAffinity(pod, meta, nodeInfo); failedPredicates != nil {
		failedPredicates := append([]algorithm.PredicateFailureReason{ErrPodAffinityNotMatch}, failedPredicates)
		return false, failedPredicates, error
	}

	// Now check if <pod> requirements will be satisfied on this node.
	affinity := pod.Spec.Affinity
	if affinity == nil || (affinity.PodAffinity == nil && affinity.PodAntiAffinity == nil) {
		return true, nil, nil
	}
	if failedPredicates, error := c.satisfiesPodsAffinityAntiAffinity(pod, meta, nodeInfo, affinity); failedPredicates != nil {
		failedPredicates := append([]algorithm.PredicateFailureReason{ErrPodAffinityNotMatch}, failedPredicates)
		return false, failedPredicates, error
	}

	if glog.V(10) {
		// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		glog.Infof("Schedule Pod %+v on Node %+v is allowed, pod (anti)affinity constraints satisfied",
			podName(pod), node.Name)
	}
	return true, nil, nil
}

// podMatchesPodAffinityTerms checks if the "targetPod" matches the given "terms"
// of the "pod" on the given "nodeInfo".Node(). It returns three values: 1) whether
// targetPod matches all the terms and their topologies, 2) whether targetPod
// matches all the terms label selector and namespaces (AKA term properties),
// 3) any error.
func (c *PodAffinityChecker) podMatchesPodAffinityTerms(pod, targetPod *v1.Pod, nodeInfo *schedulercache.NodeInfo, terms []v1.PodAffinityTerm) (bool, bool, error) {
	if len(terms) == 0 {
		return false, false, fmt.Errorf("terms array is empty")
	}
	props, err := getAffinityTermProperties(pod, terms)
	if err != nil {
		return false, false, err
	}
	if !podMatchesAllAffinityTermProperties(targetPod, props) {
		return false, false, nil
	}
	// Namespace and selector of the terms have matched. Now we check topology of the terms.
	targetPodNode, err := c.info.GetNodeInfo(targetPod.Spec.NodeName)
	if err != nil {
		return false, false, err
	}
	for _, term := range terms {
		if len(term.TopologyKey) == 0 {
			return false, false, fmt.Errorf("empty topologyKey is not allowed except for PreferredDuringScheduling pod anti-affinity")
		}
		if !priorityutil.NodesHaveSameTopologyKey(nodeInfo.Node(), targetPodNode, term.TopologyKey) {
			return false, true, nil
		}
	}
	return true, true, nil
}

// GetPodAffinityTerms gets pod affinity terms by a pod affinity object.
func GetPodAffinityTerms(podAffinity *v1.PodAffinity) (terms []v1.PodAffinityTerm) {
	if podAffinity != nil {
		if len(podAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
			terms = podAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		}
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		//if len(podAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
		//	terms = append(terms, podAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
		//}
	}
	return terms
}

// GetPodAntiAffinityTerms gets pod affinity terms by a pod anti-affinity.
func GetPodAntiAffinityTerms(podAntiAffinity *v1.PodAntiAffinity) (terms []v1.PodAffinityTerm) {
	if podAntiAffinity != nil {
		if len(podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution) != 0 {
			terms = podAntiAffinity.RequiredDuringSchedulingIgnoredDuringExecution
		}
		// TODO: Uncomment this block when implement RequiredDuringSchedulingRequiredDuringExecution.
		//if len(podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution) != 0 {
		//	terms = append(terms, podAntiAffinity.RequiredDuringSchedulingRequiredDuringExecution...)
		//}
	}
	return terms
}

// getMatchingAntiAffinityTopologyPairs calculates the following for "existingPod" on given node:
// (1) Whether it has PodAntiAffinity
// (2) Whether ANY AffinityTerm matches the incoming pod
func getMatchingAntiAffinityTopologyPairsOfPod(newPod *v1.Pod, existingPod *v1.Pod, node *v1.Node) (*topologyPairsMaps, error) {
	affinity := existingPod.Spec.Affinity
	if affinity == nil || affinity.PodAntiAffinity == nil {
		return nil, nil
	}

	topologyMaps := newTopologyPairsMaps()
	for _, term := range GetPodAntiAffinityTerms(affinity.PodAntiAffinity) {
		namespaces := priorityutil.GetNamespacesFromPodAffinityTerm(existingPod, &term)
		selector, err := metav1.LabelSelectorAsSelector(term.LabelSelector)
		if err != nil {
			return nil, err
		}
		if priorityutil.PodMatchesTermsNamespaceAndSelector(newPod, namespaces, selector) {
			if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
				pair := topologyPair{key: term.TopologyKey, value: topologyValue}
				topologyMaps.addTopologyPair(pair, existingPod)
			}
		}
	}
	return topologyMaps, nil
}

func (c *PodAffinityChecker) getMatchingAntiAffinityTopologyPairsOfPods(pod *v1.Pod, existingPods []*v1.Pod) (*topologyPairsMaps, error) {
	topologyMaps := newTopologyPairsMaps()

	for _, existingPod := range existingPods {
		existingPodNode, err := c.info.GetNodeInfo(existingPod.Spec.NodeName)
		if err != nil {
			if apierrors.IsNotFound(err) {
				glog.Errorf("Node not found, %v", existingPod.Spec.NodeName)
				continue
			}
			return nil, err
		}
		existingPodTopologyMaps, err := getMatchingAntiAffinityTopologyPairsOfPod(pod, existingPod, existingPodNode)
		if err != nil {
			return nil, err
		}
		topologyMaps.appendMaps(existingPodTopologyMaps)
	}
	return topologyMaps, nil
}

// Checks if scheduling the pod onto this node would break any anti-affinity
// terms indicated by the existing pods.
func (c *PodAffinityChecker) satisfiesExistingPodsAntiAffinity(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return ErrExistingPodsAntiAffinityRulesNotMatch, fmt.Errorf("Node is nil")
	}
	var topologyMaps *topologyPairsMaps
	if predicateMeta, ok := meta.(*predicateMetadata); ok {
		topologyMaps = predicateMeta.topologyPairsAntiAffinityPodsMap
	} else {
		// Filter out pods whose nodeName is equal to nodeInfo.node.Name, but are not
		// present in nodeInfo. Pods on other nodes pass the filter.
		filteredPods, err := c.podLister.FilteredList(nodeInfo.Filter, labels.Everything())
		if err != nil {
			errMessage := fmt.Sprintf("Failed to get all pods, %+v", err)
			glog.Error(errMessage)
			return ErrExistingPodsAntiAffinityRulesNotMatch, errors.New(errMessage)
		}
		if topologyMaps, err = c.getMatchingAntiAffinityTopologyPairsOfPods(pod, filteredPods); err != nil {
			errMessage := fmt.Sprintf("Failed to get all terms that pod %+v matches, err: %+v", podName(pod), err)
			glog.Error(errMessage)
			return ErrExistingPodsAntiAffinityRulesNotMatch, errors.New(errMessage)
		}
	}

	// Iterate over topology pairs to get any of the pods being affected by
	// the scheduled pod anti-affinity terms
	for topologyKey, topologyValue := range node.Labels {
		if topologyMaps.topologyPairToPods[topologyPair{key: topologyKey, value: topologyValue}] != nil {
			glog.V(10).Infof("Cannot schedule pod %+v onto node %v", podName(pod), node.Name)
			return ErrExistingPodsAntiAffinityRulesNotMatch, nil
		}
	}
	if glog.V(10) {
		// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		glog.Infof("Schedule Pod %+v on Node %+v is allowed, existing pods anti-affinity terms satisfied.",
			podName(pod), node.Name)
	}
	return nil, nil
}

//  nodeMatchesAllTopologyTerms checks whether "nodeInfo" matches
//  topology of all the "terms" for the given "pod".
func (c *PodAffinityChecker) nodeMatchesAllTopologyTerms(pod *v1.Pod, topologyPairs *topologyPairsMaps, nodeInfo *schedulercache.NodeInfo, terms []v1.PodAffinityTerm) bool {
	node := nodeInfo.Node()
	for _, term := range terms {
		if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
			pair := topologyPair{key: term.TopologyKey, value: topologyValue}
			if _, ok := topologyPairs.topologyPairToPods[pair]; !ok {
				return false
			}
		} else {
			return false
		}
	}
	return true
}

//  nodeMatchesAnyTopologyTerm checks whether "nodeInfo" matches
//  topology of any "term" for the given "pod".
func (c *PodAffinityChecker) nodeMatchesAnyTopologyTerm(pod *v1.Pod, topologyPairs *topologyPairsMaps, nodeInfo *schedulercache.NodeInfo, terms []v1.PodAffinityTerm) bool {
	node := nodeInfo.Node()
	for _, term := range terms {
		if topologyValue, ok := node.Labels[term.TopologyKey]; ok {
			pair := topologyPair{key: term.TopologyKey, value: topologyValue}
			if _, ok := topologyPairs.topologyPairToPods[pair]; ok {
				return true
			}
		}
	}
	return false
}

// Checks if scheduling the pod onto this node would break any term of this pod.
func (c *PodAffinityChecker) satisfiesPodsAffinityAntiAffinity(pod *v1.Pod,
	meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo,
	affinity *v1.Affinity) (algorithm.PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return ErrPodAffinityRulesNotMatch, fmt.Errorf("Node is nil")
	}
	if predicateMeta, ok := meta.(*predicateMetadata); ok {
		// Check all affinity terms.
		topologyPairsPotentialAffinityPods := predicateMeta.topologyPairsPotentialAffinityPods
		if affinityTerms := GetPodAffinityTerms(affinity.PodAffinity); len(affinityTerms) > 0 {
			matchExists := c.nodeMatchesAllTopologyTerms(pod, topologyPairsPotentialAffinityPods, nodeInfo, affinityTerms)
			if !matchExists {
				// This pod may the first pod in a series that have affinity to themselves. In order
				// to not leave such pods in pending state forever, we check that if no other pod
				// in the cluster matches the namespace and selector of this pod and the pod matches
				// its own terms, then we allow the pod to pass the affinity check.
				if !(len(topologyPairsPotentialAffinityPods.topologyPairToPods) == 0 && targetPodMatchesAffinityOfPod(pod, pod)) {
					glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because of PodAffinity",
						podName(pod), node.Name)
					return ErrPodAffinityRulesNotMatch, nil
				}
			}
		}

		// Check all anti-affinity terms.
		topologyPairsPotentialAntiAffinityPods := predicateMeta.topologyPairsPotentialAntiAffinityPods
		if antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity); len(antiAffinityTerms) > 0 {
			matchExists := c.nodeMatchesAnyTopologyTerm(pod, topologyPairsPotentialAntiAffinityPods, nodeInfo, antiAffinityTerms)
			if matchExists {
				glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because of PodAntiAffinity",
					podName(pod), node.Name)
				return ErrPodAntiAffinityRulesNotMatch, nil
			}
		}
	} else { // We don't have precomputed metadata. We have to follow a slow path to check affinity terms.
		filteredPods, err := c.podLister.FilteredList(nodeInfo.Filter, labels.Everything())
		if err != nil {
			return ErrPodAffinityRulesNotMatch, err
		}

		affinityTerms := GetPodAffinityTerms(affinity.PodAffinity)
		antiAffinityTerms := GetPodAntiAffinityTerms(affinity.PodAntiAffinity)
		matchFound, termsSelectorMatchFound := false, false
		for _, targetPod := range filteredPods {
			// Check all affinity terms.
			if !matchFound && len(affinityTerms) > 0 {
				affTermsMatch, termsSelectorMatch, err := c.podMatchesPodAffinityTerms(pod, targetPod, nodeInfo, affinityTerms)
				if err != nil {
					errMessage := fmt.Sprintf("Cannot schedule pod %+v onto node %v, because of PodAffinity, err: %v", podName(pod), node.Name, err)
					glog.Error(errMessage)
					return ErrPodAffinityRulesNotMatch, errors.New(errMessage)
				}
				if termsSelectorMatch {
					termsSelectorMatchFound = true
				}
				if affTermsMatch {
					matchFound = true
				}
			}

			// Check all anti-affinity terms.
			if len(antiAffinityTerms) > 0 {
				antiAffTermsMatch, _, err := c.podMatchesPodAffinityTerms(pod, targetPod, nodeInfo, antiAffinityTerms)
				if err != nil || antiAffTermsMatch {
					glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because of PodAntiAffinityTerm, err: %v",
						podName(pod), node.Name, err)
					return ErrPodAntiAffinityRulesNotMatch, nil
				}
			}
		}

		if !matchFound && len(affinityTerms) > 0 {
			// We have not been able to find any matches for the pod's affinity terms.
			// This pod may be the first pod in a series that have affinity to themselves. In order
			// to not leave such pods in pending state forever, we check that if no other pod
			// in the cluster matches the namespace and selector of this pod and the pod matches
			// its own terms, then we allow the pod to pass the affinity check.
			if termsSelectorMatchFound {
				glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because of PodAffinity",
					podName(pod), node.Name)
				return ErrPodAffinityRulesNotMatch, nil
			}
			// Check if pod matches its own affinity properties (namespace and label selector).
			if !targetPodMatchesAffinityOfPod(pod, pod) {
				glog.V(10).Infof("Cannot schedule pod %+v onto node %v, because of PodAffinity",
					podName(pod), node.Name)
				return ErrPodAffinityRulesNotMatch, nil
			}
		}
	}

	if glog.V(10) {
		// We explicitly don't do glog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		glog.Infof("Schedule Pod %+v on Node %+v is allowed, pod affinity/anti-affinity constraints satisfied.",
			podName(pod), node.Name)
	}
	return nil, nil
}

// CheckNodeUnschedulablePredicate checks if a pod can be scheduled on a node with Unschedulable spec.
func CheckNodeUnschedulablePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnknownCondition}, nil
	}

	// If pod tolerate unschedulable taint, it's also tolerate `node.Spec.Unschedulable`.
	podToleratesUnschedulable := v1helper.TolerationsTolerateTaint(pod.Spec.Tolerations, &v1.Taint{
		Key:    algorithm.TaintNodeUnschedulable,
		Effect: v1.TaintEffectNoSchedule,
	})

	// TODO (k82cn): deprecates `node.Spec.Unschedulable` in 1.13.
	if nodeInfo.Node().Spec.Unschedulable && !podToleratesUnschedulable {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnschedulable}, nil
	}

	return true, nil, nil
}

// PodToleratesNodeTaints checks if a pod tolerations can tolerate the node taints
func PodToleratesNodeTaints(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnknownCondition}, nil
	}

	return podToleratesNodeTaints(pod, nodeInfo, func(t *v1.Taint) bool {
		// PodToleratesNodeTaints is only interested in NoSchedule and NoExecute taints.
		return t.Effect == v1.TaintEffectNoSchedule || t.Effect == v1.TaintEffectNoExecute
	})
}

// PodToleratesNodeNoExecuteTaints checks if a pod tolerations can tolerate the node's NoExecute taints
func PodToleratesNodeNoExecuteTaints(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return podToleratesNodeTaints(pod, nodeInfo, func(t *v1.Taint) bool {
		return t.Effect == v1.TaintEffectNoExecute
	})
}

func podToleratesNodeTaints(pod *v1.Pod, nodeInfo *schedulercache.NodeInfo, filter func(t *v1.Taint) bool) (bool, []algorithm.PredicateFailureReason, error) {
	taints, err := nodeInfo.Taints()
	if err != nil {
		return false, nil, err
	}

	if v1helper.TolerationsTolerateTaintsWithFilter(pod.Spec.Tolerations, taints, filter) {
		return true, nil, nil
	}
	return false, []algorithm.PredicateFailureReason{ErrTaintsTolerationsNotMatch}, nil
}

// isPodBestEffort checks if pod is scheduled with best-effort QoS
func isPodBestEffort(pod *v1.Pod) bool {
	return v1qos.GetPodQOS(pod) == v1.PodQOSBestEffort
}

// CheckNodeMemoryPressurePredicate checks if a pod can be scheduled on a node
// reporting memory pressure condition.
func CheckNodeMemoryPressurePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	var podBestEffort bool
	if predicateMeta, ok := meta.(*predicateMetadata); ok {
		podBestEffort = predicateMeta.podBestEffort
	} else {
		// We couldn't parse metadata - fallback to computing it.
		podBestEffort = isPodBestEffort(pod)
	}
	// pod is not BestEffort pod
	if !podBestEffort {
		return true, nil, nil
	}

	// check if node is under memory pressure
	if nodeInfo.MemoryPressureCondition() == v1.ConditionTrue {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnderMemoryPressure}, nil
	}
	return true, nil, nil
}

// CheckNodeDiskPressurePredicate checks if a pod can be scheduled on a node
// reporting disk pressure condition.
func CheckNodeDiskPressurePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// check if node is under disk pressure
	if nodeInfo.DiskPressureCondition() == v1.ConditionTrue {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnderDiskPressure}, nil
	}
	return true, nil, nil
}

// CheckNodePIDPressurePredicate checks if a pod can be scheduled on a node
// reporting pid pressure condition.
func CheckNodePIDPressurePredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	// check if node is under pid pressure
	if nodeInfo.PIDPressureCondition() == v1.ConditionTrue {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnderPIDPressure}, nil
	}
	return true, nil, nil
}

// CheckNodeConditionPredicate checks if a pod can be scheduled on a node reporting out of disk,
// network unavailable and not ready condition. Only node conditions are accounted in this predicate.
func CheckNodeConditionPredicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	reasons := []algorithm.PredicateFailureReason{}

	if nodeInfo == nil || nodeInfo.Node() == nil {
		return false, []algorithm.PredicateFailureReason{ErrNodeUnknownCondition}, nil
	}

	node := nodeInfo.Node()
	for _, cond := range node.Status.Conditions {
		// We consider the node for scheduling only when its:
		// - NodeReady condition status is ConditionTrue,
		// - NodeOutOfDisk condition status is ConditionFalse,
		// - NodeNetworkUnavailable condition status is ConditionFalse.
		if cond.Type == v1.NodeReady && cond.Status != v1.ConditionTrue {
			reasons = append(reasons, ErrNodeNotReady)
		} else if cond.Type == v1.NodeOutOfDisk && cond.Status != v1.ConditionFalse {
			reasons = append(reasons, ErrNodeOutOfDisk)
		} else if cond.Type == v1.NodeNetworkUnavailable && cond.Status != v1.ConditionFalse {
			reasons = append(reasons, ErrNodeNetworkUnavailable)
		}
	}

	if node.Spec.Unschedulable {
		reasons = append(reasons, ErrNodeUnschedulable)
	}

	return len(reasons) == 0, reasons, nil
}

// VolumeBindingChecker contains information to check a volume binding.
type VolumeBindingChecker struct {
	binder *volumebinder.VolumeBinder
}

// NewVolumeBindingPredicate evaluates if a pod can fit due to the volumes it requests,
// for both bound and unbound PVCs.
//
// For PVCs that are bound, then it checks that the corresponding PV's node affinity is
// satisfied by the given node.
//
// For PVCs that are unbound, it tries to find available PVs that can satisfy the PVC requirements
// and that the PV node affinity is satisfied by the given node.
//
// The predicate returns true if all bound PVCs have compatible PVs with the node, and if all unbound
// PVCs can be matched with an available and node-compatible PV.
func NewVolumeBindingPredicate(binder *volumebinder.VolumeBinder) algorithm.FitPredicate {
	c := &VolumeBindingChecker{
		binder: binder,
	}
	return c.predicate
}

func (c *VolumeBindingChecker) predicate(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.VolumeScheduling) {
		return true, nil, nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	unboundSatisfied, boundSatisfied, err := c.binder.Binder.FindPodVolumes(pod, node)
	if err != nil {
		return false, nil, err
	}

	failReasons := []algorithm.PredicateFailureReason{}
	if !boundSatisfied {
		glog.V(5).Infof("Bound PVs not satisfied for pod %v/%v, node %q", pod.Namespace, pod.Name, node.Name)
		failReasons = append(failReasons, ErrVolumeNodeConflict)
	}

	if !unboundSatisfied {
		glog.V(5).Infof("Couldn't find matching PVs for pod %v/%v, node %q", pod.Namespace, pod.Name, node.Name)
		failReasons = append(failReasons, ErrVolumeBindConflict)
	}

	if len(failReasons) > 0 {
		return false, failReasons, nil
	}

	// All volumes bound or matching PVs found for all unbound PVCs
	glog.V(5).Infof("All PVCs found matches for pod %v/%v, node %q", pod.Namespace, pod.Name, node.Name)
	return true, nil, nil
}

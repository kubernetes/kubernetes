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

	"k8s.io/klog"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	v1helper "k8s.io/kubernetes/pkg/apis/core/v1/helper"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	schedutil "k8s.io/kubernetes/pkg/scheduler/util"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// MatchInterPodAffinityPred defines the name of predicate MatchInterPodAffinity.
	MatchInterPodAffinityPred = "MatchInterPodAffinity"
	// CheckVolumeBindingPred defines the name of predicate CheckVolumeBinding.
	CheckVolumeBindingPred = "CheckVolumeBinding"
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
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxEBSVolumeCountPred = "MaxEBSVolumeCount"
	// MaxGCEPDVolumeCountPred defines the name of predicate MaxGCEPDVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxGCEPDVolumeCountPred = "MaxGCEPDVolumeCount"
	// MaxAzureDiskVolumeCountPred defines the name of predicate MaxAzureDiskVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxAzureDiskVolumeCountPred = "MaxAzureDiskVolumeCount"
	// MaxCinderVolumeCountPred defines the name of predicate MaxCinderDiskVolumeCount.
	// DEPRECATED
	// All cloudprovider specific predicates are deprecated in favour of MaxCSIVolumeCountPred.
	MaxCinderVolumeCountPred = "MaxCinderVolumeCount"
	// MaxCSIVolumeCountPred defines the predicate that decides how many CSI volumes should be attached.
	MaxCSIVolumeCountPred = "MaxCSIVolumeCountPred"
	// NoVolumeZoneConflictPred defines the name of predicate NoVolumeZoneConflict.
	NoVolumeZoneConflictPred = "NoVolumeZoneConflict"
	// EvenPodsSpreadPred defines the name of predicate EvenPodsSpread.
	EvenPodsSpreadPred = "EvenPodsSpread"

	// DefaultMaxGCEPDVolumes defines the maximum number of PD Volumes for GCE.
	// GCE instances can have up to 16 PD volumes attached.
	DefaultMaxGCEPDVolumes = 16
	// DefaultMaxAzureDiskVolumes defines the maximum number of PD Volumes for Azure.
	// Larger Azure VMs can actually have much more disks attached.
	// TODO We should determine the max based on VM size
	DefaultMaxAzureDiskVolumes = 16

	// KubeMaxPDVols defines the maximum number of PD Volumes per kubelet.
	KubeMaxPDVols = "KUBE_MAX_PD_VOLS"

	// EBSVolumeFilterType defines the filter name for EBSVolumeFilter.
	EBSVolumeFilterType = "EBS"
	// GCEPDVolumeFilterType defines the filter name for GCEPDVolumeFilter.
	GCEPDVolumeFilterType = "GCE"
	// AzureDiskVolumeFilterType defines the filter name for AzureDiskVolumeFilter.
	AzureDiskVolumeFilterType = "AzureDisk"
	// CinderVolumeFilterType defines the filter name for CinderVolumeFilter.
	CinderVolumeFilterType = "Cinder"
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
	predicatesOrdering = []string{CheckNodeUnschedulablePred,
		GeneralPred, HostNamePred, PodFitsHostPortsPred,
		MatchNodeSelectorPred, PodFitsResourcesPred, NoDiskConflictPred,
		PodToleratesNodeTaintsPred, PodToleratesNodeNoExecuteTaintsPred, CheckNodeLabelPresencePred,
		CheckServiceAffinityPred, MaxEBSVolumeCountPred, MaxGCEPDVolumeCountPred, MaxCSIVolumeCountPred,
		MaxAzureDiskVolumeCountPred, MaxCinderVolumeCountPred, CheckVolumeBindingPred, NoVolumeZoneConflictPred,
		EvenPodsSpreadPred, MatchInterPodAffinityPred}
)

// Ordering returns the ordering of predicates.
func Ordering() []string {
	return predicatesOrdering
}

// FitPredicate is a function that indicates if a pod fits into an existing node.
// The failure information is given by the error.
type FitPredicate func(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error)

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
// This is GCE, Amazon EBS, ISCSI and Ceph RBD specific for now:
// - GCE PD allows multiple mounts as long as they're all read-only
// - AWS EBS forbids any two pods mounting the same volume ID
// - Ceph RBD forbids if any two pods share at least same monitor, and match pool and image, and the image is read-only
// - ISCSI forbids if any two pods share at least same IQN and ISCSI volume is read-only
// TODO: migrate this into some per-volume specific code?
func NoDiskConflict(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	for _, v := range pod.Spec.Volumes {
		for _, ev := range nodeInfo.Pods() {
			if isVolumeConflict(v, ev) {
				return false, []PredicateFailureReason{ErrDiskConflict}, nil
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
	csiNodeLister  storagelisters.CSINodeLister
	pvLister       corelisters.PersistentVolumeLister
	pvcLister      corelisters.PersistentVolumeClaimLister
	scLister       storagelisters.StorageClassLister

	// The string below is generated randomly during the struct's initialization.
	// It is used to prefix volumeID generated inside the predicate() method to
	// avoid conflicts with any real volume.
	randomVolumeIDPrefix string
}

// VolumeFilter contains information on how to filter PD Volumes when checking PD Volume caps.
type VolumeFilter struct {
	// Filter normal volumes
	FilterVolume           func(vol *v1.Volume) (id string, relevant bool)
	FilterPersistentVolume func(pv *v1.PersistentVolume) (id string, relevant bool)
	// MatchProvisioner evaluates if the StorageClass provisioner matches the running predicate
	MatchProvisioner func(sc *storage.StorageClass) (relevant bool)
	// IsMigrated returns a boolean specifying whether the plugin is migrated to a CSI driver
	IsMigrated func(csiNode *storage.CSINode) bool
}

// NewMaxPDVolumeCountPredicate creates a predicate which evaluates whether a pod can fit based on the
// number of volumes which match a filter that it requests, and those that are already present.
//
// DEPRECATED
// All cloudprovider specific predicates defined here are deprecated in favour of CSI volume limit
// predicate - MaxCSIVolumeCountPred.
//
// The predicate looks for both volumes used directly, as well as PVC volumes that are backed by relevant volume
// types, counts the number of unique volumes, and rejects the new pod if it would place the total count over
// the maximum.
func NewMaxPDVolumeCountPredicate(filterName string, csiNodeLister storagelisters.CSINodeLister, scLister storagelisters.StorageClassLister,
	pvLister corelisters.PersistentVolumeLister, pvcLister corelisters.PersistentVolumeClaimLister) FitPredicate {
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
	case CinderVolumeFilterType:
		filter = CinderVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.CinderVolumeLimitKey)
	default:
		klog.Fatalf("Wrong filterName, Only Support %v %v %v ", EBSVolumeFilterType,
			GCEPDVolumeFilterType, AzureDiskVolumeFilterType)
		return nil

	}
	c := &MaxPDVolumeCountChecker{
		filter:               filter,
		volumeLimitKey:       volumeLimitKey,
		maxVolumeFunc:        getMaxVolumeFunc(filterName),
		csiNodeLister:        csiNodeLister,
		pvLister:             pvLister,
		pvcLister:            pvcLister,
		scLister:             scLister,
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
			if k == v1.LabelInstanceType || k == v1.LabelInstanceTypeStable {
				nodeInstanceType = v
				break
			}
		}
		switch filterName {
		case EBSVolumeFilterType:
			return getMaxEBSVolume(nodeInstanceType)
		case GCEPDVolumeFilterType:
			return DefaultMaxGCEPDVolumes
		case AzureDiskVolumeFilterType:
			return DefaultMaxAzureDiskVolumes
		case CinderVolumeFilterType:
			return volumeutil.DefaultMaxCinderVolumes
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

// getMaxVolLimitFromEnv checks the max PD volumes environment variable, otherwise returning a default value.
func getMaxVolLimitFromEnv() int {
	if rawMaxVols := os.Getenv(KubeMaxPDVols); rawMaxVols != "" {
		if parsedMaxVols, err := strconv.Atoi(rawMaxVols); err != nil {
			klog.Errorf("Unable to parse maximum PD volumes value, using default: %v", err)
		} else if parsedMaxVols <= 0 {
			klog.Errorf("Maximum PD volumes must be a positive value, using default")
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

			pvc, err := c.pvcLister.PersistentVolumeClaims(namespace).Get(pvcName)
			if err != nil || pvc == nil {
				// If the PVC is invalid, we don't count the volume because
				// there's no guarantee that it belongs to the running predicate.
				klog.V(4).Infof("Unable to look up PVC info for %s/%s, assuming PVC doesn't match predicate when counting limits: %v", namespace, pvcName, err)
				continue
			}

			pvName := pvc.Spec.VolumeName
			if pvName == "" {
				// PVC is not bound. It was either deleted and created again or
				// it was forcefully unbound by admin. The pod can still use the
				// original PV where it was bound to, so we count the volume if
				// it belongs to the running predicate.
				if c.matchProvisioner(pvc) {
					klog.V(4).Infof("PVC %s/%s is not bound, assuming PVC matches predicate when counting limits", namespace, pvcName)
					filteredVolumes[pvID] = true
				}
				continue
			}

			pv, err := c.pvLister.Get(pvName)
			if err != nil || pv == nil {
				// If the PV is invalid and PVC belongs to the running predicate,
				// log the error and count the PV towards the PV limit.
				if c.matchProvisioner(pvc) {
					klog.V(4).Infof("Unable to look up PV info for %s/%s/%s, assuming PV matches predicate when counting limits: %v", namespace, pvcName, pvName, err)
					filteredVolumes[pvID] = true
				}
				continue
			}

			if id, ok := c.filter.FilterPersistentVolume(pv); ok {
				filteredVolumes[id] = true
			}
		}
	}

	return nil
}

// matchProvisioner helps identify if the given PVC belongs to the running predicate.
func (c *MaxPDVolumeCountChecker) matchProvisioner(pvc *v1.PersistentVolumeClaim) bool {
	if pvc.Spec.StorageClassName == nil {
		return false
	}

	storageClass, err := c.scLister.Get(*pvc.Spec.StorageClassName)
	if err != nil || storageClass == nil {
		return false
	}

	return c.filter.MatchProvisioner(storageClass)
}

func (c *MaxPDVolumeCountChecker) predicate(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
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

	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	var (
		csiNode *storage.CSINode
		err     error
	)
	if c.csiNodeLister != nil {
		csiNode, err = c.csiNodeLister.Get(node.Name)
		if err != nil {
			// we don't fail here because the CSINode object is only necessary
			// for determining whether the migration is enabled or not
			klog.V(5).Infof("Could not get a CSINode object for the node: %v", err)
		}
	}

	// if a plugin has been migrated to a CSI driver, defer to the CSI predicate
	if c.filter.IsMigrated(csiNode) {
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
	maxAttachLimit := c.maxVolumeFunc(node)

	volumeLimits := nodeInfo.VolumeLimits()
	if maxAttachLimitFromAllocatable, ok := volumeLimits[c.volumeLimitKey]; ok {
		maxAttachLimit = int(maxAttachLimitFromAllocatable)
	}

	if numExistingVolumes+numNewVolumes > maxAttachLimit {
		// violates MaxEBSVolumeCount or MaxGCEPDVolumeCount
		return false, []PredicateFailureReason{ErrMaxVolumeCountExceeded}, nil
	}
	if nodeInfo != nil && nodeInfo.TransientInfo != nil && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) {
		nodeInfo.TransientInfo.TransientLock.Lock()
		defer nodeInfo.TransientInfo.TransientLock.Unlock()
		nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount = maxAttachLimit - numExistingVolumes
		nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes = numNewVolumes
	}
	return true, nil, nil
}

// EBSVolumeFilter is a VolumeFilter for filtering AWS ElasticBlockStore Volumes.
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

	MatchProvisioner: func(sc *storage.StorageClass) (relevant bool) {
		if sc.Provisioner == csilibplugins.AWSEBSInTreePluginName {
			return true
		}
		return false
	},

	IsMigrated: func(csiNode *storage.CSINode) bool {
		return isCSIMigrationOn(csiNode, csilibplugins.AWSEBSInTreePluginName)
	},
}

// GCEPDVolumeFilter is a VolumeFilter for filtering GCE PersistentDisk Volumes.
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

	MatchProvisioner: func(sc *storage.StorageClass) (relevant bool) {
		if sc.Provisioner == csilibplugins.GCEPDInTreePluginName {
			return true
		}
		return false
	},

	IsMigrated: func(csiNode *storage.CSINode) bool {
		return isCSIMigrationOn(csiNode, csilibplugins.GCEPDInTreePluginName)
	},
}

// AzureDiskVolumeFilter is a VolumeFilter for filtering Azure Disk Volumes.
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

	MatchProvisioner: func(sc *storage.StorageClass) (relevant bool) {
		if sc.Provisioner == csilibplugins.AzureDiskInTreePluginName {
			return true
		}
		return false
	},

	IsMigrated: func(csiNode *storage.CSINode) bool {
		return isCSIMigrationOn(csiNode, csilibplugins.AzureDiskInTreePluginName)
	},
}

// CinderVolumeFilter is a VolumeFilter for filtering Cinder Volumes.
// It will be deprecated once Openstack cloudprovider has been removed from in-tree.
var CinderVolumeFilter = VolumeFilter{
	FilterVolume: func(vol *v1.Volume) (string, bool) {
		if vol.Cinder != nil {
			return vol.Cinder.VolumeID, true
		}
		return "", false
	},

	FilterPersistentVolume: func(pv *v1.PersistentVolume) (string, bool) {
		if pv.Spec.Cinder != nil {
			return pv.Spec.Cinder.VolumeID, true
		}
		return "", false
	},

	MatchProvisioner: func(sc *storage.StorageClass) (relevant bool) {
		if sc.Provisioner == csilibplugins.CinderInTreePluginName {
			return true
		}
		return false
	},

	IsMigrated: func(csiNode *storage.CSINode) bool {
		return isCSIMigrationOn(csiNode, csilibplugins.CinderInTreePluginName)
	},
}

// VolumeZoneChecker contains information to check the volume zone for a predicate.
type VolumeZoneChecker struct {
	pvLister  corelisters.PersistentVolumeLister
	pvcLister corelisters.PersistentVolumeClaimLister
	scLister  storagelisters.StorageClassLister
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
func NewVolumeZonePredicate(pvLister corelisters.PersistentVolumeLister, pvcLister corelisters.PersistentVolumeClaimLister, scLister storagelisters.StorageClassLister) FitPredicate {
	c := &VolumeZoneChecker{
		pvLister:  pvLister,
		pvcLister: pvcLister,
		scLister:  scLister,
	}
	return c.predicate
}

func (c *VolumeZoneChecker) predicate(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
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
		if k != v1.LabelZoneFailureDomain && k != v1.LabelZoneRegion {
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
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		if pvcName == "" {
			return false, nil, fmt.Errorf("PersistentVolumeClaim had no name")
		}
		pvc, err := c.pvcLister.PersistentVolumeClaims(namespace).Get(pvcName)
		if err != nil {
			return false, nil, err
		}

		if pvc == nil {
			return false, nil, fmt.Errorf("PersistentVolumeClaim was not found: %q", pvcName)
		}

		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			scName := v1helper.GetPersistentVolumeClaimClass(pvc)
			if len(scName) == 0 {
				return false, nil, fmt.Errorf("PersistentVolumeClaim had no pv name and storageClass name")
			}

			class, _ := c.scLister.Get(scName)
			if class == nil {
				return false, nil, fmt.Errorf("StorageClass %q claimed by PersistentVolumeClaim %q not found",
					scName, pvcName)

			}
			if class.VolumeBindingMode == nil {
				return false, nil, fmt.Errorf("VolumeBindingMode not set for StorageClass %q", scName)
			}
			if *class.VolumeBindingMode == storage.VolumeBindingWaitForFirstConsumer {
				// Skip unbound volumes
				continue
			}

			return false, nil, fmt.Errorf("PersistentVolume had no name")
		}

		pv, err := c.pvLister.Get(pvName)
		if err != nil {
			return false, nil, err
		}

		if pv == nil {
			return false, nil, fmt.Errorf("PersistentVolume was not found: %q", pvName)
		}

		for k, v := range pv.ObjectMeta.Labels {
			if k != v1.LabelZoneFailureDomain && k != v1.LabelZoneRegion {
				continue
			}
			nodeV, _ := nodeConstraints[k]
			volumeVSet, err := volumehelpers.LabelZonesToSet(v)
			if err != nil {
				klog.Warningf("Failed to parse label for %q: %q. Ignoring the label. err=%v. ", k, v, err)
				continue
			}

			if !volumeVSet.Has(nodeV) {
				klog.V(10).Infof("Won't schedule pod %q onto node %q due to volume %q (mismatch on %q)", pod.Name, node.Name, pvName, k)
				return false, []PredicateFailureReason{ErrVolumeZoneConflict}, nil
			}
		}
	}

	return true, nil, nil
}

// GetResourceRequest returns a *schedulernodeinfo.Resource that covers the largest
// width in each resource dimension. Because init-containers run sequentially, we collect
// the max in each dimension iteratively. In contrast, we sum the resource vectors for
// regular containers since they run simultaneously.
//
// If Pod Overhead is specified and the feature gate is set, the resources defined for Overhead
// are added to the calculated Resource request sum
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
func GetResourceRequest(pod *v1.Pod) *schedulernodeinfo.Resource {
	result := &schedulernodeinfo.Resource{}
	for _, container := range pod.Spec.Containers {
		result.Add(container.Resources.Requests)
	}

	// take max_resource(sum_pod, any_init_container)
	for _, container := range pod.Spec.InitContainers {
		result.SetMaxResource(container.Resources.Requests)
	}

	// If Overhead is being utilized, add to the total requests for the pod
	if pod.Spec.Overhead != nil && utilfeature.DefaultFeatureGate.Enabled(features.PodOverhead) {
		result.Add(pod.Spec.Overhead)
	}

	return result
}

func podName(pod *v1.Pod) string {
	return pod.Namespace + "/" + pod.Name
}

// PodFitsResources is a wrapper around PodFitsResourcesPredicate that implements FitPredicate interface.
// TODO(#85822): remove this function once predicate registration logic is deleted.
func PodFitsResources(pod *v1.Pod, _ Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	return PodFitsResourcesPredicate(pod, nil, nil, nodeInfo)
}

// PodFitsResourcesPredicate checks if a node has sufficient resources, such as cpu, memory, gpu, opaque int resources etc to run a pod.
// First return value indicates whether a node has sufficient resources to run a pod while the second return value indicates the
// predicate failure reasons if the node has insufficient resources to run the pod
func PodFitsResourcesPredicate(pod *v1.Pod, podRequest *schedulernodeinfo.Resource, ignoredExtendedResources sets.String, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	var predicateFails []PredicateFailureReason
	allowedPodNumber := nodeInfo.AllowedPodNumber()
	if len(nodeInfo.Pods())+1 > allowedPodNumber {
		predicateFails = append(predicateFails, NewInsufficientResourceError(v1.ResourcePods, 1, int64(len(nodeInfo.Pods())), int64(allowedPodNumber)))
	}

	if ignoredExtendedResources == nil {
		ignoredExtendedResources = sets.NewString()
	}

	if podRequest == nil {
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

	if klog.V(10) && len(predicateFails) == 0 {
		// We explicitly don't do klog.V(10).Infof() to avoid computing all the parameters if this is
		// not logged. There is visible performance gain from it.
		klog.Infof("Schedule Pod %+v on Node %+v is allowed, Node is running only %v out of %v Pods.",
			podName(pod), node.Name, len(nodeInfo.Pods()), allowedPodNumber)
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

// PodMatchesNodeSelectorAndAffinityTerms checks whether the pod is schedulable onto nodes according to
// the requirements in both NodeAffinity and nodeSelector.
func PodMatchesNodeSelectorAndAffinityTerms(pod *v1.Pod, node *v1.Node) bool {
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
		// 	klog.V(10).Infof("Match for RequiredDuringSchedulingRequiredDuringExecution node selector terms %+v", nodeSelectorTerms)
		// 	nodeAffinityMatches = nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		// }

		// Match node selector for requiredDuringSchedulingIgnoredDuringExecution.
		if nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution != nil {
			nodeSelectorTerms := nodeAffinity.RequiredDuringSchedulingIgnoredDuringExecution.NodeSelectorTerms
			klog.V(10).Infof("Match for RequiredDuringSchedulingIgnoredDuringExecution node selector terms %+v", nodeSelectorTerms)
			nodeAffinityMatches = nodeAffinityMatches && nodeMatchesNodeSelectorTerms(node, nodeSelectorTerms)
		}

	}
	return nodeAffinityMatches
}

// PodMatchNodeSelector checks if a pod node selector matches the node label.
func PodMatchNodeSelector(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}
	if PodMatchesNodeSelectorAndAffinityTerms(pod, node) {
		return true, nil, nil
	}
	return false, []PredicateFailureReason{ErrNodeSelectorNotMatch}, nil
}

// PodFitsHost checks if a pod spec node name matches the current node.
func PodFitsHost(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
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
	return false, []PredicateFailureReason{ErrPodNotMatchHostName}, nil
}

// PodFitsHostPorts is a wrapper around PodFitsHostPortsPredicate. This is needed until
// we are able to get rid of the FitPredicate function signature.
// TODO(#85822): remove this function once predicate registration logic is deleted.
func PodFitsHostPorts(pod *v1.Pod, _ Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	return PodFitsHostPortsPredicate(pod, nil, nodeInfo)
}

// PodFitsHostPortsPredicate checks if a node has free ports for the requested pod ports.
func PodFitsHostPortsPredicate(pod *v1.Pod, meta []*v1.ContainerPort, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	wantPorts := meta
	if wantPorts == nil {
		// Fallback to computing it.
		wantPorts = schedutil.GetContainerPorts(pod)
	}
	if len(wantPorts) == 0 {
		return true, nil, nil
	}

	existingPorts := nodeInfo.UsedPorts()

	// try to see whether existingPorts and  wantPorts will conflict or not
	if portsConflict(existingPorts, wantPorts) {
		return false, []PredicateFailureReason{ErrPodNotFitsHostPorts}, nil
	}

	return true, nil, nil
}

// haveOverlap searches two arrays and returns true if they have at least one common element; returns false otherwise.
func haveOverlap(a1, a2 []string) bool {
	if len(a1) > len(a2) {
		a1, a2 = a2, a1
	}
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

// GeneralPredicates checks a group of predicates that the kubelet cares about.
// DEPRECATED: this exist only because kubelet uses it. We should change kubelet to execute the individual predicates it requires.
func GeneralPredicates(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	var predicateFails []PredicateFailureReason
	for _, predicate := range []FitPredicate{PodFitsResources, PodFitsHost, PodFitsHostPorts, PodMatchNodeSelector} {
		fit, reasons, err := predicate(pod, meta, nodeInfo)
		if err != nil {
			return false, predicateFails, err
		}
		if !fit {
			predicateFails = append(predicateFails, reasons...)
		}
	}

	return len(predicateFails) == 0, predicateFails, nil
}

// CheckNodeUnschedulablePredicate checks if a pod can be scheduled on a node with Unschedulable spec.
func CheckNodeUnschedulablePredicate(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		return false, []PredicateFailureReason{ErrNodeUnknownCondition}, nil
	}

	// If pod tolerate unschedulable taint, it's also tolerate `node.Spec.Unschedulable`.
	podToleratesUnschedulable := v1helper.TolerationsTolerateTaint(pod.Spec.Tolerations, &v1.Taint{
		Key:    v1.TaintNodeUnschedulable,
		Effect: v1.TaintEffectNoSchedule,
	})

	// TODO (k82cn): deprecates `node.Spec.Unschedulable` in 1.13.
	if nodeInfo.Node().Spec.Unschedulable && !podToleratesUnschedulable {
		return false, []PredicateFailureReason{ErrNodeUnschedulable}, nil
	}

	return true, nil, nil
}

// PodToleratesNodeTaints checks if a pod tolerations can tolerate the node taints.
func PodToleratesNodeTaints(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	if nodeInfo == nil || nodeInfo.Node() == nil {
		return false, []PredicateFailureReason{ErrNodeUnknownCondition}, nil
	}

	return podToleratesNodeTaints(pod, nodeInfo, func(t *v1.Taint) bool {
		// PodToleratesNodeTaints is only interested in NoSchedule and NoExecute taints.
		return t.Effect == v1.TaintEffectNoSchedule || t.Effect == v1.TaintEffectNoExecute
	})
}

// PodToleratesNodeNoExecuteTaints checks if a pod tolerations can tolerate the node's NoExecute taints.
func PodToleratesNodeNoExecuteTaints(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	return podToleratesNodeTaints(pod, nodeInfo, func(t *v1.Taint) bool {
		return t.Effect == v1.TaintEffectNoExecute
	})
}

func podToleratesNodeTaints(pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo, filter func(t *v1.Taint) bool) (bool, []PredicateFailureReason, error) {
	taints, err := nodeInfo.Taints()
	if err != nil {
		return false, nil, err
	}

	if v1helper.TolerationsTolerateTaintsWithFilter(pod.Spec.Tolerations, taints, filter) {
		return true, nil, nil
	}
	return false, []PredicateFailureReason{ErrTaintsTolerationsNotMatch}, nil
}

// EvenPodsSpreadPredicate is the legacy function using old path of metadata.
// DEPRECATED
func EvenPodsSpreadPredicate(pod *v1.Pod, meta Metadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	return false, nil, fmt.Errorf("this function should never be called")
}

// PodTopologySpreadPredicate checks if a pod can be scheduled on a node which satisfies
// its topologySpreadConstraints.
func PodTopologySpreadPredicate(pod *v1.Pod, meta *PodTopologySpreadMetadata, nodeInfo *schedulernodeinfo.NodeInfo) (bool, []PredicateFailureReason, error) {
	node := nodeInfo.Node()
	if node == nil {
		return false, nil, fmt.Errorf("node not found")
	}

	// nil meta is illegal.
	if meta == nil {
		// TODO(autoscaler): get it implemented.
		return false, nil, errors.New("metadata not pre-computed for PodTopologySpreadPredicate")
	}

	// However, "empty" meta is legit which tolerates every toSchedule Pod.
	if len(meta.tpPairToMatchNum) == 0 || len(meta.constraints) == 0 {
		return true, nil, nil
	}

	podLabelSet := labels.Set(pod.Labels)
	for _, c := range meta.constraints {
		tpKey := c.topologyKey
		tpVal, ok := node.Labels[c.topologyKey]
		if !ok {
			klog.V(5).Infof("node '%s' doesn't have required label '%s'", node.Name, tpKey)
			return false, []PredicateFailureReason{ErrTopologySpreadConstraintsNotMatch}, nil
		}

		selfMatchNum := int32(0)
		if c.selector.Matches(podLabelSet) {
			selfMatchNum = 1
		}

		pair := topologyPair{key: tpKey, value: tpVal}
		paths, ok := meta.tpKeyToCriticalPaths[tpKey]
		if !ok {
			// error which should not happen
			klog.Errorf("internal error: get paths from key %q of %#v", tpKey, meta.tpKeyToCriticalPaths)
			continue
		}
		// judging criteria:
		// 'existing matching num' + 'if self-match (1 or 0)' - 'global min matching num' <= 'maxSkew'
		minMatchNum := paths[0].matchNum
		matchNum := meta.tpPairToMatchNum[pair]
		skew := matchNum + selfMatchNum - minMatchNum
		if skew > c.maxSkew {
			klog.V(5).Infof("node '%s' failed spreadConstraint[%s]: matchNum(%d) + selfMatchNum(%d) - minMatchNum(%d) > maxSkew(%d)", node.Name, tpKey, matchNum, selfMatchNum, minMatchNum, c.maxSkew)
			return false, []PredicateFailureReason{ErrTopologySpreadConstraintsNotMatch}, nil
		}
	}

	return true, nil, nil
}

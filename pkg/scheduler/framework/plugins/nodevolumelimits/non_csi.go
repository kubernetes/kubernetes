/*
Copyright 2019 The Kubernetes Authors.

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

package nodevolumelimits

import (
	"context"
	"fmt"
	"os"
	"regexp"
	"strconv"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/rand"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/features"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	"k8s.io/kubernetes/pkg/scheduler/nodeinfo"
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
)

const (
	// defaultMaxGCEPDVolumes defines the maximum number of PD Volumes for GCE.
	// GCE instances can have up to 16 PD volumes attached.
	defaultMaxGCEPDVolumes = 16
	// defaultMaxAzureDiskVolumes defines the maximum number of PD Volumes for Azure.
	// Larger Azure VMs can actually have much more disks attached.
	// TODO We should determine the max based on VM size
	defaultMaxAzureDiskVolumes = 16

	// ebsVolumeFilterType defines the filter name for ebsVolumeFilter.
	ebsVolumeFilterType = "EBS"
	// gcePDVolumeFilterType defines the filter name for gcePDVolumeFilter.
	gcePDVolumeFilterType = "GCE"
	// azureDiskVolumeFilterType defines the filter name for azureDiskVolumeFilter.
	azureDiskVolumeFilterType = "AzureDisk"
	// cinderVolumeFilterType defines the filter name for cinderVolumeFilter.
	cinderVolumeFilterType = "Cinder"

	// ErrReasonMaxVolumeCountExceeded is used for MaxVolumeCount predicate error.
	ErrReasonMaxVolumeCountExceeded = "node(s) exceed max volume count"

	// KubeMaxPDVols defines the maximum number of PD Volumes per kubelet.
	KubeMaxPDVols = "KUBE_MAX_PD_VOLS"
)

// AzureDiskName is the name of the plugin used in the plugin registry and configurations.
const AzureDiskName = "AzureDiskLimits"

// NewAzureDisk returns function that initializes a new plugin and returns it.
func NewAzureDisk(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	return newNonCSILimitsWithInformerFactory(azureDiskVolumeFilterType, informerFactory), nil
}

// CinderName is the name of the plugin used in the plugin registry and configurations.
const CinderName = "CinderLimits"

// NewCinder returns function that initializes a new plugin and returns it.
func NewCinder(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	return newNonCSILimitsWithInformerFactory(cinderVolumeFilterType, informerFactory), nil
}

// EBSName is the name of the plugin used in the plugin registry and configurations.
const EBSName = "EBSLimits"

// NewEBS returns function that initializes a new plugin and returns it.
func NewEBS(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	return newNonCSILimitsWithInformerFactory(ebsVolumeFilterType, informerFactory), nil
}

// GCEPDName is the name of the plugin used in the plugin registry and configurations.
const GCEPDName = "GCEPDLimits"

// NewGCEPD returns function that initializes a new plugin and returns it.
func NewGCEPD(_ *runtime.Unknown, handle framework.FrameworkHandle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	return newNonCSILimitsWithInformerFactory(gcePDVolumeFilterType, informerFactory), nil
}

// nonCSILimits contains information to check the max number of volumes for a plugin.
type nonCSILimits struct {
	name           string
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

var _ framework.FilterPlugin = &nonCSILimits{}

// newNonCSILimitsWithInformerFactory returns a plugin with filter name and informer factory.
func newNonCSILimitsWithInformerFactory(
	filterName string,
	informerFactory informers.SharedInformerFactory,
) framework.Plugin {
	pvLister := informerFactory.Core().V1().PersistentVolumes().Lister()
	pvcLister := informerFactory.Core().V1().PersistentVolumeClaims().Lister()
	scLister := informerFactory.Storage().V1().StorageClasses().Lister()

	return newNonCSILimits(filterName, getCSINodeListerIfEnabled(informerFactory), scLister, pvLister, pvcLister)
}

// newNonCSILimits creates a plugin which evaluates whether a pod can fit based on the
// number of volumes which match a filter that it requests, and those that are already present.
//
// DEPRECATED
// All cloudprovider specific predicates defined here are deprecated in favour of CSI volume limit
// predicate - MaxCSIVolumeCountPred.
//
// The predicate looks for both volumes used directly, as well as PVC volumes that are backed by relevant volume
// types, counts the number of unique volumes, and rejects the new pod if it would place the total count over
// the maximum.
func newNonCSILimits(
	filterName string,
	csiNodeLister storagelisters.CSINodeLister,
	scLister storagelisters.StorageClassLister,
	pvLister corelisters.PersistentVolumeLister,
	pvcLister corelisters.PersistentVolumeClaimLister,
) framework.Plugin {
	var filter VolumeFilter
	var volumeLimitKey v1.ResourceName
	var name string

	switch filterName {
	case ebsVolumeFilterType:
		name = EBSName
		filter = ebsVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.EBSVolumeLimitKey)
	case gcePDVolumeFilterType:
		name = GCEPDName
		filter = gcePDVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.GCEVolumeLimitKey)
	case azureDiskVolumeFilterType:
		name = AzureDiskName
		filter = azureDiskVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.AzureVolumeLimitKey)
	case cinderVolumeFilterType:
		name = CinderName
		filter = cinderVolumeFilter
		volumeLimitKey = v1.ResourceName(volumeutil.CinderVolumeLimitKey)
	default:
		klog.Fatalf("Wrong filterName, Only Support %v %v %v %v", ebsVolumeFilterType,
			gcePDVolumeFilterType, azureDiskVolumeFilterType, cinderVolumeFilterType)
		return nil
	}
	pl := &nonCSILimits{
		name:                 name,
		filter:               filter,
		volumeLimitKey:       volumeLimitKey,
		maxVolumeFunc:        getMaxVolumeFunc(filterName),
		csiNodeLister:        csiNodeLister,
		pvLister:             pvLister,
		pvcLister:            pvcLister,
		scLister:             scLister,
		randomVolumeIDPrefix: rand.String(32),
	}

	return pl
}

// Name returns name of the plugin. It is used in logs, etc.
func (pl *nonCSILimits) Name() string {
	return pl.name
}

// Filter invoked at the filter extension point.
func (pl *nonCSILimits) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *nodeinfo.NodeInfo) *framework.Status {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return nil
	}

	newVolumes := make(map[string]bool)
	if err := pl.filterVolumes(pod.Spec.Volumes, pod.Namespace, newVolumes); err != nil {
		return framework.NewStatus(framework.Error, err.Error())
	}

	// quick return
	if len(newVolumes) == 0 {
		return nil
	}

	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, fmt.Sprintf("node not found"))
	}

	var csiNode *storage.CSINode
	var err error
	if pl.csiNodeLister != nil {
		csiNode, err = pl.csiNodeLister.Get(node.Name)
		if err != nil {
			// we don't fail here because the CSINode object is only necessary
			// for determining whether the migration is enabled or not
			klog.V(5).Infof("Could not get a CSINode object for the node: %v", err)
		}
	}

	// if a plugin has been migrated to a CSI driver, defer to the CSI predicate
	if pl.filter.IsMigrated(csiNode) {
		return nil
	}

	// count unique volumes
	existingVolumes := make(map[string]bool)
	for _, existingPod := range nodeInfo.Pods() {
		if err := pl.filterVolumes(existingPod.Spec.Volumes, existingPod.Namespace, existingVolumes); err != nil {
			return framework.NewStatus(framework.Error, err.Error())
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
	maxAttachLimit := pl.maxVolumeFunc(node)
	volumeLimits := nodeInfo.VolumeLimits()
	if maxAttachLimitFromAllocatable, ok := volumeLimits[pl.volumeLimitKey]; ok {
		maxAttachLimit = int(maxAttachLimitFromAllocatable)
	}

	if numExistingVolumes+numNewVolumes > maxAttachLimit {
		// violates MaxEBSVolumeCount or MaxGCEPDVolumeCount
		return framework.NewStatus(framework.Unschedulable, ErrReasonMaxVolumeCountExceeded)
	}
	if nodeInfo != nil && nodeInfo.TransientInfo != nil && utilfeature.DefaultFeatureGate.Enabled(features.BalanceAttachedNodeVolumes) {
		nodeInfo.TransientInfo.TransientLock.Lock()
		defer nodeInfo.TransientInfo.TransientLock.Unlock()
		nodeInfo.TransientInfo.TransNodeInfo.AllocatableVolumesCount = maxAttachLimit - numExistingVolumes
		nodeInfo.TransientInfo.TransNodeInfo.RequestedVolumes = numNewVolumes
	}
	return nil
}

func (pl *nonCSILimits) filterVolumes(volumes []v1.Volume, namespace string, filteredVolumes map[string]bool) error {
	for i := range volumes {
		vol := &volumes[i]
		if id, ok := pl.filter.FilterVolume(vol); ok {
			filteredVolumes[id] = true
		} else if vol.PersistentVolumeClaim != nil {
			pvcName := vol.PersistentVolumeClaim.ClaimName
			if pvcName == "" {
				return fmt.Errorf("PersistentVolumeClaim had no name")
			}

			// Until we know real ID of the volume use namespace/pvcName as substitute
			// with a random prefix (calculated and stored inside 'c' during initialization)
			// to avoid conflicts with existing volume IDs.
			pvID := fmt.Sprintf("%s-%s/%s", pl.randomVolumeIDPrefix, namespace, pvcName)

			pvc, err := pl.pvcLister.PersistentVolumeClaims(namespace).Get(pvcName)
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
				if pl.matchProvisioner(pvc) {
					klog.V(4).Infof("PVC %s/%s is not bound, assuming PVC matches predicate when counting limits", namespace, pvcName)
					filteredVolumes[pvID] = true
				}
				continue
			}

			pv, err := pl.pvLister.Get(pvName)
			if err != nil || pv == nil {
				// If the PV is invalid and PVC belongs to the running predicate,
				// log the error and count the PV towards the PV limit.
				if pl.matchProvisioner(pvc) {
					klog.V(4).Infof("Unable to look up PV info for %s/%s/%s, assuming PV matches predicate when counting limits: %v", namespace, pvcName, pvName, err)
					filteredVolumes[pvID] = true
				}
				continue
			}

			if id, ok := pl.filter.FilterPersistentVolume(pv); ok {
				filteredVolumes[id] = true
			}
		}
	}

	return nil
}

// matchProvisioner helps identify if the given PVC belongs to the running predicate.
func (pl *nonCSILimits) matchProvisioner(pvc *v1.PersistentVolumeClaim) bool {
	if pvc.Spec.StorageClassName == nil {
		return false
	}

	storageClass, err := pl.scLister.Get(*pvc.Spec.StorageClassName)
	if err != nil || storageClass == nil {
		return false
	}

	return pl.filter.MatchProvisioner(storageClass)
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

// ebsVolumeFilter is a VolumeFilter for filtering AWS ElasticBlockStore Volumes.
var ebsVolumeFilter = VolumeFilter{
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

// gcePDVolumeFilter is a VolumeFilter for filtering gce PersistentDisk Volumes.
var gcePDVolumeFilter = VolumeFilter{
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

// azureDiskVolumeFilter is a VolumeFilter for filtering azure Disk Volumes.
var azureDiskVolumeFilter = VolumeFilter{
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

// cinderVolumeFilter is a VolumeFilter for filtering cinder Volumes.
// It will be deprecated once Openstack cloudprovider has been removed from in-tree.
var cinderVolumeFilter = VolumeFilter{
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
		case ebsVolumeFilterType:
			return getMaxEBSVolume(nodeInstanceType)
		case gcePDVolumeFilterType:
			return defaultMaxGCEPDVolumes
		case azureDiskVolumeFilterType:
			return defaultMaxAzureDiskVolumes
		case cinderVolumeFilterType:
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

// getCSINodeListerIfEnabled returns the CSINode lister or nil if the feature is disabled
func getCSINodeListerIfEnabled(factory informers.SharedInformerFactory) storagelisters.CSINodeLister {
	if !utilfeature.DefaultFeatureGate.Enabled(kubefeatures.CSINodeInfo) {
		return nil
	}
	return factory.Storage().V1().CSINodes().Lister()
}

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

package volumezone

import (
	"context"
	"fmt"

	v1 "k8s.io/api/core/v1"
	storage "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	dynamic "k8s.io/client-go/dynamic/dynamiclister"
	corelisters "k8s.io/client-go/listers/core/v1"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	cache "k8s.io/client-go/tools/cache"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	storagehelpers "k8s.io/component-helpers/storage/volume"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

// VolumeZone is a plugin that checks volume zone.
type VolumeZone struct {
	pvLister  corelisters.PersistentVolumeLister
	pvcLister corelisters.PersistentVolumeClaimLister
	scLister  storagelisters.StorageClassLister
	snapshotLister dynamic.Lister
	snapshotContentLister dynamic.Lister
}

var snapshotGVR = schema.GroupVersionResource{
	Group:   "snapshot.storage.k8s.io",
	Version: "v1",
	Resource:    "VolumeSnapshots",
}

var snapshotContentGVR = schema.GroupVersionResource{
	Group:   "snapshot.storage.k8s.io",
	Version: "v1",
	Resource:    "VolumeSnapshotContents",
}

var _ framework.FilterPlugin = &VolumeZone{}
var _ framework.EnqueueExtensions = &VolumeZone{}

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.VolumeZone

	// ErrReasonConflict is used for NoVolumeZoneConflict predicate error.
	ErrReasonConflict = "node(s) had no available volume zone"

	//boundVolumeSnapshotContentName is the name of the volume snapshot content we need to inspect to get the node label from.
	boundVolumeSnapshotContentName = "BoundVolumeSnapshotContentName"

	labels = "labels"

	// volumeSnapshotContentManagedByLabel is applied by the snapshot controller to the VolumeSnapshotContent object in case distributed snapshotting is enabled.
	// The value contains the name of the node that handles the snapshot for the volume local to that node.
	volumeSnapshotContentManagedByLabel = "snapshot.storage.kubernetes.io/managed-by"
)

var volumeZoneLabels = sets.NewString(
	v1.LabelFailureDomainBetaZone,
	v1.LabelFailureDomainBetaRegion,
	v1.LabelTopologyZone,
	v1.LabelTopologyRegion,
)

// Name returns name of the plugin. It is used in logs, etc.
func (pl *VolumeZone) Name() string {
	return Name
}

// Filter invoked at the filter extension point.
//
// It evaluates if a pod can fit due to the volumes it requests, given
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
func (pl *VolumeZone) Filter(ctx context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	// If a pod doesn't have any volume attached to it, the predicate will always be true.
	// Thus we make a fast path for it, to avoid unnecessary computations in this case.
	if len(pod.Spec.Volumes) == 0 {
		return nil
	}
	node := nodeInfo.Node()
	if node == nil {
		return framework.NewStatus(framework.Error, "node not found")
	}
	nodeConstraints := make(map[string]string)
	for k, v := range node.ObjectMeta.Labels {
		if !volumeZoneLabels.Has(k) {
			continue
		}
		nodeConstraints[k] = v
	}
	if len(nodeConstraints) == 0 {
		// The node has no zone constraints, so we're OK to schedule.
		// In practice, when using zones, all nodes must be labeled with zone labels.
		// We want to fast-path this case though.
		return nil
	}

	for i := range pod.Spec.Volumes {
		volume := pod.Spec.Volumes[i]
		if volume.PersistentVolumeClaim == nil {
			continue
		}
		pvcName := volume.PersistentVolumeClaim.ClaimName
		if pvcName == "" {
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no name")
		}
		pvc, err := pl.pvcLister.PersistentVolumeClaims(pod.Namespace).Get(pvcName)
		if s := getErrorAsStatus(err); !s.IsSuccess() {
			return s
		}

		pvName := pvc.Spec.VolumeName
		if pvName == "" {
			scName := storagehelpers.GetPersistentVolumeClaimClass(pvc)
			if len(scName) == 0 {
				return framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolumeClaim had no pv name and storageClass name")
			}

			class, err := pl.scLister.Get(scName)
			if s := getErrorAsStatus(err); !s.IsSuccess() {
				return s
			}
			if class.VolumeBindingMode == nil {
				return framework.NewStatus(framework.UnschedulableAndUnresolvable, fmt.Sprintf("VolumeBindingMode not set for StorageClass %q", scName))
			}
			if *class.VolumeBindingMode == storage.VolumeBindingWaitForFirstConsumer {
				if sourcePvName, err := pl.getPVFromPVCDatasource(pvc); err != nil {
					if s := getErrorAsStatus(err); !s.IsSuccess() {
						return s
					}
				} else if sourcePvName != nil {
					if status := pl.verifyPVLabelZones(*sourcePvName, nodeConstraints, pod, node); status != nil {
						return status
					}
				}
				if pl.sourceSnapshotOnNode(pvc, node); err != nil {
					if s := getErrorAsStatus(err); !s.IsSuccess() {
						return s
					}
				}

				// Skip unbound volumes without datasource or datasourceRef set.
				continue
			}

			return framework.NewStatus(framework.UnschedulableAndUnresolvable, "PersistentVolume had no name")
		}

		if status := pl.verifyPVLabelZones(pvName, nodeConstraints, pod, node); status != nil {
			return status
		}
	}
	return nil
}

func (pl *VolumeZone) verifyPVLabelZones(pvName string, nodeConstraints map[string]string, pod *v1.Pod, node *v1.Node) *framework.Status {
	pv, err := pl.pvLister.Get(pvName)
	if s := getErrorAsStatus(err); !s.IsSuccess() {
		return s
	}

	for k, v := range pv.ObjectMeta.Labels {
		if !volumeZoneLabels.Has(k) {
			continue
		}
		nodeV := nodeConstraints[k]
		volumeVSet, err := volumehelpers.LabelZonesToSet(v)
		if err != nil {
			klog.InfoS("Failed to parse label, ignoring the label", "label", fmt.Sprintf("%s:%s", k, v), "err", err)
			continue
		}

		if !volumeVSet.Has(nodeV) {
			klog.V(10).InfoS("Won't schedule pod onto node due to volume (mismatch on label key)", "pod", klog.KObj(pod), "node", klog.KObj(node), "PV", klog.KRef("", pvName), "PVLabelKey", k)
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, ErrReasonConflict)
		}
	}
	return nil
}

func getErrorAsStatus(err error) *framework.Status {
	if err != nil {
		if errors.IsNotFound(err) {
			return framework.NewStatus(framework.UnschedulableAndUnresolvable, err.Error())
		}
		return framework.AsStatus(err)
	}
	return nil
}

// EventsToRegister returns the possible events that may make a Pod
// failed by this plugin schedulable.
func (pl *VolumeZone) EventsToRegister() []framework.ClusterEvent {
	return []framework.ClusterEvent{
		// New storageClass with bind mode `VolumeBindingWaitForFirstConsumer` will make a pod schedulable.
		// Due to immutable field `storageClass.volumeBindingMode`, storageClass update events are ignored.
		{Resource: framework.StorageClass, ActionType: framework.Add},
		// A new node or updating a node's volume zone labels may make a pod schedulable.
		{Resource: framework.Node, ActionType: framework.Add | framework.UpdateNodeLabel},
		// A new pvc may make a pod schedulable.
		// Due to fields are immutable except `spec.resources`, pvc update events are ignored.
		{Resource: framework.PersistentVolumeClaim, ActionType: framework.Add},
		// A new pv or updating a pv's volume zone labels may make a pod shedulable.
		{Resource: framework.PersistentVolume, ActionType: framework.Add | framework.Update},
	}
}

// New initializes a new plugin and returns it.
func New(_ runtime.Object, handle framework.Handle) (framework.Plugin, error) {
	informerFactory := handle.SharedInformerFactory()
	pvLister := informerFactory.Core().V1().PersistentVolumes().Lister()
	pvcLister := informerFactory.Core().V1().PersistentVolumeClaims().Lister()
	scLister := informerFactory.Storage().V1().StorageClasses().Lister()
	snapshotIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	snapshotLister := dynamic.New(snapshotIndexer, snapshotGVR)
	snapshotContentIndexer := cache.NewIndexer(cache.MetaNamespaceKeyFunc, cache.Indexers{})
	snapshotContentLister := dynamic.New(snapshotContentIndexer, snapshotContentGVR)
	return &VolumeZone{
		pvLister,
		pvcLister,
		scLister,
		snapshotLister,
		snapshotContentLister,
	}, nil
}

func (pl *VolumeZone) getPVFromPVCDatasource(pvc *v1.PersistentVolumeClaim) (*string, error) {
	if pvc.Spec.DataSource != nil {
		return pl.getDataSourcePVFromObjectReference(pvc.Namespace, pvc.Spec.DataSource)
	} else if pvc.Spec.DataSourceRef != nil {
		return pl.getDataSourcePVFromObjectReference(pvc.Namespace, pvc.Spec.DataSourceRef)
	}
	return nil, nil
}

func (pl *VolumeZone) getDataSourcePVFromObjectReference(namespace string, obj *v1.TypedLocalObjectReference) (*string, error) {
	if obj != nil && obj.APIGroup == &v1.SchemeGroupVersion.Group && obj.Kind == string(framework.PersistentVolumeClaim) {
		sourcePvc, err := pl.pvcLister.PersistentVolumeClaims(namespace).Get(obj.Name)
		if err != nil {
			return nil, err
		}
		pvName := sourcePvc.Spec.VolumeName
		if pvName == "" {
			return nil, nil
		}
		return &pvName, nil
	}
	return nil, nil
}

func (pl *VolumeZone) sourceSnapshotOnNode(pvc *v1.PersistentVolumeClaim, node *v1.Node) error {
	if pvc.Spec.DataSource != nil {
		return pl.sourceSnapshotOnNodeFromObjectReference(pvc.Namespace, pvc.Spec.DataSource, node)
	} else if pvc.Spec.DataSourceRef != nil {
		return pl.sourceSnapshotOnNodeFromObjectReference(pvc.Namespace, pvc.Spec.DataSourceRef, node)
	}
	return nil
}

func (pl *VolumeZone) sourceSnapshotOnNodeFromObjectReference(namespace string, obj *v1.TypedLocalObjectReference, node *v1.Node) error {
	if obj != nil && obj.APIGroup == &snapshotGVR.Group && obj.Kind == "VolumeSnapshot" {
		// Lookup snapshot so we can find the associated snapshot content name
		snapshot, err := pl.snapshotLister.Namespace(namespace).Get(obj.Name)
		if err != nil {
			return err
		}
		snapshotContentName, bound, err := unstructured.NestedString(snapshot.UnstructuredContent(), boundVolumeSnapshotContentName)
		if !bound {
			// snapshot not bound to content yet, so we can ignore
			return nil
		} else if err != nil {
			return err
		}
		// Look up snapshot content so we can find the label set in PR https://github.com/kubernetes-csi/external-snapshotter/pull/585
		snapshotContent, err := pl.snapshotContentLister.Get(snapshotContentName)
		if err != nil {
			return err
		}
		labels, exists, err := unstructured.NestedStringMap(snapshotContent.UnstructuredContent(), labels)
		if !exists {
			return nil
		} else if err != nil {
			return err
		}
		labelNode, ok := labels[volumeSnapshotContentManagedByLabel]
		if !ok || labelNode == node.Name {
			return nil
		}
		return fmt.Errorf("snapshot content not on node %s", node.Name)
	}
	return nil
}

/*
Copyright 2018 The Kubernetes Authors.

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

// Package nodeinfomanager includes internal functions used to add/delete labels to
// kubernetes nodes for corresponding CSI drivers
package nodeinfomanager

import (
	"context"
	"encoding/json"
	goerrors "errors"
	"fmt"
	"math"
	"slices"
	"strings"
	"sync"

	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientset "k8s.io/client-go/kubernetes"
	nodeutil "k8s.io/component-helpers/node/util"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
)

const (
	// Name of node annotation that contains JSON map of driver names to node
	annotationKeyNodeID = "csi.volume.kubernetes.io/nodeid"
)

var (
	nodeKind      = v1.SchemeGroupVersion.WithKind("Node")
	updateBackoff = wait.Backoff{
		Steps:    4,
		Duration: 10 * time.Millisecond,
		Factor:   5.0,
		Jitter:   0.1,
	}
)

// NodeInfoManager contains necessary common dependencies to update node info on both
// the Node and CSINode objects.
type NodeInfoManager struct {
	nodeName        types.NodeName
	nodeID          types.UID
	volumeHost      volume.VolumeHost
	migratedPlugins map[string](func() bool)
	// lock protects changes to node.
	lock sync.Mutex
}

// If no updates is needed, the function must return the same Node object as the input.
type nodeUpdateFunc func(*v1.Node) (newNode *v1.Node, updated bool, err error)

// NewNodeInfoManager initializes NodeInfoManager
func NewNodeInfoManager(
	nodeName types.NodeName,
	volumeHost volume.VolumeHost,
	migratedPlugins map[string](func() bool)) *NodeInfoManager {
	return &NodeInfoManager{
		nodeName:        nodeName,
		volumeHost:      volumeHost,
		migratedPlugins: migratedPlugins,
	}
}

// InstallCSIDriver updates the node ID annotation in the Node object and CSIDrivers field in the
// CSINode object. If the CSINode object doesn't yet exist, it will be created.
// If multiple calls to InstallCSIDriver() are made in parallel, some calls might receive Node or
// CSINode update conflicts, which causes the function to retry the corresponding update.
// Concurrent calls are allowed, but they must not be intertwined with calls to other methods.
func (nim *NodeInfoManager) InstallCSIDriver(ctx context.Context, driverName string, driverNodeID string, maxAttachLimit int64, topology map[string]string) error {
	if driverNodeID == "" {
		return fmt.Errorf("error adding CSI driver node info: driverNodeID must not be empty")
	}

	nodeUpdateFuncs := []nodeUpdateFunc{
		updateNodeIDInNode(driverName, driverNodeID),
		updateTopologyLabels(topology),
	}

	err := nim.updateNode(ctx, nodeUpdateFuncs...)
	if err != nil {
		return fmt.Errorf("error updating Node object with CSI driver node info: %w", err)
	}

	err = nim.updateCSINode(ctx, driverName, driverNodeID, false, maxAttachLimit, topology)
	if err != nil {
		return fmt.Errorf("error updating CSINode object with CSI driver node info: %w", err)
	}

	return nil
}

func (nim *NodeInfoManager) RegisterCSIDriver(ctx context.Context, driverName string, driverNodeID string) error {
	if driverNodeID == "" {
		return fmt.Errorf("error adding CSI driver node info: driverNodeID must not be empty")
	}
	err := nim.updateCSINode(ctx, driverName, driverNodeID, true, 0, nil)
	if err != nil {
		return fmt.Errorf("error updating CSINode object with CSI driver node info: %w", err)
	}
	return nil
}

// UpdateCSIDriver updates CSIDrivers field in the CSINode object.
func (nim *NodeInfoManager) UpdateCSIDriver(ctx context.Context, driverName string, driverNodeID string, maxAttachLimit int64, topology map[string]string) error {
	err := nim.updateCSINode(ctx, driverName, driverNodeID, false, maxAttachLimit, topology)
	if err != nil {
		return fmt.Errorf("error updating CSINode object with CSI driver node info: %w", err)
	}
	return nil
}

// UninstallCSIDriver removes the node ID annotation from the Node object and CSIDrivers field from the
// CSINode object. If the CSINodeInfo object contains no CSIDrivers, it will be deleted.
// If multiple calls to UninstallCSIDriver() are made in parallel, some calls might receive Node or
// CSINode update conflicts, which causes the function to retry the corresponding update.
// Concurrent calls are allowed, but they must not be intertwined with calls to other methods.
func (nim *NodeInfoManager) UninstallCSIDriver(ctx context.Context, driverName string) error {
	err := nim.uninstallDriverFromCSINode(ctx, driverName)
	if err != nil {
		return fmt.Errorf("error uninstalling CSI driver from CSINode object %v", err)
	}

	err = nim.updateNode(ctx,
		removeNodeIDFromNode(driverName),
	)
	if err != nil {
		return fmt.Errorf("error removing CSI driver node info from Node object %v", err)
	}
	return nil
}

func (nim *NodeInfoManager) updateNode(ctx context.Context, updateFuncs ...nodeUpdateFunc) error {
	if err := retry(ctx, func(ctx context.Context) error {
		return nim.tryUpdateNode(ctx, updateFuncs...)
	}); err != nil {
		return fmt.Errorf("error updating node: %w", err)
	}
	return nil
}

// retry calls fn until it succeeds, ctx is done, or updateBackoff runs out.
// Errors of the failed attempts are appended to the returned one.
func retry(ctx context.Context, fn func(context.Context) error) error {
	var errs []error
	err := wait.ExponentialBackoffWithContext(ctx, updateBackoff, func(ctx context.Context) (bool, error) {
		if err := fn(ctx); err != nil {
			errs = append(errs, err)
			return false, nil
		}
		return true, nil
	})
	if err == nil {
		return nil
	}
	if len(errs) == 0 {
		return err
	}
	return fmt.Errorf("%w; caused by: %w", err, utilerrors.NewAggregate(errs))
}

// updateNode repeatedly attempts to update the corresponding node object
// which is modified by applying the given update functions sequentially.
// Because updateFuncs are applied sequentially, later updateFuncs should take into account
// the effects of previous updateFuncs to avoid potential conflicts. For example, if multiple
// functions update the same field, updates in the last function are persisted.
func (nim *NodeInfoManager) tryUpdateNode(ctx context.Context, updateFuncs ...nodeUpdateFunc) error {
	nim.lock.Lock()
	defer nim.lock.Unlock()

	// Retrieve the latest version of Node before attempting update, so that
	// existing changes are not overwritten.

	kubeClient := nim.volumeHost.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("error getting kube client")
	}

	nodeClient := kubeClient.CoreV1().Nodes()
	originalNode, err := nodeClient.Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return err
	}
	// Refresh cached node UID in case the Node was recreated with the same name since startup.
	nim.nodeID = originalNode.UID
	node := originalNode.DeepCopy()

	needUpdate := false
	for _, update := range updateFuncs {
		newNode, updated, err := update(node)
		if err != nil {
			return err
		}
		node = newNode
		needUpdate = needUpdate || updated
	}

	if needUpdate {
		// PatchNodeStatus can update both node's status and labels or annotations
		// Updating status by directly updating node does not work
		_, _, updateErr := nodeutil.PatchNodeStatus(kubeClient.CoreV1(), types.NodeName(node.Name), originalNode, node)
		return updateErr
	}

	return nil
}

// Guarantees the map is non-nil if no error is returned.
func buildNodeIDMapFromAnnotation(node *v1.Node) (map[string]string, error) {
	var previousAnnotationValue string
	if node.ObjectMeta.Annotations != nil {
		previousAnnotationValue =
			node.ObjectMeta.Annotations[annotationKeyNodeID]
	}

	var existingDriverMap map[string]string
	if previousAnnotationValue != "" {
		// Parse previousAnnotationValue as JSON
		if err := json.Unmarshal([]byte(previousAnnotationValue), &existingDriverMap); err != nil {
			return nil, fmt.Errorf(
				"failed to parse node's %q annotation value (%q) err=%v",
				annotationKeyNodeID,
				previousAnnotationValue,
				err)
		}
	}

	if existingDriverMap == nil {
		return make(map[string]string), nil
	}
	return existingDriverMap, nil
}

// updateNodeIDInNode returns a function that updates a Node object with the given
// Node ID information.
func updateNodeIDInNode(
	csiDriverName string,
	csiDriverNodeID string) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		existingDriverMap, err := buildNodeIDMapFromAnnotation(node)
		if err != nil {
			return nil, false, err
		}

		if val, ok := existingDriverMap[csiDriverName]; ok {
			if val == csiDriverNodeID {
				// Value already exists in node annotation, nothing more to do
				return node, false, nil
			}
		}

		// Add/update annotation value
		existingDriverMap[csiDriverName] = csiDriverNodeID
		jsonObj, err := json.Marshal(existingDriverMap)
		if err != nil {
			return nil, false, fmt.Errorf(
				"error while marshalling node ID map updated with driverName=%q, nodeID=%q: %v",
				csiDriverName,
				csiDriverNodeID,
				err)
		}

		if node.ObjectMeta.Annotations == nil {
			node.ObjectMeta.Annotations = make(map[string]string)
		}
		node.ObjectMeta.Annotations[annotationKeyNodeID] = string(jsonObj)

		return node, true, nil
	}
}

// removeNodeIDFromNode returns a function that removes node ID information matching the given
// driver name from a Node object.
func removeNodeIDFromNode(csiDriverName string) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		var previousAnnotationValue string
		if node.ObjectMeta.Annotations != nil {
			previousAnnotationValue =
				node.ObjectMeta.Annotations[annotationKeyNodeID]
		}

		if previousAnnotationValue == "" {
			return node, false, nil
		}

		// Parse previousAnnotationValue as JSON
		existingDriverMap := map[string]string{}
		if err := json.Unmarshal([]byte(previousAnnotationValue), &existingDriverMap); err != nil {
			return nil, false, fmt.Errorf(
				"failed to parse node's %q annotation value (%q) err=%v",
				annotationKeyNodeID,
				previousAnnotationValue,
				err)
		}

		if _, ok := existingDriverMap[csiDriverName]; !ok {
			// Value is already missing in node annotation, nothing more to do
			return node, false, nil
		}

		// Delete annotation value
		delete(existingDriverMap, csiDriverName)
		if len(existingDriverMap) == 0 {
			delete(node.ObjectMeta.Annotations, annotationKeyNodeID)
		} else {
			jsonObj, err := json.Marshal(existingDriverMap)
			if err != nil {
				return nil, false, fmt.Errorf(
					"failed while trying to remove key %q from node %q annotation. Existing data: %v",
					csiDriverName,
					annotationKeyNodeID,
					previousAnnotationValue)
			}

			node.ObjectMeta.Annotations[annotationKeyNodeID] = string(jsonObj)
		}

		return node, true, nil
	}
}

// updateTopologyLabels returns a function that updates labels of a Node object with the given
// topology information.
func updateTopologyLabels(topology map[string]string) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		if len(topology) == 0 {
			return node, false, nil
		}

		for k, v := range topology {
			if curVal, exists := node.Labels[k]; exists && curVal != v {
				return nil, false, fmt.Errorf("detected topology value collision: driver reported %q:%q but existing label is %q:%q", k, v, k, curVal)
			}
		}

		if node.Labels == nil {
			node.Labels = make(map[string]string)
		}
		for k, v := range topology {
			node.Labels[k] = v
		}
		return node, true, nil
	}
}

func (nim *NodeInfoManager) updateCSINode(
	ctx context.Context,
	driverName string,
	driverNodeID string,
	forController bool,
	maxAttachLimit int64,
	topology map[string]string) error {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	if err := retry(ctx, func(ctx context.Context) error {
		return nim.tryUpdateCSINode(ctx, csiKubeClient, driverName, driverNodeID, forController, maxAttachLimit, topology)
	}); err != nil {
		return fmt.Errorf("error updating CSINode: %w", err)
	}
	return nil
}

func (nim *NodeInfoManager) tryUpdateCSINode(
	ctx context.Context,
	csiKubeClient clientset.Interface,
	driverName string,
	driverNodeID string,
	forController bool,
	maxAttachLimit int64,
	topology map[string]string) error {

	nim.lock.Lock()
	defer nim.lock.Unlock()

	nodeInfo, err := csiKubeClient.StorageV1().CSINodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if nodeInfo == nil || errors.IsNotFound(err) {
		nodeInfo, err = nim.CreateCSINode(ctx)
	}
	if err != nil {
		return err
	}

	if err = nim.ensureNodeOwnsCSINode(ctx, csiKubeClient, nodeInfo); err != nil {
		return err
	}

	if forController {
		return nim.registerDriverToCSINode(ctx, nodeInfo, driverName, driverNodeID)
	} else {
		return nim.installDriverToCSINode(ctx, nodeInfo, driverName, driverNodeID, maxAttachLimit, topology)
	}
}

func (nim *NodeInfoManager) InitializeCSINodeWithAnnotation() error {
	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return goerrors.New("error getting CSI client")
	}

	err := retry(context.TODO(), func(ctx context.Context) error {
		err := nim.tryInitializeCSINodeWithAnnotation(ctx, csiKubeClient)
		if err != nil {
			klog.V(2).Infof("Failed to publish CSINode: %v", err)
		}
		return err
	})
	if err != nil {
		return fmt.Errorf("error updating CSINode annotation: %w", err)
	}

	return nil
}

func (nim *NodeInfoManager) tryInitializeCSINodeWithAnnotation(ctx context.Context, csiKubeClient clientset.Interface) error {
	nim.lock.Lock()
	defer nim.lock.Unlock()

	node, err := csiKubeClient.CoreV1().Nodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return err
	}
	nim.nodeID = node.UID

	nodeInfo, err := csiKubeClient.StorageV1().CSINodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if nodeInfo == nil || errors.IsNotFound(err) {
		// CreateCSINode will set the annotation
		_, err = nim.CreateCSINode(ctx)
		return err
	} else if err != nil {
		return err
	}

	if err = nim.ensureNodeOwnsCSINode(ctx, csiKubeClient, nodeInfo); err != nil {
		return err
	}

	annotationModified := setMigrationAnnotation(nim.migratedPlugins, nodeInfo)

	if annotationModified {
		_, err := csiKubeClient.StorageV1().CSINodes().Update(ctx, nodeInfo, metav1.UpdateOptions{})
		return err
	}
	return nil

}

// ensureNodeOwnsCSINode will ensure that the current CSINode object is owned by the node represented by this NodeInfoManager.
// If not, it will delete the existing CSINode object and return an error.
func (nim *NodeInfoManager) ensureNodeOwnsCSINode(ctx context.Context, csiKubeClient clientset.Interface, nodeInfo *storagev1.CSINode) error {
	owned, csiNodeOwnerID := nim.nodeOwnsCSINode(nodeInfo)
	if owned {
		return nil
	}

	// Re-fetch Node UID before deleting in case the Node was recreated with the
	// same name and nim.nodeID still holds a predecessor Node's UID.
	node, err := csiKubeClient.CoreV1().Nodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return err
	}
	nim.nodeID = node.UID
	if owned, _ := nim.nodeOwnsCSINode(nodeInfo); owned {
		return nil
	}

	klog.V(2).Infof("existing CSINode %q is owned by different node (oldNodeID=%q, newNodeID=%q), cleaning up...", nodeInfo.Name, csiNodeOwnerID, nim.nodeID)
	if err := nim.DeleteCSINode(ctx); err != nil {
		return fmt.Errorf("error deleting existing CSINode %q: %w", nodeInfo.Name, err)
	}
	// Returning now so that the next attempt can create a new CSINode object
	return fmt.Errorf("CSINode %q was owned by different node (oldNodeID=%q, newNodeID=%q), deleted it", nodeInfo.Name, csiNodeOwnerID, nim.nodeID)
}

// nodeOwnsCSINode checks if the CSINode object is owned by the node represented by this NodeInfoManager.
// It also returns the current owner UID.
func (nim *NodeInfoManager) nodeOwnsCSINode(nodeInfo *storagev1.CSINode) (bool, types.UID) {
	var csiNodeOwnerID types.UID
	var found bool
	for _, ownerRef := range nodeInfo.OwnerReferences {
		if ownerRef.Kind != nodeKind.Kind {
			continue
		}

		csiNodeOwnerID = ownerRef.UID
		if ownerRef.Name != string(nim.nodeName) {
			continue
		}

		if csiNodeOwnerID == nim.nodeID {
			found = true
			break
		}
	}
	return found, csiNodeOwnerID
}

// CreateCSINode creates the CSINode object of this node, owned by the Node object.
func (nim *NodeInfoManager) CreateCSINode(ctx context.Context) (*storagev1.CSINode, error) {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return nil, fmt.Errorf("error getting CSI client")
	}

	node, err := csiKubeClient.CoreV1().Nodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	nodeInfo := &storagev1.CSINode{
		ObjectMeta: metav1.ObjectMeta{
			Name: string(nim.nodeName),
			OwnerReferences: []metav1.OwnerReference{
				{
					APIVersion: nodeKind.Version,
					Kind:       nodeKind.Kind,
					Name:       node.Name,
					UID:        node.UID,
				},
			},
		},
		Spec: storagev1.CSINodeSpec{
			Drivers: []storagev1.CSINodeDriver{},
		},
	}

	setMigrationAnnotation(nim.migratedPlugins, nodeInfo)

	return csiKubeClient.StorageV1().CSINodes().Create(ctx, nodeInfo, metav1.CreateOptions{})
}

// DeleteCSINode deletes the CSINode object of this node.
func (nim *NodeInfoManager) DeleteCSINode(ctx context.Context) error {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	return csiKubeClient.StorageV1().CSINodes().Delete(ctx, string(nim.nodeName), metav1.DeleteOptions{})
}

func setMigrationAnnotation(migratedPlugins map[string](func() bool), nodeInfo *storagev1.CSINode) (modified bool) {
	if migratedPlugins == nil {
		return false
	}

	nodeInfoAnnotations := nodeInfo.GetAnnotations()
	if nodeInfoAnnotations == nil {
		nodeInfoAnnotations = map[string]string{}
	}

	var oldAnnotationSet sets.Set[string]
	mpa := nodeInfoAnnotations[v1.MigratedPluginsAnnotationKey]
	tok := strings.Split(mpa, ",")
	if len(mpa) == 0 {
		oldAnnotationSet = sets.New[string]()
	} else {
		oldAnnotationSet = sets.New[string](tok...)
	}

	newAnnotationSet := sets.New[string]()
	for pluginName, migratedFunc := range migratedPlugins {
		if migratedFunc() {
			newAnnotationSet.Insert(pluginName)
		}
	}

	if oldAnnotationSet.Equal(newAnnotationSet) {
		return false
	}

	nas := strings.Join(sets.List[string](newAnnotationSet), ",")
	if len(nas) != 0 {
		nodeInfoAnnotations[v1.MigratedPluginsAnnotationKey] = nas
	} else {
		delete(nodeInfoAnnotations, v1.MigratedPluginsAnnotationKey)
	}

	nodeInfo.Annotations = nodeInfoAnnotations
	return true
}

// Returns true if and only if new maxAttachLimit doesn't require CSINode update
func keepAllocatableCount(driverInfoSpec storagev1.CSINodeDriver, maxAttachLimit int64) bool {
	if maxAttachLimit == 0 {
		return driverInfoSpec.Allocatable == nil || driverInfoSpec.Allocatable.Count == nil
	}

	return driverInfoSpec.Allocatable != nil && driverInfoSpec.Allocatable.Count != nil && int64(*driverInfoSpec.Allocatable.Count) == maxAttachLimit
}

func (nim *NodeInfoManager) installDriverToCSINode(
	ctx context.Context,
	nodeInfo *storagev1.CSINode,
	driverName string,
	driverNodeID string,
	maxAttachLimit int64,
	topology map[string]string) error {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	topologyKeys := sets.KeySet[string, string](topology)

	specModified := true
	// Clone driver list, omitting the driver that matches the given driverName
	newDriverSpecs := []storagev1.CSINodeDriver{}
	for _, driverInfoSpec := range nodeInfo.Spec.Drivers {
		if driverInfoSpec.Name == driverName {
			if driverInfoSpec.NodeID == driverNodeID &&
				sets.New[string](driverInfoSpec.TopologyKeys...).Equal(topologyKeys) &&
				keepAllocatableCount(driverInfoSpec, maxAttachLimit) {
				specModified = false
			}
		} else {
			// Omit driverInfoSpec matching given driverName
			newDriverSpecs = append(newDriverSpecs, driverInfoSpec)
		}
	}

	// CSIControllerGetNodeInfo feature reverted,
	// move the control of this driver from external-attacher back to kubelet.
	newDriverRegs := slices.DeleteFunc(nodeInfo.Spec.DriverRegistrations, func(r storagev1.CSINodeDriverRegistration) bool {
		return r.Name == driverName
	})
	specModified = specModified || len(newDriverRegs) < len(nodeInfo.Spec.DriverRegistrations)

	annotationModified := setMigrationAnnotation(nim.migratedPlugins, nodeInfo)

	if !specModified && !annotationModified {
		return nil
	}

	// Append new driver
	driverSpec := storagev1.CSINodeDriver{
		Name:         driverName,
		NodeID:       driverNodeID,
		TopologyKeys: sets.List[string](topologyKeys),
	}

	if maxAttachLimit > 0 {
		if maxAttachLimit > math.MaxInt32 {
			klog.Warningf("Exceeded max supported attach limit value, truncating it to %d", math.MaxInt32)
			maxAttachLimit = math.MaxInt32
		}
		m := int32(maxAttachLimit)
		driverSpec.Allocatable = &storagev1.VolumeNodeResources{Count: &m}
	} else if maxAttachLimit != 0 {
		klog.Errorf("Invalid attach limit value %d cannot be added to CSINode object for %q", maxAttachLimit, driverName)
	}

	newDriverSpecs = append(newDriverSpecs, driverSpec)
	nodeInfo.Spec.Drivers = newDriverSpecs
	nodeInfo.Spec.DriverRegistrations = newDriverRegs

	_, err := csiKubeClient.StorageV1().CSINodes().Update(ctx, nodeInfo, metav1.UpdateOptions{})
	return err
}

func (nim *NodeInfoManager) registerDriverToCSINode(
	ctx context.Context,
	nodeInfo *storagev1.CSINode,
	driverName string,
	driverNodeID string) error {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	regs := nodeInfo.Spec.DriverRegistrations
	i := slices.IndexFunc(regs, func(r storagev1.CSINodeDriverRegistration) bool {
		return r.Name == driverName
	})
	annotationModified := setMigrationAnnotation(nim.migratedPlugins, nodeInfo)

	if i >= 0 && regs[i].NodeID == driverNodeID && !annotationModified {
		return nil
	}

	// Append new driver
	if i >= 0 {
		regs = append(regs[:i], regs[i+1:]...)
	}
	regs = append(regs, storagev1.CSINodeDriverRegistration{
		Name:   driverName,
		NodeID: driverNodeID,
	})
	nodeInfo.Spec.DriverRegistrations = regs

	updatedNodeInfo, err := csiKubeClient.StorageV1().CSINodes().Update(ctx, nodeInfo, metav1.UpdateOptions{})
	if err != nil {
		return err
	}
	if !slices.ContainsFunc(updatedNodeInfo.Spec.DriverRegistrations, func(r storagev1.CSINodeDriverRegistration) bool {
		return r.Name == driverName && r.NodeID == driverNodeID
	}) {
		return fmt.Errorf("CSINode %q did not retain registration for CSI driver %q with node ID %q; ensure CSIControllerGetNodeInfo is enabled on kube-apiserver",
			nodeInfo.Name, driverName, driverNodeID)
	}
	return nil
}

func (nim *NodeInfoManager) uninstallDriverFromCSINode(
	ctx context.Context,
	csiDriverName string) error {

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	if err := retry(ctx, func(ctx context.Context) error {
		return nim.tryUninstallDriverFromCSINode(ctx, csiKubeClient, csiDriverName)
	}); err != nil {
		return fmt.Errorf("error updating CSINode: %w", err)
	}
	return nil
}

func (nim *NodeInfoManager) tryUninstallDriverFromCSINode(
	ctx context.Context,
	csiKubeClient clientset.Interface,
	csiDriverName string) error {

	nim.lock.Lock()
	defer nim.lock.Unlock()

	nodeInfoClient := csiKubeClient.StorageV1().CSINodes()
	nodeInfo, err := nodeInfoClient.Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil && errors.IsNotFound(err) {
		return nil
	} else if err != nil {
		return err
	}

	// Uninstall CSINodeDriver with name csiDriverName
	drivers := slices.DeleteFunc(nodeInfo.Spec.Drivers, func(d storagev1.CSINodeDriver) bool {
		return d.Name == csiDriverName
	})
	regs := slices.DeleteFunc(nodeInfo.Spec.DriverRegistrations, func(r storagev1.CSINodeDriverRegistration) bool {
		return r.Name == csiDriverName
	})

	if len(drivers) < len(nodeInfo.Spec.Drivers) || len(regs) < len(nodeInfo.Spec.DriverRegistrations) {
		nodeInfo.Spec.Drivers = drivers
		nodeInfo.Spec.DriverRegistrations = regs
		nodeInfo, err = nodeInfoClient.Update(ctx, nodeInfo, metav1.UpdateOptions{})
		if err != nil {
			return err // do not wrap error
		}
	}

	// Update() ignores status, so the driver's storage health is removed with a
	// separate status update. On retry the spec is already clean and only this runs.
	storageHealth, statusModified := withoutStorageHealth(nodeInfo.Status.StorageHealth, csiDriverName)
	if !statusModified {
		return nil
	}
	nodeInfo.Status.StorageHealth = storageHealth
	_, err = nodeInfoClient.UpdateStatus(ctx, nodeInfo, metav1.UpdateOptions{})
	return err // do not wrap error
}

// withoutStorageHealth returns health without driverName's entry and whether
// an entry was removed.
func withoutStorageHealth(health []storagev1.StorageHealth, driverName string) ([]storagev1.StorageHealth, bool) {
	remaining := make([]storagev1.StorageHealth, 0, len(health))
	for _, h := range health {
		if h.Name != driverName {
			remaining = append(remaining, h)
		}
	}
	return remaining, len(remaining) != len(health)
}

// UpdateCSINodeStorageHealth updates CSINode.Status.StorageHealth for the given
// driver with the provided conditions. Other drivers' entries are preserved.
// No-ops if the status, reason, and volume-capability identities for this driver
// are unchanged.
func (nim *NodeInfoManager) UpdateCSINodeStorageHealth(ctx context.Context, driverName string, conditions []storagev1.StorageHealthCondition) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CSIVolumeHealth) {
		return nil
	}

	csiKubeClient := nim.volumeHost.GetKubeClient()
	if csiKubeClient == nil {
		return goerrors.New("error getting CSI client")
	}

	if err := retry(ctx, func(ctx context.Context) error {
		return nim.tryUpdateCSINodeStorageHealth(ctx, csiKubeClient, driverName, conditions)
	}); err != nil {
		return fmt.Errorf("error updating CSINode status: %w", err)
	}
	return nil
}

func (nim *NodeInfoManager) tryUpdateCSINodeStorageHealth(
	ctx context.Context,
	csiKubeClient clientset.Interface,
	driverName string,
	conditions []storagev1.StorageHealthCondition) error {

	nim.lock.Lock()
	defer nim.lock.Unlock()

	nodeInfo, err := csiKubeClient.StorageV1().CSINodes().Get(ctx, string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return err
	}

	if err = nim.ensureNodeOwnsCSINode(ctx, csiKubeClient, nodeInfo); err != nil {
		return err
	}

	// A probe that started before the driver was uninstalled must not re-add
	// its health entry. Uninstall holds nim.lock too, so this check is not racy.
	if !slices.ContainsFunc(nodeInfo.Spec.Drivers, func(d storagev1.CSINodeDriver) bool { return d.Name == driverName }) {
		klog.V(4).InfoS("Skipping storage health update for driver not installed in CSINode", "driver", driverName)
		return nil
	}

	var existingDriverConds []storagev1.StorageHealthCondition
	otherDriversHealth := make([]storagev1.StorageHealth, 0, len(nodeInfo.Status.StorageHealth))
	for _, health := range nodeInfo.Status.StorageHealth {
		if health.Name == driverName {
			existingDriverConds = health.HealthConditions
		} else {
			otherDriversHealth = append(otherDriversHealth, health)
		}
	}

	if storageHealthIdentityEqual(existingDriverConds, conditions) {
		return nil
	}

	transitionTimes := make(map[storageHealthIdentity]metav1.Time, len(existingDriverConds))
	for _, condition := range existingDriverConds {
		transitionTimes[storageHealthConditionIdentity(condition)] = condition.LastTransitionTime
	}

	combined := make([]storagev1.StorageHealth, 0, len(otherDriversHealth)+1)
	combined = append(combined, otherDriversHealth...)
	if len(conditions) > 0 {
		now := metav1.Now().Rfc3339Copy()
		updatedConditions := make([]storagev1.StorageHealthCondition, len(conditions))
		for i, condition := range conditions {
			updatedConditions[i] = *condition.DeepCopy()
			transitionTime, exists := transitionTimes[storageHealthConditionIdentity(condition)]
			if !exists {
				transitionTime = now
			}
			updatedConditions[i].LastTransitionTime = transitionTime
		}
		combined = append(combined, storagev1.StorageHealth{Name: driverName, HealthConditions: updatedConditions})
	}
	nodeInfo.Status.StorageHealth = combined
	_, err = csiKubeClient.StorageV1().CSINodes().UpdateStatus(ctx, nodeInfo, metav1.UpdateOptions{})
	return err
}

// storageHealthIdentityEqual reports whether two condition slices have the same
// set of status, reason, and volume-capability identities. Message, transition
// time, and duplicate entries are ignored so repeated or duplicate reports do
// not cause status write churn.
func storageHealthIdentityEqual(a, b []storagev1.StorageHealthCondition) bool {
	return storageHealthIdentities(a).Equal(storageHealthIdentities(b))
}

func storageHealthIdentities(conditions []storagev1.StorageHealthCondition) sets.Set[storageHealthIdentity] {
	identities := sets.New[storageHealthIdentity]()
	for _, c := range conditions {
		identities.Insert(storageHealthConditionIdentity(c))
	}
	return identities
}

type storageHealthIdentity struct {
	status, reason, accessMode, volumeMode string
}

func storageHealthConditionIdentity(condition storagev1.StorageHealthCondition) storageHealthIdentity {
	identity := storageHealthIdentity{
		status: string(condition.Status),
		reason: condition.Reason,
	}
	if condition.AccessMode != nil {
		identity.accessMode = string(*condition.AccessMode)
	}
	if condition.VolumeMode != nil {
		identity.volumeMode = string(*condition.VolumeMode)
	}
	return identity
}

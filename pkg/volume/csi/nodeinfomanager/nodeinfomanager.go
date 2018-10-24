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
package nodeinfomanager // import "k8s.io/kubernetes/pkg/volume/csi/nodeinfomanager"

import (
	"encoding/json"
	"fmt"

	csipb "github.com/container-storage-interface/spec/lib/go/csi/v0"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/util/retry"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	// Name of node annotation that contains JSON map of driver names to node
	annotationKeyNodeID = "csi.volume.kubernetes.io/nodeid"
)

var nodeKind = v1.SchemeGroupVersion.WithKind("Node")

// nodeInfoManager contains necessary common dependencies to update node info on both
// the Node and CSINodeInfo objects.
type nodeInfoManager struct {
	nodeName   types.NodeName
	volumeHost volume.VolumeHost
}

// If no updates is needed, the function must return the same Node object as the input.
type nodeUpdateFunc func(*v1.Node) (newNode *v1.Node, updated bool, err error)

// Interface implements an interface for managing labels of a node
type Interface interface {
	// Record in the cluster the given node information from the CSI driver with the given name.
	// Concurrent calls to AddNodeInfo() is allowed, but they should not be intertwined with calls
	// to other methods in this interface.
	AddNodeInfo(driverName string, driverNodeID string, maxVolumeLimit int64, topology *csipb.Topology) error

	// Remove in the cluster node information from the CSI driver with the given name.
	// Concurrent calls to RemoveNodeInfo() is allowed, but they should not be intertwined with calls
	// to other methods in this interface.
	RemoveNodeInfo(driverName string) error
}

// NewNodeInfoManager initializes nodeInfoManager
func NewNodeInfoManager(
	nodeName types.NodeName,
	volumeHost volume.VolumeHost) Interface {
	return &nodeInfoManager{
		nodeName:   nodeName,
		volumeHost: volumeHost,
	}
}

// AddNodeInfo updates the node ID annotation in the Node object and CSIDrivers field in the
// CSINodeInfo object. If the CSINodeInfo object doesn't yet exist, it will be created.
// If multiple calls to AddNodeInfo() are made in parallel, some calls might receive Node or
// CSINodeInfo update conflicts, which causes the function to retry the corresponding update.
func (nim *nodeInfoManager) AddNodeInfo(driverName string, driverNodeID string, maxAttachLimit int64, topology *csipb.Topology) error {
	if driverNodeID == "" {
		return fmt.Errorf("error adding CSI driver node info: driverNodeID must not be empty")
	}

	nodeUpdateFuncs := []nodeUpdateFunc{
		updateNodeIDInNode(driverName, driverNodeID),
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		nodeUpdateFuncs = append(nodeUpdateFuncs, updateTopologyLabels(topology))
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.AttachVolumeLimit) {
		nodeUpdateFuncs = append(nodeUpdateFuncs, updateMaxAttachLimit(driverName, maxAttachLimit))
	}

	err := nim.updateNode(nodeUpdateFuncs...)
	if err != nil {
		return fmt.Errorf("error updating Node object with CSI driver node info: %v", err)
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CSINodeInfo) {
		err = nim.updateCSINodeInfo(driverName, driverNodeID, topology)
		if err != nil {
			return fmt.Errorf("error updating CSINodeInfo object with CSI driver node info: %v", err)
		}
	}
	return nil
}

// RemoveNodeInfo removes the node ID annotation from the Node object and CSIDrivers field from the
// CSINodeInfo object. If the CSINOdeInfo object contains no CSIDrivers, it will be deleted.
// If multiple calls to RemoveNodeInfo() are made in parallel, some calls might receive Node or
// CSINodeInfo update conflicts, which causes the function to retry the corresponding update.
func (nim *nodeInfoManager) RemoveNodeInfo(driverName string) error {
	err := nim.removeCSINodeInfo(driverName)
	if err != nil {
		return fmt.Errorf("error removing CSI driver node info from CSINodeInfo object %v", err)
	}

	err = nim.updateNode(
		removeMaxAttachLimit(driverName),
		removeNodeIDFromNode(driverName),
	)
	if err != nil {
		return fmt.Errorf("error removing CSI driver node info from Node object %v", err)
	}
	return nil
}

// updateNode repeatedly attempts to update the corresponding node object
// which is modified by applying the given update functions sequentially.
// Because updateFuncs are applied sequentially, later updateFuncs should take into account
// the effects of previous updateFuncs to avoid potential conflicts. For example, if multiple
// functions update the same field, updates in the last function are persisted.
func (nim *nodeInfoManager) updateNode(updateFuncs ...nodeUpdateFunc) error {
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Retrieve the latest version of Node before attempting update, so that
		// existing changes are not overwritten. RetryOnConflict uses
		// exponential backoff to avoid exhausting the apiserver.

		kubeClient := nim.volumeHost.GetKubeClient()
		if kubeClient == nil {
			return fmt.Errorf("error getting kube client")
		}

		nodeClient := kubeClient.CoreV1().Nodes()
		node, err := nodeClient.Get(string(nim.nodeName), metav1.GetOptions{})
		if err != nil {
			return err // do not wrap error
		}

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
			_, updateErr := nodeClient.Update(node)
			return updateErr // do not wrap error
		}

		return nil
	})
	if retryErr != nil {
		return fmt.Errorf("node update failed: %v", retryErr)
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
func updateTopologyLabels(topology *csipb.Topology) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		if topology == nil || len(topology.Segments) == 0 {
			return node, false, nil
		}

		for k, v := range topology.Segments {
			if curVal, exists := node.Labels[k]; exists && curVal != v {
				return nil, false, fmt.Errorf("detected topology value collision: driver reported %q:%q but existing label is %q:%q", k, v, k, curVal)
			}
		}

		if node.Labels == nil {
			node.Labels = make(map[string]string)
		}
		for k, v := range topology.Segments {
			node.Labels[k] = v
		}
		return node, true, nil
	}
}

func (nim *nodeInfoManager) updateCSINodeInfo(
	driverName string,
	driverNodeID string,
	topology *csipb.Topology) error {

	csiKubeClient := nim.volumeHost.GetCSIClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		nodeInfo, err := csiKubeClient.CsiV1alpha1().CSINodeInfos().Get(string(nim.nodeName), metav1.GetOptions{})
		if nodeInfo == nil || errors.IsNotFound(err) {
			return nim.createNodeInfoObject(driverName, driverNodeID, topology)
		}
		if err != nil {
			return err // do not wrap error
		}

		return nim.updateNodeInfoObject(nodeInfo, driverName, driverNodeID, topology)
	})
	if retryErr != nil {
		return fmt.Errorf("CSINodeInfo update failed: %v", retryErr)
	}
	return nil
}

func (nim *nodeInfoManager) createNodeInfoObject(
	driverName string,
	driverNodeID string,
	topology *csipb.Topology) error {

	kubeClient := nim.volumeHost.GetKubeClient()
	if kubeClient == nil {
		return fmt.Errorf("error getting kube client")
	}

	csiKubeClient := nim.volumeHost.GetCSIClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	topologyKeys := []string{} // must be an empty slice instead of nil to satisfy CRD OpenAPI Schema validation
	if topology != nil {
		for k := range topology.Segments {
			topologyKeys = append(topologyKeys, k)
		}
	}

	node, err := kubeClient.CoreV1().Nodes().Get(string(nim.nodeName), metav1.GetOptions{})
	if err != nil {
		return err // do not wrap error
	}

	nodeInfo := &csiv1alpha1.CSINodeInfo{
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
		CSIDrivers: []csiv1alpha1.CSIDriverInfo{
			{
				Driver:       driverName,
				NodeID:       driverNodeID,
				TopologyKeys: topologyKeys,
			},
		},
	}

	_, err = csiKubeClient.CsiV1alpha1().CSINodeInfos().Create(nodeInfo)
	return err // do not wrap error
}

func (nim *nodeInfoManager) updateNodeInfoObject(
	nodeInfo *csiv1alpha1.CSINodeInfo,
	driverName string,
	driverNodeID string,
	topology *csipb.Topology) error {

	csiKubeClient := nim.volumeHost.GetCSIClient()
	if csiKubeClient == nil {
		return fmt.Errorf("error getting CSI client")
	}

	topologyKeys := make(sets.String)
	if topology != nil {
		for k := range topology.Segments {
			topologyKeys.Insert(k)
		}
	}

	// Clone driver list, omitting the driver that matches the given driverName,
	// unless the driver is identical to information provided, in which case no update is necessary.
	var newDriverInfos []csiv1alpha1.CSIDriverInfo
	for _, driverInfo := range nodeInfo.CSIDrivers {
		if driverInfo.Driver == driverName {
			prevTopologyKeys := sets.NewString(driverInfo.TopologyKeys...)
			if driverInfo.NodeID == driverNodeID && prevTopologyKeys.Equal(topologyKeys) {
				// No update needed
				return nil
			}
		} else {
			// Omit driverInfo matching given driverName
			newDriverInfos = append(newDriverInfos, driverInfo)
		}
	}

	// Append new driver
	driverInfo := csiv1alpha1.CSIDriverInfo{
		Driver:       driverName,
		NodeID:       driverNodeID,
		TopologyKeys: topologyKeys.List(),
	}
	newDriverInfos = append(newDriverInfos, driverInfo)
	nodeInfo.CSIDrivers = newDriverInfos

	_, err := csiKubeClient.CsiV1alpha1().CSINodeInfos().Update(nodeInfo)
	return err // do not wrap error
}

func (nim *nodeInfoManager) removeCSINodeInfo(csiDriverName string) error {
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {

		csiKubeClient := nim.volumeHost.GetCSIClient()
		if csiKubeClient == nil {
			return fmt.Errorf("error getting CSI client")
		}

		nodeInfoClient := csiKubeClient.CsiV1alpha1().CSINodeInfos()
		nodeInfo, err := nodeInfoClient.Get(string(nim.nodeName), metav1.GetOptions{})
		if nodeInfo == nil || errors.IsNotFound(err) {
			// do nothing
			return nil
		}
		if err != nil {
			return err // do not wrap error
		}

		// Remove matching driver from driver list
		var newDriverInfos []csiv1alpha1.CSIDriverInfo
		for _, driverInfo := range nodeInfo.CSIDrivers {
			if driverInfo.Driver != csiDriverName {
				newDriverInfos = append(newDriverInfos, driverInfo)
			}
		}

		if len(newDriverInfos) == len(nodeInfo.CSIDrivers) {
			// No changes, don't update
			return nil
		}

		if len(newDriverInfos) == 0 {
			// No drivers left, delete CSINodeInfo object
			return nodeInfoClient.Delete(string(nim.nodeName), &metav1.DeleteOptions{})
		}

		// TODO (verult) make sure CSINodeInfo has validation logic to prevent duplicate driver names
		_, updateErr := nodeInfoClient.Update(nodeInfo)
		return updateErr // do not wrap error

	})
	if retryErr != nil {
		return fmt.Errorf("CSINodeInfo update failed: %v", retryErr)
	}
	return nil
}

func updateMaxAttachLimit(driverName string, maxLimit int64) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		if maxLimit <= 0 {
			glog.V(4).Infof("skipping adding attach limit for %s", driverName)
			return node, false, nil
		}

		if node.Status.Capacity == nil {
			node.Status.Capacity = v1.ResourceList{}
		}
		if node.Status.Allocatable == nil {
			node.Status.Allocatable = v1.ResourceList{}
		}
		limitKeyName := util.GetCSIAttachLimitKey(driverName)
		node.Status.Capacity[v1.ResourceName(limitKeyName)] = *resource.NewQuantity(maxLimit, resource.DecimalSI)
		node.Status.Allocatable[v1.ResourceName(limitKeyName)] = *resource.NewQuantity(maxLimit, resource.DecimalSI)

		return node, true, nil
	}
}

func removeMaxAttachLimit(driverName string) nodeUpdateFunc {
	return func(node *v1.Node) (*v1.Node, bool, error) {
		limitKey := v1.ResourceName(util.GetCSIAttachLimitKey(driverName))

		capacityExists := false
		if node.Status.Capacity != nil {
			_, capacityExists = node.Status.Capacity[limitKey]
		}

		allocatableExists := false
		if node.Status.Allocatable != nil {
			_, allocatableExists = node.Status.Allocatable[limitKey]
		}

		if !capacityExists && !allocatableExists {
			return node, false, nil
		}

		delete(node.Status.Capacity, limitKey)
		if len(node.Status.Capacity) == 0 {
			node.Status.Capacity = nil
		}

		delete(node.Status.Allocatable, limitKey)
		if len(node.Status.Allocatable) == 0 {
			node.Status.Allocatable = nil
		}

		return node, true, nil
	}
}

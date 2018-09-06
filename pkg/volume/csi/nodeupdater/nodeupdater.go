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

// Package nodeupdater includes internal functions used to add/delete labels to
// kubernetes nodes for corresponding CSI drivers
package nodeupdater // import "k8s.io/kubernetes/pkg/volume/csi/nodeupdater"

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/util/retry"
	"k8s.io/kubernetes/pkg/volume/util"
)

const (
	// Name of node annotation that contains JSON map of driver names to node
	// names
	annotationKey = "csi.volume.kubernetes.io/nodeid"
)

// labelManagementStruct is struct of channels used for communication between the driver registration
// code and the go routine responsible for managing the node's labels
type nodeUpdateStruct struct {
	nodeName types.NodeName
	k8s      kubernetes.Interface
}

// Interface implements an interface for managing labels of a node
type Interface interface {
	AddLabelsAndLimits(driverName string, driverNodeId string, maxLimit int64) error
}

// NewNodeupdater initializes nodeUpdateStruct and returns available interfaces
func NewNodeUpdater(nodeName types.NodeName, kubeClient kubernetes.Interface) Interface {
	return nodeUpdateStruct{
		nodeName: nodeName,
		k8s:      kubeClient,
	}
}

// AddLabelsAndLimits nodeUpdater waits for labeling requests initiated by the driver's registration
// process and updates labels and attach limits
func (nodeUpdater nodeUpdateStruct) AddLabelsAndLimits(driverName string, driverNodeId string, maxLimit int64) error {
	err := addLabelsAndLimits(string(nodeUpdater.nodeName), nodeUpdater.k8s.CoreV1().Nodes(), driverName, driverNodeId, maxLimit)
	if err != nil {
		return err
	}
	return nil
}

func addMaxAttachLimitToNode(node *v1.Node, driverName string, maxLimit int64) *v1.Node {
	if maxLimit <= 0 {
		glog.V(4).Infof("skipping adding attach limit for %s", driverName)
		return node
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
	return node
}

// Clones the given map and returns a new map with the given key and value added.
// Returns the given map, if annotationKey is empty.
func cloneAndAddAnnotation(
	annotations map[string]string,
	annotationKey,
	annotationValue string) map[string]string {
	if annotationKey == "" {
		// Don't need to add an annotation.
		return annotations
	}
	// Clone.
	newAnnotations := map[string]string{}
	for key, value := range annotations {
		newAnnotations[key] = value
	}
	newAnnotations[annotationKey] = annotationValue
	return newAnnotations
}

func addNodeIdToNode(node *v1.Node, driverName string, csiDriverNodeId string) (*v1.Node, error) {
	var previousAnnotationValue string
	if node.ObjectMeta.Annotations != nil {
		previousAnnotationValue =
			node.ObjectMeta.Annotations[annotationKey]
		glog.V(3).Infof(
			"previousAnnotationValue=%q", previousAnnotationValue)
	}

	existingDriverMap := map[string]string{}
	if previousAnnotationValue != "" {
		// Parse previousAnnotationValue as JSON
		if err := json.Unmarshal([]byte(previousAnnotationValue), &existingDriverMap); err != nil {
			return node, fmt.Errorf(
				"failed to parse node's %q annotation value (%q) err=%v",
				annotationKey,
				previousAnnotationValue,
				err)
		}
	}

	if val, ok := existingDriverMap[driverName]; ok {
		if val == csiDriverNodeId {
			// Value already exists in node annotation, nothing more to do
			glog.V(2).Infof(
				"The key value {%q: %q} alredy eixst in node %q annotation, no need to update: %v",
				driverName,
				csiDriverNodeId,
				annotationKey,
				previousAnnotationValue)
			return node, nil
		}
	}

	// Add/update annotation value
	existingDriverMap[driverName] = csiDriverNodeId
	jsonObj, err := json.Marshal(existingDriverMap)
	if err != nil {
		return node, fmt.Errorf(
			"failed while trying to add key value {%q: %q} to node %q annotation. Existing value: %v",
			driverName,
			csiDriverNodeId,
			annotationKey,
			previousAnnotationValue)
	}

	node.ObjectMeta.Annotations = cloneAndAddAnnotation(
		node.ObjectMeta.Annotations,
		annotationKey,
		string(jsonObj))
	return node, nil
}

func addLabelsAndLimits(nodeName string, nodeClient corev1.NodeInterface, driverName string, csiDriverNodeId string, maxLimit int64) error {
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Retrieve the latest version of Node before attempting update, so that
		// existing changes are not overwritten. RetryOnConflict uses
		// exponential backoff to avoid exhausting the apiserver.
		node, getErr := nodeClient.Get(nodeName, metav1.GetOptions{})
		if getErr != nil {
			glog.Errorf("Failed to get latest version of Node: %v", getErr)
			return getErr // do not wrap error
		}
		var labelErr error
		node, labelErr = addNodeIdToNode(node, driverName, csiDriverNodeId)
		if labelErr != nil {
			return labelErr
		}
		node = addMaxAttachLimitToNode(node, driverName, maxLimit)

		_, updateErr := nodeClient.Update(node)
		if updateErr == nil {
			glog.V(2).Infof(
				"Updated node %q successfully for CSI driver %q and CSI node name %q",
				nodeName,
				driverName,
				csiDriverNodeId)
		}
		return updateErr // do not wrap error
	})
	if retryErr != nil {
		return fmt.Errorf("error setting attach limit and labels for %s with : %v", driverName, retryErr)
	}
	return nil
}

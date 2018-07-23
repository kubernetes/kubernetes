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

// Package labelmanager includes internal functions used to add/delete labels to
// kubernetes nodes for corresponding CSI drivers
package labelmanager // import "k8s.io/kubernetes/pkg/volume/csi/labelmanager"

import (
	"encoding/json"
	"fmt"

	"github.com/golang/glog"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"
	corev1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/util/retry"
)

const (
	// Name of node annotation that contains JSON map of driver names to node
	// names
	annotationKey = "csi.volume.kubernetes.io/nodeid"
	csiPluginName = "kubernetes.io/csi"
)

// labelManagementStruct is struct of channels used for communication between the driver registration
// code and the go routine responsible for managing the node's labels
type labelManagerStruct struct {
	nodeName types.NodeName
	k8s      kubernetes.Interface
}

// Interface implements an interface for managing labels of a node
type Interface interface {
	AddLabels(driverName string) error
}

// NewLabelManager initializes labelManagerStruct and returns available interfaces
func NewLabelManager(nodeName types.NodeName, kubeClient kubernetes.Interface) Interface {
	return labelManagerStruct{
		nodeName: nodeName,
		k8s:      kubeClient,
	}
}

// nodeLabelManager waits for labeling requests initiated by the driver's registration
// process.
func (lm labelManagerStruct) AddLabels(driverName string) error {
	err := verifyAndAddNodeId(string(lm.nodeName), lm.k8s.CoreV1().Nodes(), driverName, string(lm.nodeName))
	if err != nil {
		return fmt.Errorf("failed to update node %s's annotation with error: %+v", lm.nodeName, err)
	}
	return nil
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

func verifyAndAddNodeId(
	k8sNodeName string,
	k8sNodesClient corev1.NodeInterface,
	csiDriverName string,
	csiDriverNodeId string) error {
	// Add or update annotation on Node object
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Retrieve the latest version of Node before attempting update, so that
		// existing changes are not overwritten. RetryOnConflict uses
		// exponential backoff to avoid exhausting the apiserver.
		result, getErr := k8sNodesClient.Get(k8sNodeName, metav1.GetOptions{})
		if getErr != nil {
			glog.Errorf("Failed to get latest version of Node: %v", getErr)
			return getErr // do not wrap error
		}

		var previousAnnotationValue string
		if result.ObjectMeta.Annotations != nil {
			previousAnnotationValue =
				result.ObjectMeta.Annotations[annotationKey]
			glog.V(3).Infof(
				"previousAnnotationValue=%q", previousAnnotationValue)
		}

		existingDriverMap := map[string]string{}
		if previousAnnotationValue != "" {
			// Parse previousAnnotationValue as JSON
			if err := json.Unmarshal([]byte(previousAnnotationValue), &existingDriverMap); err != nil {
				return fmt.Errorf(
					"failed to parse node's %q annotation value (%q) err=%v",
					annotationKey,
					previousAnnotationValue,
					err)
			}
		}

		if val, ok := existingDriverMap[csiDriverName]; ok {
			if val == csiDriverNodeId {
				// Value already exists in node annotation, nothing more to do
				glog.V(2).Infof(
					"The key value {%q: %q} alredy eixst in node %q annotation, no need to update: %v",
					csiDriverName,
					csiDriverNodeId,
					annotationKey,
					previousAnnotationValue)
				return nil
			}
		}

		// Add/update annotation value
		existingDriverMap[csiDriverName] = csiDriverNodeId
		jsonObj, err := json.Marshal(existingDriverMap)
		if err != nil {
			return fmt.Errorf(
				"failed while trying to add key value {%q: %q} to node %q annotation. Existing value: %v",
				csiDriverName,
				csiDriverNodeId,
				annotationKey,
				previousAnnotationValue)
		}

		result.ObjectMeta.Annotations = cloneAndAddAnnotation(
			result.ObjectMeta.Annotations,
			annotationKey,
			string(jsonObj))
		_, updateErr := k8sNodesClient.Update(result)
		if updateErr == nil {
			glog.V(2).Infof(
				"Updated node %q successfully for CSI driver %q and CSI node name %q",
				k8sNodeName,
				csiDriverName,
				csiDriverNodeId)
		}
		return updateErr // do not wrap error
	})
	if retryErr != nil {
		return fmt.Errorf("node update failed: %v", retryErr)
	}
	return nil
}

// Fetches Kubernetes node API object corresponding to k8sNodeName.
// If the csiDriverName is present in the node annotation, it is removed.
func verifyAndDeleteNodeId(
	k8sNodeName string,
	k8sNodesClient corev1.NodeInterface,
	csiDriverName string) error {
	retryErr := retry.RetryOnConflict(retry.DefaultRetry, func() error {
		// Retrieve the latest version of Node before attempting update, so that
		// existing changes are not overwritten. RetryOnConflict uses
		// exponential backoff to avoid exhausting the apiserver.
		result, getErr := k8sNodesClient.Get(k8sNodeName, metav1.GetOptions{})
		if getErr != nil {
			glog.Errorf("failed to get latest version of Node: %v", getErr)
			return getErr // do not wrap error
		}

		var previousAnnotationValue string
		if result.ObjectMeta.Annotations != nil {
			previousAnnotationValue =
				result.ObjectMeta.Annotations[annotationKey]
			glog.V(3).Infof(
				"previousAnnotationValue=%q", previousAnnotationValue)
		}

		existingDriverMap := map[string]string{}
		if previousAnnotationValue == "" {
			// Value already exists in node annotation, nothing more to do
			glog.V(2).Infof(
				"The key %q does not exist in node %q annotation, no need to cleanup.",
				csiDriverName,
				annotationKey)
			return nil
		}

		// Parse previousAnnotationValue as JSON
		if err := json.Unmarshal([]byte(previousAnnotationValue), &existingDriverMap); err != nil {
			return fmt.Errorf(
				"failed to parse node's %q annotation value (%q) err=%v",
				annotationKey,
				previousAnnotationValue,
				err)
		}

		if _, ok := existingDriverMap[csiDriverName]; !ok {
			// Value already exists in node annotation, nothing more to do
			glog.V(2).Infof(
				"The key %q does not eixst in node %q annotation, no need to cleanup: %v",
				csiDriverName,
				annotationKey,
				previousAnnotationValue)
			return nil
		}

		// Add/update annotation value
		delete(existingDriverMap, csiDriverName)
		jsonObj, err := json.Marshal(existingDriverMap)
		if err != nil {
			return fmt.Errorf(
				"failed while trying to remove key %q from node %q annotation. Existing data: %v",
				csiDriverName,
				annotationKey,
				previousAnnotationValue)
		}

		result.ObjectMeta.Annotations = cloneAndAddAnnotation(
			result.ObjectMeta.Annotations,
			annotationKey,
			string(jsonObj))
		_, updateErr := k8sNodesClient.Update(result)
		if updateErr == nil {
			fmt.Printf(
				"Updated node %q annotation to remove CSI driver %q.",
				k8sNodeName,
				csiDriverName)
		}
		return updateErr // do not wrap error
	})
	if retryErr != nil {
		return fmt.Errorf("node update failed: %v", retryErr)
	}
	return nil
}

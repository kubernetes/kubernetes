/*
Copyright 2016 The Kubernetes Authors.

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

package utils

import (
	"fmt"
	"strings"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/klog"
)

const (
	retries = 5
)

func AddLabelsToNode(c clientset.Interface, nodeName string, labels map[string]string) error {
	tokens := make([]string, 0, len(labels))
	for k, v := range labels {
		tokens = append(tokens, "\""+k+"\":\""+v+"\"")
	}
	labelString := "{" + strings.Join(tokens, ",") + "}"
	patch := fmt.Sprintf(`{"metadata":{"labels":%v}}`, labelString)
	var err error
	for attempt := 0; attempt < retries; attempt++ {
		_, err = c.CoreV1().Nodes().Patch(nodeName, types.MergePatchType, []byte(patch))
		if err != nil {
			if !apierrors.IsConflict(err) {
				return err
			}
		} else {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	return err
}

// RemoveLabelOffNode is for cleaning up labels temporarily added to node,
// won't fail if target label doesn't exist or has been removed.
func RemoveLabelOffNode(c clientset.Interface, nodeName string, labelKeys []string) error {
	var node *v1.Node
	var err error
	for attempt := 0; attempt < retries; attempt++ {
		node, err = c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if node.Labels == nil {
			return nil
		}
		for _, labelKey := range labelKeys {
			if node.Labels == nil || len(node.Labels[labelKey]) == 0 {
				break
			}
			delete(node.Labels, labelKey)
		}
		_, err = c.CoreV1().Nodes().Update(node)
		if err != nil {
			if !apierrors.IsConflict(err) {
				return err
			} else {
				klog.V(2).Infof("Conflict when trying to remove a labels %v from %v", labelKeys, nodeName)
			}
		} else {
			break
		}
		time.Sleep(100 * time.Millisecond)
	}
	return err
}

// VerifyLabelsRemoved checks if Node for given nodeName does not have any of labels from labelKeys.
// Return non-nil error if it does.
func VerifyLabelsRemoved(c clientset.Interface, nodeName string, labelKeys []string) error {
	node, err := c.CoreV1().Nodes().Get(nodeName, metav1.GetOptions{})
	if err != nil {
		return err
	}
	for _, labelKey := range labelKeys {
		if node.Labels != nil && len(node.Labels[labelKey]) != 0 {
			return fmt.Errorf("Failed removing label " + labelKey + " of the node " + nodeName)
		}
	}
	return nil
}

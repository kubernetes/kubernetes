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

/*

NOTE: the contents of this file has been copied from k8s.io/kubernetes/pkg/controller
and k8s.io/kubernetes/pkg/util/taints. The reason for duplicating this code is to remove
dependencies to k8s.io/kubernetes in all the cloud providers. Once k8s.io/kubernetes/pkg/util/taints
is moved to an external repository, this file should be removed and replaced with that one.
*/

package helpers

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	clientretry "k8s.io/client-go/util/retry"
)

var updateTaintBackoff = wait.Backoff{
	Steps:    5,
	Duration: 100 * time.Millisecond,
	Jitter:   1.0,
}

// AddOrUpdateTaintOnNode add taints to the node. If taint was added into node, it'll issue API calls
// to update nodes; otherwise, no API calls. Return error if any.
func AddOrUpdateTaintOnNode(c clientset.Interface, nodeName string, taints ...*v1.Taint) error {
	if len(taints) == 0 {
		return nil
	}
	firstTry := true
	return clientretry.RetryOnConflict(updateTaintBackoff, func() error {
		var err error
		var oldNode *v1.Node
		// First we try getting node from the API server cache, as it's cheaper. If it fails
		// we get it from etcd to be sure to have fresh data.
		if firstTry {
			oldNode, err = c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{ResourceVersion: "0"})
			firstTry = false
		} else {
			oldNode, err = c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		}
		if err != nil {
			return err
		}

		var newNode *v1.Node
		oldNodeCopy := oldNode
		updated := false
		for _, taint := range taints {
			curNewNode, ok, err := addOrUpdateTaint(oldNodeCopy, taint)
			if err != nil {
				return fmt.Errorf("failed to update taint of node")
			}
			updated = updated || ok
			newNode = curNewNode
			oldNodeCopy = curNewNode
		}
		if !updated {
			return nil
		}
		return PatchNodeTaints(c, nodeName, oldNode, newNode)
	})
}

// PatchNodeTaints patches node's taints.
func PatchNodeTaints(c clientset.Interface, nodeName string, oldNode *v1.Node, newNode *v1.Node) error {
	oldData, err := json.Marshal(oldNode)
	if err != nil {
		return fmt.Errorf("failed to marshal old node %#v for node %q: %v", oldNode, nodeName, err)
	}

	newTaints := newNode.Spec.Taints
	newNodeClone := oldNode.DeepCopy()
	newNodeClone.Spec.Taints = newTaints
	newData, err := json.Marshal(newNodeClone)
	if err != nil {
		return fmt.Errorf("failed to marshal new node %#v for node %q: %v", newNodeClone, nodeName, err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
	if err != nil {
		return fmt.Errorf("failed to create patch for node %q: %v", nodeName, err)
	}

	_, err = c.CoreV1().Nodes().Patch(context.TODO(), nodeName, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
	return err
}

// addOrUpdateTaint tries to add a taint to annotations list. Returns a new copy of updated Node and true if something was updated
// false otherwise.
func addOrUpdateTaint(node *v1.Node, taint *v1.Taint) (*v1.Node, bool, error) {
	newNode := node.DeepCopy()
	nodeTaints := newNode.Spec.Taints

	var newTaints []v1.Taint
	updated := false
	for i := range nodeTaints {
		if taint.MatchTaint(&nodeTaints[i]) {
			if equality.Semantic.DeepEqual(*taint, nodeTaints[i]) {
				return newNode, false, nil
			}
			newTaints = append(newTaints, *taint)
			updated = true
			continue
		}

		newTaints = append(newTaints, nodeTaints[i])
	}

	if !updated {
		newTaints = append(newTaints, *taint)
	}

	newNode.Spec.Taints = newTaints
	return newNode, true, nil
}

// RemoveTaintOffNode is for cleaning up taints temporarily added to node,
// won't fail if target taint doesn't exist or has been removed.
// If passed a node it'll check if there's anything to be done, if taint is not present it won't issue
// any API calls.
func RemoveTaintOffNode(c clientset.Interface, nodeName string, node *v1.Node, taints ...*v1.Taint) error {
	if len(taints) == 0 {
		return nil
	}
	// Short circuit for limiting amount of API calls.
	if node != nil {
		match := false
		for _, taint := range taints {
			if taintExists(node.Spec.Taints, taint) {
				match = true
				break
			}
		}
		if !match {
			return nil
		}
	}

	firstTry := true
	return clientretry.RetryOnConflict(updateTaintBackoff, func() error {
		var err error
		var oldNode *v1.Node
		// First we try getting node from the API server cache, as it's cheaper. If it fails
		// we get it from etcd to be sure to have fresh data.
		if firstTry {
			oldNode, err = c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{ResourceVersion: "0"})
			firstTry = false
		} else {
			oldNode, err = c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		}
		if err != nil {
			return err
		}

		var newNode *v1.Node
		oldNodeCopy := oldNode
		updated := false
		for _, taint := range taints {
			curNewNode, ok, err := removeTaint(oldNodeCopy, taint)
			if err != nil {
				return fmt.Errorf("failed to remove taint of node")
			}
			updated = updated || ok
			newNode = curNewNode
			oldNodeCopy = curNewNode
		}
		if !updated {
			return nil
		}
		return PatchNodeTaints(c, nodeName, oldNode, newNode)
	})
}

// taintExists checks if the given taint exists in list of taints. Returns true if exists false otherwise.
func taintExists(taints []v1.Taint, taintToFind *v1.Taint) bool {
	for _, taint := range taints {
		if taint.MatchTaint(taintToFind) {
			return true
		}
	}
	return false
}

// removeTaint tries to remove a taint from annotations list. Returns a new copy of updated Node and true if something was updated
// false otherwise.
func removeTaint(node *v1.Node, taint *v1.Taint) (*v1.Node, bool, error) {
	newNode := node.DeepCopy()
	nodeTaints := newNode.Spec.Taints
	if len(nodeTaints) == 0 {
		return newNode, false, nil
	}

	if !taintExists(nodeTaints, taint) {
		return newNode, false, nil
	}

	newTaints, _ := deleteTaint(nodeTaints, taint)
	newNode.Spec.Taints = newTaints
	return newNode, true, nil
}

// deleteTaint removes all the taints that have the same key and effect to given taintToDelete.
func deleteTaint(taints []v1.Taint, taintToDelete *v1.Taint) ([]v1.Taint, bool) {
	newTaints := []v1.Taint{}
	deleted := false
	for i := range taints {
		if taintToDelete.MatchTaint(&taints[i]) {
			deleted = true
			continue
		}
		newTaints = append(newTaints, taints[i])
	}
	return newTaints, deleted
}

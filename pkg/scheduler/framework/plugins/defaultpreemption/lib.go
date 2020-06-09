/*
Copyright 2020 The Kubernetes Authors.

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

package defaultpreemption

import (
	"context"
	"encoding/json"
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/kubernetes"
)

// GetLivePod returns the latest version of a Pod from API server.
func GetLivePod(cs kubernetes.Interface, ns, name string) (*v1.Pod, error) {
	return cs.CoreV1().Pods(ns).Get(context.TODO(), name, metav1.GetOptions{})
}

// RemoveNominatedNodeName clears the ".Status.NominatedNodeName" field for the given pod.
func RemoveNominatedNodeName(cs kubernetes.Interface, pod *v1.Pod) error {
	if len(pod.Status.NominatedNodeName) == 0 {
		return nil
	}
	podCopy := pod.DeepCopy()
	podCopy.Status.NominatedNodeName = ""
	return patchPod(cs, pod, podCopy)
}

// TODO(Huang-Wei): de-duplicate this with pkg/scheduler/scheduler.go#patchPod()
func patchPod(client kubernetes.Interface, old *v1.Pod, new *v1.Pod) error {
	oldData, err := json.Marshal(old)
	if err != nil {
		return err
	}

	newData, err := json.Marshal(new)
	if err != nil {
		return err
	}
	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, &v1.Pod{})
	if err != nil {
		return fmt.Errorf("failed to create merge patch for pod %q/%q: %v", old.Namespace, old.Name, err)
	}
	_, err = client.CoreV1().Pods(old.Namespace).Patch(context.TODO(), old.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status")
	return err
}

// DeleteVictim deletes the given victim from API server.
func DeleteVictim(cs kubernetes.Interface, victim *v1.Pod) error {
	return cs.CoreV1().Pods(victim.Namespace).Delete(context.TODO(), victim.Name, metav1.DeleteOptions{})
}

/*
Copyright 2021 The Kubernetes Authors.

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

// Package ephemeral provides code that supports the usual pattern
// for accessing the PVC that provides a generic ephemeral inline volume:
//
// - determine the PVC name that corresponds to the inline volume source
// - retrieve the PVC
// - verify that the PVC is owned by the pod
// - use the PVC
package ephemeral

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// VolumeClaimName returns the name of the PersistentVolumeClaim
// object that gets created for the generic ephemeral inline volume. The
// name is deterministic and therefore this function does not need any
// additional information besides the Pod name and volume name and it
// will never fail.
//
// Before using the PVC for the Pod, the caller must check that it is
// indeed the PVC that was created for the Pod by calling IsUsable.
func VolumeClaimName(pod *v1.Pod, volume *v1.Volume) string {
	return pod.Name + "-" + volume.Name
}

// VolumeIsForPod checks that the PVC is the ephemeral volume that
// was created for the Pod. It returns an error that is informative
// enough to be returned by the caller without adding further details
// about the Pod or PVC.
func VolumeIsForPod(pod *v1.Pod, pvc *v1.PersistentVolumeClaim) error {
	// Checking the namespaces is just a precaution. The caller should
	// never pass in a PVC that isn't from the same namespace as the
	// Pod.
	if pvc.Namespace != pod.Namespace || !metav1.IsControlledBy(pvc, pod) {
		return fmt.Errorf("PVC %s/%s was not created for pod %s/%s (pod is not owner)", pvc.Namespace, pvc.Name, pod.Namespace, pod.Name)
	}
	return nil
}

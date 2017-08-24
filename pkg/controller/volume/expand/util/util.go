/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"fmt"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
)

// ClaimToClaimKey return namespace/name string for pvc
func ClaimToClaimKey(claim *v1.PersistentVolumeClaim) string {
	return fmt.Sprintf("%s/%s", claim.Namespace, claim.Name)
}

// UpdatePVCCondition updates pvc with given condition status
func UpdatePVCCondition(pvc *v1.PersistentVolumeClaim,
	pvcConditions []v1.PersistentVolumeClaimCondition,
	kubeClient clientset.Interface) (*v1.PersistentVolumeClaim, error) {

	claimClone, err := ClonePVC(pvc)
	if err != nil {
		return nil, err
	}
	claimClone.Status.Conditions = pvcConditions
	updatedClaim, updateErr := kubeClient.CoreV1().PersistentVolumeClaims(claimClone.Namespace).UpdateStatus(claimClone)
	if updateErr != nil {
		glog.V(4).Infof("updating PersistentVolumeClaim[%s] status: failed: %v", ClaimToClaimKey(pvc), updateErr)
		return nil, updateErr
	}
	return updatedClaim, nil
}

// ClonePVC returns a clone of PVC to be used while updating it
func ClonePVC(oldPvc *v1.PersistentVolumeClaim) (*v1.PersistentVolumeClaim, error) {
	clone, err := scheme.Scheme.DeepCopy(oldPvc)

	if err != nil {
		return nil, fmt.Errorf("Error cloning claim %s : %v", ClaimToClaimKey(oldPvc), err)
	}

	claimClone, ok := clone.(*v1.PersistentVolumeClaim)

	if !ok {
		return nil, fmt.Errorf("Unexpected claim cast error : %v", claimClone)
	}
	return claimClone, nil

}

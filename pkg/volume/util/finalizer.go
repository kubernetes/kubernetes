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
	"k8s.io/api/core/v1"
)

const (
	// Name of finalizer on PVCs that have a running pod.
	PVCProtectionFinalizer = "kubernetes.io/pvc-protection"
)

// IsPVCBeingDeleted returns:
// true: in case PVC is being deleted, i.e. ObjectMeta.DeletionTimestamp is set
// false: in case PVC is not being deleted, i.e. ObjectMeta.DeletionTimestamp is nil
func IsPVCBeingDeleted(pvc *v1.PersistentVolumeClaim) bool {
	return pvc.ObjectMeta.DeletionTimestamp != nil
}

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

package persistentvolume

import (
	"k8s.io/kubernetes/pkg/api"
)

func getClaimRefNamespace(pv *api.PersistentVolume) string {
	if pv.Spec.ClaimRef != nil {
		return pv.Spec.ClaimRef.Namespace
	}
	return ""
}

// VisitPVSecretNames invokes the visitor function with the name of every secret
// referenced by the PV spec. If visitor returns false, visiting is short-circuited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPVSecretNames(pv *api.PersistentVolume, visitor func(string, string) bool) bool {
	source := &pv.Spec.PersistentVolumeSource
	switch {
	case source.AzureFile != nil:
		if len(source.AzureFile.SecretName) > 0 && !visitor(getClaimRefNamespace(pv), source.AzureFile.SecretName) {
			return false
		}
		return true
	case source.CephFS != nil:
		if source.CephFS.SecretRef != nil && !visitor(getClaimRefNamespace(pv), source.CephFS.SecretRef.Name) {
			return false
		}
	case source.FlexVolume != nil:
		if source.FlexVolume.SecretRef != nil && !visitor(getClaimRefNamespace(pv), source.FlexVolume.SecretRef.Name) {
			return false
		}
	case source.RBD != nil:
		if source.RBD.SecretRef != nil && !visitor(getClaimRefNamespace(pv), source.RBD.SecretRef.Name) {
			return false
		}
	case source.ScaleIO != nil:
		if source.ScaleIO.SecretRef != nil && !visitor(getClaimRefNamespace(pv), source.ScaleIO.SecretRef.Name) {
			return false
		}
	case source.ISCSI != nil:
		if source.ISCSI.SecretRef != nil && !visitor(getClaimRefNamespace(pv), source.ISCSI.SecretRef.Name) {
			return false
		}
	case source.StorageOS != nil:
		if source.StorageOS.SecretRef != nil && !visitor(source.StorageOS.SecretRef.Namespace, source.StorageOS.SecretRef.Name) {
			return false
		}
	}
	return true
}

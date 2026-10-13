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
	corev1 "k8s.io/api/core/v1"
)

func getClaimRefNamespace(pv *corev1.PersistentVolume) string {
	if pv.Spec.ClaimRef != nil {
		return pv.Spec.ClaimRef.Namespace
	}
	return ""
}

// Visitor is called with each object's namespace and name, and returns true if visiting should continue
type Visitor func(namespace, name string, kubeletVisible bool) (shouldContinue bool)

func skipEmptyNames(visitor Visitor) Visitor {
	return func(namespace, name string, kubeletVisible bool) bool {
		if len(name) == 0 {
			// continue visiting
			return true
		}
		// delegate to visitor
		return visitor(namespace, name, kubeletVisible)
	}
}

// VisitPVSecretNames invokes the visitor function with the name of every secret
// referenced by the PV spec. If visitor returns false, visiting is short-circuited.
// Returns true if visiting completed, false if visiting was short-circuited.
func VisitPVSecretNames(pv *corev1.PersistentVolume, visitor Visitor) bool {
	visitor = skipEmptyNames(visitor)
	source := &pv.Spec.PersistentVolumeSource
	switch {
	case source.AzureFile != nil:
		if source.AzureFile.SecretNamespace != nil && len(*source.AzureFile.SecretNamespace) > 0 {
			if len(source.AzureFile.SecretName) > 0 && !visitor(*source.AzureFile.SecretNamespace, source.AzureFile.SecretName, true /* kubeletVisible */) {
				return false
			}
		} else {
			if len(source.AzureFile.SecretName) > 0 && !visitor(getClaimRefNamespace(pv), source.AzureFile.SecretName, true /* kubeletVisible */) {
				return false
			}
		}
		return true
	case source.CephFS != nil:
		if source.CephFS.SecretRef != nil {
			// previously persisted PV objects use claimRef namespace
			ns := getClaimRefNamespace(pv)
			if len(source.CephFS.SecretRef.Namespace) > 0 {
				// use the secret namespace if namespace is set
				ns = source.CephFS.SecretRef.Namespace
			}
			if !visitor(ns, source.CephFS.SecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	case source.Cinder != nil:
		if source.Cinder.SecretRef != nil && !visitor(source.Cinder.SecretRef.Namespace, source.Cinder.SecretRef.Name, true /* kubeletVisible */) {
			return false
		}
	case source.FlexVolume != nil:
		if source.FlexVolume.SecretRef != nil {
			// previously persisted PV objects use claimRef namespace
			ns := getClaimRefNamespace(pv)
			if len(source.FlexVolume.SecretRef.Namespace) > 0 {
				// use the secret namespace if namespace is set
				ns = source.FlexVolume.SecretRef.Namespace
			}
			if !visitor(ns, source.FlexVolume.SecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	case source.RBD != nil:
		if source.RBD.SecretRef != nil {
			// previously persisted PV objects use claimRef namespace
			ns := getClaimRefNamespace(pv)
			if len(source.RBD.SecretRef.Namespace) > 0 {
				// use the secret namespace if namespace is set
				ns = source.RBD.SecretRef.Namespace
			}
			if !visitor(ns, source.RBD.SecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	case source.ScaleIO != nil:
		if source.ScaleIO.SecretRef != nil {
			ns := getClaimRefNamespace(pv)
			if source.ScaleIO.SecretRef != nil && len(source.ScaleIO.SecretRef.Namespace) > 0 {
				ns = source.ScaleIO.SecretRef.Namespace
			}
			if !visitor(ns, source.ScaleIO.SecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	case source.ISCSI != nil:
		if source.ISCSI.SecretRef != nil {
			// previously persisted PV objects use claimRef namespace
			ns := getClaimRefNamespace(pv)
			if len(source.ISCSI.SecretRef.Namespace) > 0 {
				// use the secret namespace if namespace is set
				ns = source.ISCSI.SecretRef.Namespace
			}
			if !visitor(ns, source.ISCSI.SecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	case source.StorageOS != nil:
		if source.StorageOS.SecretRef != nil && !visitor(source.StorageOS.SecretRef.Namespace, source.StorageOS.SecretRef.Name, true /* kubeletVisible */) {
			return false
		}
	case source.CSI != nil:
		if source.CSI.ControllerPublishSecretRef != nil {
			if !visitor(source.CSI.ControllerPublishSecretRef.Namespace, source.CSI.ControllerPublishSecretRef.Name, false /* kubeletVisible */) {
				return false
			}
		}
		if source.CSI.ControllerExpandSecretRef != nil {
			if !visitor(source.CSI.ControllerExpandSecretRef.Namespace, source.CSI.ControllerExpandSecretRef.Name, false /* kubeletVisible */) {
				return false
			}
		}

		if source.CSI.NodePublishSecretRef != nil {
			if !visitor(source.CSI.NodePublishSecretRef.Namespace, source.CSI.NodePublishSecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
		if source.CSI.NodeStageSecretRef != nil {
			if !visitor(source.CSI.NodeStageSecretRef.Namespace, source.CSI.NodeStageSecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
		if source.CSI.NodeExpandSecretRef != nil {
			if !visitor(source.CSI.NodeExpandSecretRef.Namespace, source.CSI.NodeExpandSecretRef.Name, true /* kubeletVisible */) {
				return false
			}
		}
	}
	return true
}

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
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestPVSecrets(t *testing.T) {
	// Stub containing all possible secret references in a PV.
	// The names of the referenced secrets match struct paths detected by reflection.
	secretNamespace := "Spec.PersistentVolumeSource.AzureFile.SecretNamespace"
	pvs := []*corev1.PersistentVolume{
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				AzureFile: &corev1.AzureFilePersistentVolumeSource{
					SecretName: "Spec.PersistentVolumeSource.AzureFile.SecretName"}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				AzureFile: &corev1.AzureFilePersistentVolumeSource{
					SecretName:      "Spec.PersistentVolumeSource.AzureFile.SecretName",
					SecretNamespace: &secretNamespace}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CephFS: &corev1.CephFSPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.CephFS.SecretRef",
						Namespace: "cephfs"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CephFS: &corev1.CephFSPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name: "Spec.PersistentVolumeSource.CephFS.SecretRef"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				Cinder: &corev1.CinderPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.Cinder.SecretRef",
						Namespace: "cinder"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				FlexVolume: &corev1.FlexPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.FlexVolume.SecretRef",
						Namespace: "flexns"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				FlexVolume: &corev1.FlexPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name: "Spec.PersistentVolumeSource.FlexVolume.SecretRef"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				RBD: &corev1.RBDPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name: "Spec.PersistentVolumeSource.RBD.SecretRef"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				RBD: &corev1.RBDPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.RBD.SecretRef",
						Namespace: "rbdns"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				ScaleIO: &corev1.ScaleIOPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name: "Spec.PersistentVolumeSource.ScaleIO.SecretRef"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				ScaleIO: &corev1.ScaleIOPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.ScaleIO.SecretRef",
						Namespace: "scaleions"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				ISCSI: &corev1.ISCSIPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.ISCSI.SecretRef",
						Namespace: "iscsi"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				ISCSI: &corev1.ISCSIPersistentVolumeSource{
					SecretRef: &corev1.SecretReference{
						Name: "Spec.PersistentVolumeSource.ISCSI.SecretRef"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				StorageOS: &corev1.StorageOSPersistentVolumeSource{
					SecretRef: &corev1.ObjectReference{
						Name:      "Spec.PersistentVolumeSource.StorageOS.SecretRef",
						Namespace: "storageosns"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CSI: &corev1.CSIPersistentVolumeSource{
					ControllerPublishSecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.CSI.ControllerPublishSecretRef",
						Namespace: "csi"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CSI: &corev1.CSIPersistentVolumeSource{
					NodePublishSecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.CSI.NodePublishSecretRef",
						Namespace: "csi"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CSI: &corev1.CSIPersistentVolumeSource{
					NodeStageSecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.CSI.NodeStageSecretRef",
						Namespace: "csi"}}}}},
		{Spec: corev1.PersistentVolumeSpec{
			ClaimRef: &corev1.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: corev1.PersistentVolumeSource{
				CSI: &corev1.CSIPersistentVolumeSource{
					ControllerExpandSecretRef: &corev1.SecretReference{
						Name:      "Spec.PersistentVolumeSource.CSI.ControllerExpandSecretRef",
						Namespace: "csi"}}}}},
	}
	extractedNames := sets.NewString()
	extractedNamesWithNamespace := sets.NewString()

	for _, pv := range pvs {
		VisitPVSecretNames(pv, func(namespace, name string, kubeletVisible bool) bool {
			extractedNames.Insert(name)
			extractedNamesWithNamespace.Insert(namespace + "/" + name)
			return true
		})
	}

	// excludedSecretPaths holds struct paths to fields with "secret" in the name that are not actually references to secret API objects
	excludedSecretPaths := sets.NewString(
		"Spec.PersistentVolumeSource.CephFS.SecretFile",
		"Spec.PersistentVolumeSource.AzureFile.SecretNamespace",
	)
	// expectedSecretPaths holds struct paths to fields with "secret" in the name that are references to secret API objects.
	// every path here should be represented as an example in the PV stub above, with the secret name set to the path.
	expectedSecretPaths := sets.NewString(
		"Spec.PersistentVolumeSource.AzureFile.SecretName",
		"Spec.PersistentVolumeSource.CephFS.SecretRef",
		"Spec.PersistentVolumeSource.Cinder.SecretRef",
		"Spec.PersistentVolumeSource.FlexVolume.SecretRef",
		"Spec.PersistentVolumeSource.RBD.SecretRef",
		"Spec.PersistentVolumeSource.ScaleIO.SecretRef",
		"Spec.PersistentVolumeSource.ISCSI.SecretRef",
		"Spec.PersistentVolumeSource.StorageOS.SecretRef",
		"Spec.PersistentVolumeSource.CSI.ControllerPublishSecretRef",
		"Spec.PersistentVolumeSource.CSI.NodePublishSecretRef",
		"Spec.PersistentVolumeSource.CSI.NodeStageSecretRef",
		"Spec.PersistentVolumeSource.CSI.ControllerExpandSecretRef",
	)
	secretPaths := collectSecretPaths(t, nil, "", reflect.TypeOf(&api.PersistentVolume{}))
	secretPaths = secretPaths.Difference(excludedSecretPaths)
	if missingPaths := expectedSecretPaths.Difference(secretPaths); len(missingPaths) > 0 {
		t.Logf("Missing expected secret paths:\n%s", strings.Join(missingPaths.List(), "\n"))
		t.Error("Missing expected secret paths. Verify VisitPVSecretNames() is correctly finding the missing paths, then correct expectedSecretPaths")
	}
	if extraPaths := secretPaths.Difference(expectedSecretPaths); len(extraPaths) > 0 {
		t.Logf("Extra secret paths:\n%s", strings.Join(extraPaths.List(), "\n"))
		t.Error("Extra fields with 'secret' in the name found. Verify VisitPVSecretNames() is including these fields if appropriate, then correct expectedSecretPaths")
	}

	if missingNames := expectedSecretPaths.Difference(extractedNames); len(missingNames) > 0 {
		t.Logf("Missing expected secret names:\n%s", strings.Join(missingNames.List(), "\n"))
		t.Error("Missing expected secret names. Verify the PV stub above includes these references, then verify VisitPVSecretNames() is correctly finding the missing names")
	}
	if extraNames := extractedNames.Difference(expectedSecretPaths); len(extraNames) > 0 {
		t.Logf("Extra secret names:\n%s", strings.Join(extraNames.List(), "\n"))
		t.Error("Extra secret names extracted. Verify VisitPVSecretNames() is correctly extracting secret names")
	}

	expectedNamespacedNames := sets.NewString(
		"claimrefns/Spec.PersistentVolumeSource.AzureFile.SecretName",
		"Spec.PersistentVolumeSource.AzureFile.SecretNamespace/Spec.PersistentVolumeSource.AzureFile.SecretName",

		"claimrefns/Spec.PersistentVolumeSource.CephFS.SecretRef",
		"cephfs/Spec.PersistentVolumeSource.CephFS.SecretRef",

		"cinder/Spec.PersistentVolumeSource.Cinder.SecretRef",

		"claimrefns/Spec.PersistentVolumeSource.FlexVolume.SecretRef",
		"flexns/Spec.PersistentVolumeSource.FlexVolume.SecretRef",

		"claimrefns/Spec.PersistentVolumeSource.RBD.SecretRef",
		"rbdns/Spec.PersistentVolumeSource.RBD.SecretRef",

		"claimrefns/Spec.PersistentVolumeSource.ScaleIO.SecretRef",
		"scaleions/Spec.PersistentVolumeSource.ScaleIO.SecretRef",

		"claimrefns/Spec.PersistentVolumeSource.ISCSI.SecretRef",
		"iscsi/Spec.PersistentVolumeSource.ISCSI.SecretRef",

		"storageosns/Spec.PersistentVolumeSource.StorageOS.SecretRef",

		"csi/Spec.PersistentVolumeSource.CSI.ControllerPublishSecretRef",
		"csi/Spec.PersistentVolumeSource.CSI.NodePublishSecretRef",
		"csi/Spec.PersistentVolumeSource.CSI.NodeStageSecretRef",
		"csi/Spec.PersistentVolumeSource.CSI.ControllerExpandSecretRef",
	)
	if missingNames := expectedNamespacedNames.Difference(extractedNamesWithNamespace); len(missingNames) > 0 {
		t.Logf("Missing expected namespaced names:\n%s", strings.Join(missingNames.List(), "\n"))
		t.Error("Missing expected namespaced names. Verify the PV stub above includes these references, then verify VisitPVSecretNames() is correctly finding the missing names")
	}
	if extraNames := extractedNamesWithNamespace.Difference(expectedNamespacedNames); len(extraNames) > 0 {
		t.Logf("Extra namespaced names:\n%s", strings.Join(extraNames.List(), "\n"))
		t.Error("Extra namespaced names extracted. Verify VisitPVSecretNames() is correctly extracting secret names")
	}
}

// collectSecretPaths traverses the object, computing all the struct paths that lead to fields with "secret" in the name.
func collectSecretPaths(t *testing.T, path *field.Path, name string, tp reflect.Type) sets.String {
	secretPaths := sets.NewString()

	if tp.Kind() == reflect.Ptr {
		secretPaths.Insert(collectSecretPaths(t, path, name, tp.Elem()).List()...)
		return secretPaths
	}

	if strings.Contains(strings.ToLower(name), "secret") {
		secretPaths.Insert(path.String())
	}

	switch tp.Kind() {
	case reflect.Ptr:
		secretPaths.Insert(collectSecretPaths(t, path, name, tp.Elem()).List()...)
	case reflect.Struct:
		// ObjectMeta should not have any field with the word "secret" in it;
		// it contains cycles so it's easiest to just skip it.
		if name == "ObjectMeta" {
			break
		}
		for i := 0; i < tp.NumField(); i++ {
			field := tp.Field(i)
			secretPaths.Insert(collectSecretPaths(t, path.Child(field.Name), field.Name, field.Type).List()...)
		}
	case reflect.Interface:
		t.Errorf("cannot find secret fields in interface{} field %s", path.String())
	case reflect.Map:
		secretPaths.Insert(collectSecretPaths(t, path.Key("*"), "", tp.Elem()).List()...)
	case reflect.Slice:
		secretPaths.Insert(collectSecretPaths(t, path.Key("*"), "", tp.Elem()).List()...)
	default:
		// all primitive types
	}

	return secretPaths
}

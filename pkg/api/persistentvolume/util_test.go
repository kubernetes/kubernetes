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
	"testing"

	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/kubernetes/pkg/api"
)

func TestPVSecrets(t *testing.T) {
	// Stub containing all possible secret references in a PV.
	// The names of the referenced secrets match struct paths detected by reflection.
	pvs := []*api.PersistentVolume{
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				AzureFile: &api.AzureFileVolumeSource{
					SecretName: "Spec.PersistentVolumeSource.AzureFile.SecretName"}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				CephFS: &api.CephFSVolumeSource{
					SecretRef: &api.LocalObjectReference{
						Name: "Spec.PersistentVolumeSource.CephFS.SecretRef"}}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				FlexVolume: &api.FlexVolumeSource{
					SecretRef: &api.LocalObjectReference{
						Name: "Spec.PersistentVolumeSource.FlexVolume.SecretRef"}}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				RBD: &api.RBDVolumeSource{
					SecretRef: &api.LocalObjectReference{
						Name: "Spec.PersistentVolumeSource.RBD.SecretRef"}}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				ScaleIO: &api.ScaleIOVolumeSource{
					SecretRef: &api.LocalObjectReference{
						Name: "Spec.PersistentVolumeSource.ScaleIO.SecretRef"}}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				ISCSI: &api.ISCSIVolumeSource{
					SecretRef: &api.LocalObjectReference{
						Name: "Spec.PersistentVolumeSource.ISCSI.SecretRef"}}}}},
		{Spec: api.PersistentVolumeSpec{
			ClaimRef: &api.ObjectReference{Namespace: "claimrefns", Name: "claimrefname"},
			PersistentVolumeSource: api.PersistentVolumeSource{
				StorageOS: &api.StorageOSPersistentVolumeSource{
					SecretRef: &api.ObjectReference{
						Name:      "Spec.PersistentVolumeSource.StorageOS.SecretRef",
						Namespace: "storageosns"}}}}},
	}

	extractedNames := sets.NewString()
	extractedNamesWithNamespace := sets.NewString()
	for _, pv := range pvs {
		VisitPVSecretNames(pv, func(namespace, name string) bool {
			extractedNames.Insert(name)
			extractedNamesWithNamespace.Insert(namespace + "/" + name)
			return true
		})
	}

	// excludedSecretPaths holds struct paths to fields with "secret" in the name that are not actually references to secret API objects
	excludedSecretPaths := sets.NewString(
		"Spec.PersistentVolumeSource.CephFS.SecretFile",
	)
	// expectedSecretPaths holds struct paths to fields with "secret" in the name that are references to secret API objects.
	// every path here should be represented as an example in the PV stub above, with the secret name set to the path.
	expectedSecretPaths := sets.NewString(
		"Spec.PersistentVolumeSource.AzureFile.SecretName",
		"Spec.PersistentVolumeSource.CephFS.SecretRef",
		"Spec.PersistentVolumeSource.FlexVolume.SecretRef",
		"Spec.PersistentVolumeSource.RBD.SecretRef",
		"Spec.PersistentVolumeSource.ScaleIO.SecretRef",
		"Spec.PersistentVolumeSource.ISCSI.SecretRef",
		"Spec.PersistentVolumeSource.StorageOS.SecretRef",
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
		"claimrefns/Spec.PersistentVolumeSource.CephFS.SecretRef",
		"claimrefns/Spec.PersistentVolumeSource.FlexVolume.SecretRef",
		"claimrefns/Spec.PersistentVolumeSource.RBD.SecretRef",
		"claimrefns/Spec.PersistentVolumeSource.ScaleIO.SecretRef",
		"claimrefns/Spec.PersistentVolumeSource.ISCSI.SecretRef",
		"storageosns/Spec.PersistentVolumeSource.StorageOS.SecretRef",
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

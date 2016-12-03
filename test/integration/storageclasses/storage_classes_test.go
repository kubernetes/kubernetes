// +build integration,!no-etcd

/*
Copyright 2015 The Kubernetes Authors.

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

package storageclasses

// This file contains tests for the storage classes API resource.

import (
	"testing"

	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	storage "k8s.io/kubernetes/pkg/apis/storage/v1beta1"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/v1beta1/util"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/test/integration/framework"
)

const provisionerPluginName = "kubernetes.io/mock-provisioner"

// TestStorageClasses tests apiserver-side behavior of creation of storage class objects and their use by pvcs.
func TestStorageClasses(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	client := clientset.NewForConfigOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})

	ns := framework.CreateTestingNamespace("storageclass", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	DoTestStorageClasses(t, client, ns)
}

// DoTestStorageClasses tests storage classes for one api version.
func DoTestStorageClasses(t *testing.T, client clientset.Interface, ns *v1.Namespace) {
	// Make a storage class object.
	s := storage.StorageClass{
		TypeMeta: metav1.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: v1.ObjectMeta{
			Name: "gold",
		},
		Provisioner: provisionerPluginName,
	}

	if _, err := client.Storage().StorageClasses().Create(&s); err != nil {
		t.Errorf("unable to create test storage class: %v", err)
	}
	defer deleteStorageClassOrErrorf(t, client, s.Namespace, s.Name)

	// Template for pvcs that specify a storage class
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: v1.ObjectMeta{
			Name:      "XXX",
			Namespace: ns.Name,
			Annotations: map[string]string{
				storageutil.StorageClassAnnotation: "gold",
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources:   v1.ResourceRequirements{Requests: v1.ResourceList{v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G")}},
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
		},
	}

	pvc.ObjectMeta.Name = "uses-storageclass"
	if _, err := client.Core().PersistentVolumeClaims(ns.Name).Create(pvc); err != nil {
		t.Errorf("Failed to create pvc: %v", err)
	}
	defer deletePersistentVolumeClaimOrErrorf(t, client, ns.Name, pvc.Name)
}

func deleteStorageClassOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.Storage().StorageClasses().Delete(name, nil); err != nil {
		t.Errorf("unable to delete storage class %v: %v", name, err)
	}
}

func deletePersistentVolumeClaimOrErrorf(t *testing.T, c clientset.Interface, ns, name string) {
	if err := c.Core().PersistentVolumeClaims(ns).Delete(name, nil); err != nil {
		t.Errorf("unable to delete persistent volume claim %v: %v", name, err)
	}
}

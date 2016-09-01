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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/client/restclient"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/integration/framework"
)

const provisionerPluginName = "kubernetes.io/mock-provisioner"

// TestStorageClasses tests apiserver-side behavior of creation of storage class objects and their use by pvcs.
func TestStorageClasses(t *testing.T) {
	_, s := framework.RunAMaster(nil)
	defer s.Close()

	client := client.NewOrDie(&restclient.Config{Host: s.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Default.GroupVersion()}})

	ns := framework.CreateTestingNamespace("storageclass", s, t)
	defer framework.DeleteTestingNamespace(ns, s, t)

	DoTestStorageClasses(t, client, ns)
}

// DoTestStorageClasses tests storage classes for one api version.
func DoTestStorageClasses(t *testing.T, client *client.Client, ns *api.Namespace) {
	// Make a storage class object.
	s := storage.StorageClass{
		TypeMeta: unversioned.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: api.ObjectMeta{
			Name: "gold",
		},
		Provisioner: provisionerPluginName,
	}

	if _, err := client.Storage().StorageClasses().Create(&s); err != nil {
		t.Errorf("unable to create test storage class: %v", err)
	}
	defer deleteStorageClassOrErrorf(t, client, s.Namespace, s.Name)

	// Template for pvcs that specify a storage class
	pvc := &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "XXX",
			Namespace: ns.Name,
			Annotations: map[string]string{
				"volume.beta.kubernetes.io/storage-class": "gold",
			},
		},
		Spec: api.PersistentVolumeClaimSpec{
			Resources:   api.ResourceRequirements{Requests: api.ResourceList{api.ResourceName(api.ResourceStorage): resource.MustParse("1G")}},
			AccessModes: []api.PersistentVolumeAccessMode{api.ReadWriteOnce},
		},
	}

	pvc.ObjectMeta.Name = "uses-storageclass"
	if _, err := client.PersistentVolumeClaims(ns.Name).Create(pvc); err != nil {
		t.Errorf("Failed to create pvc: %v", err)
	}
	defer deletePersistentVolumeClaimOrErrorf(t, client, ns.Name, pvc.Name)
}

func deleteStorageClassOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.Storage().StorageClasses().Delete(name); err != nil {
		t.Errorf("unable to delete storage class %v: %v", name, err)
	}
}

func deletePersistentVolumeClaimOrErrorf(t *testing.T, c *client.Client, ns, name string) {
	if err := c.PersistentVolumeClaims(ns).Delete(name); err != nil {
		t.Errorf("unable to delete persistent volume claim %v: %v", name, err)
	}
}

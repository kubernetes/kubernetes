/*
Copyright 2023 The Kubernetes Authors.

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

package storage

import (
	"context"

	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2econformance "k8s.io/kubernetes/test/e2e/framework/conformance"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

var _ = utils.SIGDescribe("StorageClass", func() {
	f := framework.NewDefaultFramework("storageclass-test")
	f.NamespacePodSecurityLevel = admissionapi.LevelRestricted

	/*
	   Release: v1.29
	   Testname: CRUD operations for StorageClasses
	   Description: kube-apiserver must support create/get/list/update/patch/delete operations for storage.k8s.io/v1 StorageClass.
	*/
	framework.ConformanceIt("StorageClass API availability", func(ctx context.Context) {
		e2econformance.TestResource(ctx, f,
			&e2econformance.ResourceTestcase[*storagev1.StorageClass]{
				GVR:        storagev1.SchemeGroupVersion.WithResource("storageclasses"),
				Namespaced: ptr.To(false),
				InitialSpec: &storagev1.StorageClass{
					Provisioner: "e2e-fake-provisioner",
					Parameters:  map[string]string{"foo": "bar"},
				},
				UpdateSpec: func(obj *storagev1.StorageClass) *storagev1.StorageClass {
					obj.MountOptions = []string{"ro", "soft"}
					return obj
				},
				StrategicMergePatchSpec: `{"metadata": {"labels": {"foo": "bar"}}}`,
			},
		)
	})
})
